#!/usr/bin/env python3
"""Rule-based episode quality scan for LeRobot v3 buckets.

Computes cheap per-episode metrics, flags episodes that break a rule, and
(with ``--apply``) merges them into each bucket's ``meta/excluded_episodes.json``
(or with ``--delete`` physically removes them from converter-layout buckets)
— the blacklist every LeRobotV3Reader and stats tool already honors. Raw data
is never modified. Regenerate normalization stats after applying.

Rules (thresholds are CLI flags; inspect ``quality.parquet`` before applying):

* ``too_short`` / ``too_long``   length < ``--min-seconds`` or > ``--max-len-x-median`` × bucket median
* ``nonfinite``                  NaN/inf in a pose / effector column
* ``zero_pose``                  any frame with an arm at exactly xyz = 0 (tracking loss)
* ``bad_image``                  converter replaced undecodable source frames (``bad_image_frames``)
* ``pos_jump``                   per-step EEF xyz move > ``--max-step-m``
* ``rot_jump``                   per-step EEF rotation > ``--max-step-deg``
* ``invalid_rot6d``              rot6d columns far from orthonormal
* ``static``                     EEF path < ``--min-path-m`` and effector range < ``--min-effector-range``
* ``bad_prompt``                 empty / too short / placeholder instruction, a template slot left
  empty ("move the  to the box"), a repeated word ("with with"), a file-name slug
  ("fold_mat", "hit-ball-with-gripper"), or an asset ID ("microwave_gr", "Galbot_G1_…_new1");
  see ``prompt_text``. Readers with ``normalize_prompt: true`` clean the last two at load time,
  so ``--apply`` on such a dataset drops episodes the reader could still use
* ``official_unqualified``       source annotators marked frames/episode unqualified
  (``unqualified_frames`` / ``coarse_quality`` written by ``galaxea_convert``)
* ``bag_unqualified``            source per-recording automatic check failed (``bag_quality`` = ``不合格``)
* ``body_motion``                (opt-in) base command share > ``--max-chassis-cmd-frac`` or torso
  joint range > ``--max-torso-range``; keeps tabletop-only episodes
* ``black_video`` / ``frozen_video`` / ``bad_video``   (``--video``) sampled head frames
  dark, identical, or undecodable

    python -m openwam.dataloader.utils.episode_quality \
        --root /root/AgiBotWorld-Beta-lerobotv3 --out /root/agibot_quality --video --workers 48
    # review, then rerun with --apply

Mobile-base statistics (``chassis_cmd_frac`` = share of frames with a nonzero
``action.chassis.velocities`` command, ``torso_range`` = largest per-joint range
of ``observation.state.torso``) are always recorded; they only flag episodes
when the ``body_motion`` thresholds are given.

Single-arm Franka buckets are scanned from DROID's ``observation.state.cartesian_position``
(xyz + euler) / ``gripper_position`` or LIBERO's 8-D ``observation.state`` (xyz + axis-angle +
two fingers); the pose is converted to one xyz + rot6d arm, so all rules apply unchanged.

Buckets without ``ee_base`` but with InternData-A1's ``states.{left,right}_ee_to_robot_pose``
(xyz + quaternion wxyz) are scanned from those columns, with grippers scaled to the
reader's [0, 1] aperture. Buckets are found recursively (bucket = path under root).
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from openwam.dataloader.interndata_a1 import ROBOT_TYPE_TO_EMBODIMENT, iter_data_shards, resolve_gripper_scale
from openwam.dataloader.utils.eef import quat_wxyz_to_rot6d
from openwam.dataloader.utils.exclusion_io import atomic_publish_text, locked_exclusion_files
from openwam.dataloader.utils.lerobotv3 import compute_file_local_offsets, load_episodes_parquet, parse_info_json
from openwam.dataloader.utils.prompt_text import EMPTY_SLOT, REPEATED_WORD, SLUG, has_asset_id

POSE_COL = "observation.state.ee_base"
EFFECTOR_COLS = ("observation.state.gripper", "observation.state.dex")
# 18-D ee_base = [L_xyz, L_rot6d, R_xyz, R_rot6d]
ARMS = ((slice(0, 3), slice(3, 9)), (slice(9, 12), slice(12, 18)))
HEAD_CAMERAS = (
    "observation.images.head", "images.rgb.head",
    "observation.image.exterior_image_1_left",  # DROID
    "observation.images.image",  # LIBERO
)
# InternData-A1: per-arm xyz + quaternion wxyz; grippers in native stroke units
INTERN_POSE_COLS = ("states.left_ee_to_robot_pose", "states.right_ee_to_robot_pose")
INTERN_GRIPPER_COLS = ("states.left_gripper.position", "states.right_gripper.position")
# Single-arm Franka sets: DROID (xyz + euler "xyz", gripper [0,1]) and LIBERO (xyz + axis-angle,
# two finger positions in m; their difference / 0.08 ≈ [0,1] aperture)
DROID_POSE_COL, DROID_GRIPPER_COL = "observation.state.cartesian_position", "observation.state.gripper_position"
LIBERO_STATE_COL = "observation.state"
CHASSIS_CMD_COL, TORSO_COL = "action.chassis.velocities", "observation.state.torso"
# optional per-episode columns converters write into meta/episodes, copied into the scan
EPISODE_EXTRAS = {"bad_image_frames": 0, "unqualified_frames": 0, "coarse_quality": "", "bag_quality": ""}
PLACEHOLDER = re.compile(r"^(null|none|nan|n/?a|todo|test|task|do something|default)\W*$", re.I)


def _rot6d_to_mat(r6: np.ndarray) -> np.ndarray:
    """(T,6) → (T,3,3) via Gram-Schmidt, columns = [a, b, a×b]."""
    a = r6[:, 0:3] / np.linalg.norm(r6[:, 0:3], axis=-1, keepdims=True)
    b = r6[:, 3:6] - (a * r6[:, 3:6]).sum(-1, keepdims=True) * a
    b /= np.linalg.norm(b, axis=-1, keepdims=True)
    return np.stack([a, b, np.cross(a, b)], axis=-1)


def pose_metrics(pose: np.ndarray) -> dict:
    """Motion metrics over every arm of a (T, 9·arms) track: xyz + rot6d per arm."""
    out = {"max_step_m": 0.0, "max_step_deg": 0.0, "path_m": 0.0, "rot6d_err": 0.0}
    for xyz_s, rot_s in ARMS[: pose.shape[1] // 9]:
        xyz, r6 = pose[:, xyz_s], pose[:, rot_s]
        out["rot6d_err"] = max(
            out["rot6d_err"],
            float(np.abs(np.linalg.norm(r6[:, :3], axis=-1) - 1).max()),
            float(np.abs(np.linalg.norm(r6[:, 3:], axis=-1) - 1).max()),
            float(np.abs((r6[:, :3] * r6[:, 3:]).sum(-1)).max()),
        )
        if len(pose) < 2:
            continue
        step = np.linalg.norm(np.diff(xyz, axis=0), axis=-1)
        out["max_step_m"] = max(out["max_step_m"], float(step.max()))
        out["path_m"] += float(step.sum())
        m = _rot6d_to_mat(r6)
        cos = (np.einsum("tij,tij->t", m[1:], m[:-1]) - 1) / 2  # trace(R1ᵀR0)
        out["max_step_deg"] = max(out["max_step_deg"], float(np.degrees(np.arccos(np.clip(cos, -1, 1))).max()))
    return out


def prompt_ok(text: str) -> bool:
    if " | " in (text or ""):  # DROID: several annotator variants joined by " | "; one usable variant is enough
        return any(prompt_ok(v) for v in text.split(" | "))
    text = (text or "").strip()
    words = len(re.findall(r"[A-Za-z]{2,}", text)) + len(re.findall(r"[\u4e00-\u9fff]", text))  # CJK: per character
    return (
        words >= 2
        and not PLACEHOLDER.match(text)
        and not EMPTY_SLOT.search(text)
        and not REPEATED_WORD.search(text)
        and not SLUG.match(text)
        and not has_asset_id(text)
    )


def video_metrics(path: Path, t0: float, t1: float, samples: int) -> dict:
    """Keyframe-only decode of ``[t0, t1)`` s: cheap enough for black / frozen checks."""
    import av

    try:
        with av.open(str(path)) as c:
            stream = c.streams.video[0]
            keys = [
                p for p in c.demux(stream)
                if p.is_keyframe and p.pts is not None and t0 <= float(p.pts * stream.time_base) < t1
            ]
            pick = [keys[i] for i in np.unique(np.linspace(0, len(keys) - 1, min(samples, len(keys))).astype(int))]
            frames = [f for p in pick for f in stream.codec_context.decode(p)]
            frames += stream.codec_context.decode(None)  # dav1d buffers until flushed
            imgs = [f.to_ndarray(width=128, height=96, format="rgb24").astype(np.float32) for f in frames]
    except Exception:  # noqa: BLE001 - any decode failure is the finding
        imgs = []
    if not imgs:
        return {"video_ok": False, "video_mean": np.nan, "video_max_diff": np.nan}
    imgs = np.stack(imgs)
    return {
        "video_ok": True,
        "video_mean": float(imgs.mean()),
        "video_max_diff": float(np.abs(np.diff(imgs, axis=0)).mean(axis=(1, 2, 3)).max()) if len(imgs) > 1 else np.nan,
    }


def _column(table, name: str) -> np.ndarray:
    return np.stack(table[name].to_numpy(zero_copy_only=False)).astype(np.float64).reshape(len(table), -1)


def _single_arm(win, names: set, robot_type: str | None):
    """(pose (T,9), [effector (T,1)]) for DROID / LIBERO windows, else None."""
    from scipy.spatial.transform import Rotation

    if DROID_POSE_COL in names:
        x, eff = _column(win, DROID_POSE_COL), [_column(win, DROID_GRIPPER_COL)]
        rot = Rotation.from_euler("xyz", np.nan_to_num(x[:, 3:6]))
    elif LIBERO_STATE_COL in names and robot_type == "franka":
        x = _column(win, LIBERO_STATE_COL)
        rot, eff = Rotation.from_rotvec(np.nan_to_num(x[:, 3:6])), [(x[:, 6:7] - x[:, 7:8]) / 0.08]
    else:
        return None
    m = rot.as_matrix()
    return np.concatenate([x[:, :3], m[:, :, 0], m[:, :, 1]], 1), eff


def _relocate_by_global_index(root: Path, eps: pd.DataFrame) -> None:
    """Place episodes by ``dataset_from_index`` against physical shard lengths (as InternDataA1 does)."""
    shards = iter_data_shards(root)
    starts = np.cumsum([0] + [pq.read_metadata(p).num_rows for _, _, p in shards])
    g = eps["dataset_from_index"].to_numpy()
    pos = np.searchsorted(starts, g, side="right") - 1
    if starts[-1] != eps["dataset_to_index"].max() or (g + eps["length"].to_numpy() > starts[pos + 1]).any():
        raise ValueError(f"{root}: data shards do not match the episodes manifest")
    eps["data/chunk_index"] = [shards[i][0] for i in pos]
    eps["data/file_index"] = [shards[i][1] for i in pos]
    eps["_row"] = g - starts[pos]


def scan_bucket(bucket: str, video: bool = False, video_samples: int = 5, shard: tuple[int, int] = (0, 1)) -> pd.DataFrame:
    root = Path(bucket)
    info = parse_info_json(root)
    eps = load_episodes_parquet(root)
    eps["_row"] = compute_file_local_offsets(eps, "data/chunk_index", "data/file_index")
    if info.get("robot_type") in ROBOT_TYPE_TO_EMBODIMENT:  # InternData-A1: manifest file indices can be stale
        _relocate_by_global_index(root, eps)
    cams = [c.split("/")[1] for c in eps.columns if c.startswith("videos/") and c.endswith("/chunk_index")]
    head = next((c for c in HEAD_CAMERAS if c in cams), cams[0] if cams else HEAD_CAMERAS[0])  # e.g. Hy: cam_high
    vcol = (f"videos/{head}/chunk_index", f"videos/{head}/file_index")
    tasks = pd.read_parquet(root / "meta" / "tasks.parquet")
    task_text = dict(zip(tasks["task_index"], tasks.index))
    grip_scale = {}
    if info.get("robot_type") in ROBOT_TYPE_TO_EMBODIMENT:  # InternData-A1
        emb = ROBOT_TYPE_TO_EMBODIMENT[info["robot_type"]]
        grip_scale = {c: resolve_gripper_scale(root, emb, c) for c in INTERN_GRIPPER_COLS}

    rows = []
    for k, ((chunk, file), group) in enumerate(eps.groupby(["data/chunk_index", "data/file_index"])):
        if k % shard[1] != shard[0]:  # --bucket-shards: every n-th data file
            continue
        path = root / info["data_path"].format(chunk_index=chunk, file_index=file)
        names = set(pq.read_schema(path).names)
        single = POSE_COL not in names and (
            DROID_POSE_COL in names or (LIBERO_STATE_COL in names and info.get("robot_type") == "franka")
        )
        intern = POSE_COL not in names and not single
        pose_cols, eff_cols = (INTERN_POSE_COLS, INTERN_GRIPPER_COLS) if intern else ((POSE_COL,), EFFECTOR_COLS)
        if single:
            pose_cols, eff_cols = (DROID_POSE_COL, LIBERO_STATE_COL), (DROID_GRIPPER_COL,)
        cols = [c for c in (*pose_cols, *eff_cols, CHASSIS_CMD_COL, TORSO_COL, "task_index") if c in names]
        table = pq.read_table(path, columns=cols)
        for _, ep in group.iterrows():
            n, start = int(ep["length"]), int(ep["_row"])
            win = table.slice(start, n)
            r = {"bucket": root.name, "episode_index": int(ep["episode_index"]), "length": n}
            r.update({k: type(d)(ep[k]) if k in ep and pd.notna(ep[k]) else d for k, d in EPISODE_EXTRAS.items()})
            if CHASSIS_CMD_COL in cols:
                cmd = np.stack(win[CHASSIS_CMD_COL].to_numpy(zero_copy_only=False)).astype(np.float64)
                r["chassis_cmd_frac"] = float((np.abs(cmd).max(axis=1) > 1e-3).mean())
            if TORSO_COL in cols:
                r["torso_range"] = float(np.ptp(np.stack(win[TORSO_COL].to_numpy(zero_copy_only=False)), axis=0).max())
            if single:
                pose, eff = _single_arm(win, names, info.get("robot_type"))
                raw = [pose]
            else:
                raw = [_column(win, c) for c in pose_cols]
                eff = [_column(win, c) / grip_scale.get(c, 1.0) for c in eff_cols if c in cols]
            r["nonfinite"] = not all(np.isfinite(x).all() for x in (*raw, *eff))
            pose = pose if single else np.concatenate(
                [np.concatenate([a[:, :3], quat_wxyz_to_rot6d(np.nan_to_num(a[:, 3:]))], 1) for a in raw], 1
            ).astype(np.float64) if intern else raw[0]
            r["zero_pose_frames"] = int(sum((np.abs(pose[:, xyz]).sum(1) == 0).sum() for xyz, _ in ARMS[: pose.shape[1] // 9]))
            r.update(pose_metrics(np.nan_to_num(pose)) if not r["nonfinite"] else {})
            r["effector_range"] = float(max((np.ptp(e, axis=0).max() for e in eff), default=0.0))
            r["prompt_ok"] = all(prompt_ok(task_text.get(int(t), "")) for t in set(win["task_index"].to_pylist()))
            if video and vcol[0] in eps.columns:
                vpath = root / info["video_path"].format(
                    video_key=head, chunk_index=int(ep[vcol[0]]), file_index=int(ep[vcol[1]])
                )
                t0 = float(ep.get(f"videos/{head}/from_timestamp", 0.0))
                t1 = float(ep.get(f"videos/{head}/to_timestamp", np.inf))
                r.update(video_metrics(vpath, t0, t1, video_samples))
            rows.append(r)
    return pd.DataFrame(rows)


def flag(df: pd.DataFrame, a: argparse.Namespace, fps_by_bucket: dict[str, float]) -> pd.Series:
    """Return a ``;``-joined reason string per episode ('' = keep)."""
    fps = df["bucket"].map(fps_by_bucket)
    median = df.groupby("bucket")["length"].transform("median")
    rules = {
        "too_short": df["length"] < a.min_seconds * fps,
        "too_long": df["length"] > a.max_len_x_median * median,
        "nonfinite": df["nonfinite"],
        "zero_pose": df["zero_pose_frames"] > 0 if "zero_pose_frames" in df else pd.Series(False, index=df.index),
        "bad_image": df["bad_image_frames"] > 0 if "bad_image_frames" in df else pd.Series(False, index=df.index),
        "pos_jump": df["max_step_m"] > a.max_step_m,
        "rot_jump": df["max_step_deg"] > a.max_step_deg,
        "invalid_rot6d": df["rot6d_err"] > 0.05,
        "static": (df["path_m"] < a.min_path_m) & (df["effector_range"] < a.min_effector_range),
        "bad_prompt": ~df["prompt_ok"],
    }
    if "unqualified_frames" in df:  # older quality.parquet files (--quality) lack these columns
        coarse = df["coarse_quality"].fillna("") if "coarse_quality" in df else pd.Series("", index=df.index)
        rules["official_unqualified"] = (df["unqualified_frames"] > 0) | coarse.str.contains("unqualified")
        if "bag_quality" in df:
            rules["bag_unqualified"] = df["bag_quality"] == "不合格"
    if getattr(a, "max_chassis_cmd_frac", None) is not None or getattr(a, "max_torso_range", None) is not None:
        moving = pd.Series(False, index=df.index)
        if a.max_chassis_cmd_frac is not None and "chassis_cmd_frac" in df:
            moving |= df["chassis_cmd_frac"] > a.max_chassis_cmd_frac
        if a.max_torso_range is not None and "torso_range" in df:
            moving |= df["torso_range"] > a.max_torso_range
        rules["body_motion"] = moving
    if "video_ok" in df.columns:
        rules["bad_video"] = df["video_ok"] == False  # noqa: E712 - NaN for unscanned rows stays False
        rules["black_video"] = df["video_mean"] < a.min_brightness
        rules["frozen_video"] = df["video_max_diff"] < a.min_frame_diff
    reasons = pd.Series([""] * len(df), index=df.index)
    for name, hit in rules.items():
        hit = hit.fillna(False).astype(bool)
        reasons[hit] = reasons[hit] + name + ";"
    return reasons.str.rstrip(";")


def apply_exclusions(root: Path, df: pd.DataFrame) -> None:
    """Merge flagged episodes into each bucket's excluded_episodes.json."""
    for bucket, group in df[df["reasons"] != ""].groupby("bucket"):
        path = root / bucket / "meta" / "excluded_episodes.json"
        with locked_exclusion_files([path]):
            payload = json.loads(path.read_text()) if path.exists() else {"episode_indices": []}
            payload["episode_indices"] = sorted(set(payload["episode_indices"]) | set(group["episode_index"].tolist()))
            payload.setdefault("episode_quality", {}).update(
                {str(e): r for e, r in zip(group["episode_index"], group["reasons"])}
            )
            atomic_publish_text(path, json.dumps(payload, indent=1))


def delete_episodes(root: Path, df: pd.DataFrame) -> None:
    """Physically drop flagged episodes from converter-layout buckets.

    Only for the one-episode-per-data-file layout written by
    ``agibotworld_convert`` (anything else would need shard rewrites, so it is
    refused). Removes the data file and video links, rewrites the episodes
    table and info totals, and logs reasons to ``meta/deleted_episodes.json``. A bucket left empty
    is removed and recorded in ``<root>/deleted_buckets.json``.
    """
    for bucket, group in df[df["reasons"] != ""].groupby("bucket"):
        b = root / bucket
        info = parse_info_json(b)
        eps = load_episodes_parquet(b)
        if eps.duplicated(["data/chunk_index", "data/file_index"]).any():
            raise ValueError(f"{b}: several episodes share a data file; delete needs the converter layout")
        drop = eps["episode_index"].isin(group["episode_index"])
        video_keys = [c.split("/")[1] for c in eps.columns if c.startswith("videos/") and c.endswith("/chunk_index")]
        for _, ep in eps[drop].iterrows():
            ci, fi = int(ep["data/chunk_index"]), int(ep["data/file_index"])
            (b / info["data_path"].format(chunk_index=ci, file_index=fi)).unlink(missing_ok=True)
            for k in video_keys:
                (b / info["video_path"].format(video_key=k, chunk_index=ci, file_index=fi)).unlink(missing_ok=True)
        kept = eps[~drop]
        if kept.empty:  # readers reject empty buckets: drop it, keep a root-level record
            log = root / "deleted_buckets.json"
            record = json.loads(log.read_text()) if log.exists() else {}
            record[bucket] = {"episodes": len(eps), "reasons": group["reasons"].value_counts().to_dict()}
            log.write_text(json.dumps(record, indent=1))
            shutil.rmtree(b)
            continue
        for f in (b / "meta" / "episodes").glob("*.parquet"):
            f.unlink()
        kept.to_parquet(b / "meta" / "episodes" / "chunk-000.parquet", index=False)
        raw_info = json.loads((b / "meta" / "info.json").read_text())
        raw_info.update(total_episodes=len(kept), total_frames=int(kept["length"].sum()))
        (b / "meta" / "info.json").write_text(json.dumps(raw_info, indent=2, ensure_ascii=False))
        log = b / "meta" / "deleted_episodes.json"
        record = json.loads(log.read_text()) if log.exists() else {}
        record.update({str(e): r for e, r in zip(group["episode_index"], group["reasons"])})
        log.write_text(json.dumps(record, indent=1))


def find_buckets(root: Path) -> list[str]:
    """Bucket dirs (``meta/info.json``) at any depth, as paths relative to ``root``; no descent into buckets."""
    out = []
    for d, dirs, _ in os.walk(root):
        if (Path(d) / "meta" / "info.json").is_file():
            out.append(str(Path(d).relative_to(root)))
            dirs.clear()
    return sorted(out)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", required=True, help="dataset root containing bucket dirs (or a single bucket)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--buckets", nargs="*")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--bucket-shards", type=int, default=1, help="split each bucket's data files over N processes")
    ap.add_argument("--video", action="store_true", help="also decode sampled head frames")
    ap.add_argument("--quality", help="reuse an existing quality.parquet instead of rescanning")
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument("--apply", action="store_true", help="write meta/excluded_episodes.json")
    mode.add_argument("--delete", action="store_true", help="physically remove flagged episodes (converter layout only)")
    ap.add_argument("--min-seconds", type=float, default=2.0)
    ap.add_argument("--max-len-x-median", type=float, default=5.0)
    ap.add_argument("--max-step-m", type=float, default=0.10)
    ap.add_argument("--max-step-deg", type=float, default=20.0)
    ap.add_argument("--min-path-m", type=float, default=0.05)
    ap.add_argument("--min-effector-range", type=float, default=0.05)
    ap.add_argument("--min-brightness", type=float, default=10.0)
    ap.add_argument("--min-frame-diff", type=float, default=0.5)
    ap.add_argument("--max-chassis-cmd-frac", type=float, help="opt-in body_motion: share of base-command frames")
    ap.add_argument("--max-torso-range", type=float, help="opt-in body_motion: torso joint range (rad)")
    a = ap.parse_args()

    root = Path(a.root)
    if (root / "meta" / "info.json").is_file():
        root, buckets = root.parent, [root.name]
    else:
        buckets = a.buckets or find_buckets(root)
    if a.quality:
        df = pd.read_parquet(a.quality).drop(columns="reasons", errors="ignore")
        df = df[df["bucket"].isin(buckets)].reset_index(drop=True)
    else:
        with ProcessPoolExecutor(a.workers) as pool:
            n = a.bucket_shards
            futs = [(b, pool.submit(scan_bucket, str(root / b), a.video, 5, (i, n))) for b in buckets for i in range(n)]
            df = pd.concat([f.result().assign(bucket=b) for b, f in futs], ignore_index=True)
            df = df.sort_values(["bucket", "episode_index"], ignore_index=True)
    fps = {b: float(parse_info_json(root / b)["fps"]) for b in buckets}
    df["reasons"] = flag(df, a, fps)

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out / "quality.parquet", index=False)
    hits = df["reasons"].str.split(";").explode()
    summary = {
        "episodes": len(df),
        "flagged": int((df["reasons"] != "").sum()),
        "frames_flagged": int(df.loc[df["reasons"] != "", "length"].sum()),
        "frames_total": int(df["length"].sum()),
        "by_reason": hits[hits != ""].value_counts().to_dict(),
        "thresholds": {k: v for k, v in vars(a).items() if k.startswith(("min_", "max_"))},
        "applied": "delete" if a.delete else a.apply,
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    if a.apply:
        apply_exclusions(root, df)
    if a.delete:
        delete_episodes(root, df)


if __name__ == "__main__":
    main()
