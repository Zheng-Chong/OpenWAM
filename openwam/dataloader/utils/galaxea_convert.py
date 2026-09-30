#!/usr/bin/env python3
"""Convert Galaxea-Open-World-Dataset task archives (LeRobot v2.1 tar.gz) into
LeRobot v3 buckets in the ``agibotworld_convert`` layout.

Use ``lerobot/`` (2026-04 re-export), not ``lerobot_opensource/`` (2025-09):
same tasks plus one, and only it carries the per-bag quality check in
``training_data_set_meta.json`` and the full chassis state.

Input: ``<src>/<name>.tar.gz`` → ``<name>/{meta,data,videos}`` (v2.1, 15 fps,
AV1) + ``training_data_set_meta.json``. Output: one bucket per archive,
``<out>/<name>/``; one data parquet per episode, the source mp4s moved in
unchanged (no re-encode). ``episode_index`` is the source one.

Columns:

* ``observation.state.ee_base`` 18-D ``[L_xyz, L_rot6d, R_xyz, R_rot6d]`` from
  ``*_ee_pose`` (xyz + quat xyzw). The pose is in the **torso-end frame**
  (it does not move when only the torso moves), not the chassis frame.
* ``action.ee_base`` = next state (last row repeated).
* ``observation.state.gripper`` / ``action.gripper`` ``[L, R]`` in [0, 1],
  0 = closed, 1 = open: ``clip(raw / 100, 0, 1)`` (raw 0..100, low = closed,
  checked against the wrist cameras).
* every other source column (joints, torso, chassis, IMU, velocity commands,
  R1 Pro ``action.*_ee_pose`` …) is passed through unchanged.
* ``task_index`` → per-frame prompt: English half of the ``中文@English``
  subtask, else the coarse task. ``quality_index`` / ``coarse_quality_index``
  keep the source meaning (per-frame subtask label / episode label, text
  ``qualified``/``unqualified``) re-indexed into this bucket's ``tasks.parquet``.

The episodes table adds ``unqualified_frames``, ``bag_quality`` (source
``qualityLabel``, ``合格`` = pass) and ``bag_quality_sub`` for
``episode_quality``. Episodes whose frame count differs from any video's, whose
``frame_index`` is not ``0..n-1``, that miss a video/ee pose, or whose
quaternions are non-unit are skipped and
reported. ``meta/info.json`` is written last (completion marker).

    python -m openwam.dataloader.utils.galaxea_convert \
        --src /mnt/data/datasets/Galaxea-Open-World-Dataset/lerobot \
        --out /root/Galaxea-lerobotv3 --workers 32
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from openwam.dataloader.utils.eef import quat_xyzw_to_rot6d

FPS = 15
CAMS = {  # output key → source key
    "observation.images.head": "observation.images.head_rgb",
    "observation.images.head_right": "observation.images.head_right_rgb",
    "observation.images.hand_left": "observation.images.left_wrist_rgb",
    "observation.images.hand_right": "observation.images.right_wrist_rgb",
}
DATA_PATH = "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet"
VIDEO_PATH = "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4"
FILES_PER_CHUNK = 1000
GRIPPER_UNITS = {"range": [0, 1], "convention": "0 = closed, 1 = open",
                 "source": "observation.state.{left,right}_gripper / action.{left,right}_gripper, 0..100, low = closed",
                 "formula": "clip(raw / 100, 0, 1)"}
REINDEXED = ("timestamp", "frame_index", "episode_index", "index", "task_index", "coarse_task_index",
             "quality_index", "coarse_quality_index")  # fmt: skip


def _arr(df: pd.DataFrame, col: str) -> np.ndarray:
    return np.stack(df[col].to_numpy()).astype(np.float64).reshape(len(df), -1)


def episode_columns(df: pd.DataFrame) -> dict[str, np.ndarray]:
    """Source v2.1 episode frame → new per-frame columns (ee_base, gripper)."""
    arms = [_arr(df, f"observation.state.{s}_ee_pose") for s in ("left", "right")]
    ee = np.concatenate([np.concatenate([p[:, :3], quat_xyzw_to_rot6d(p[:, 3:7])], axis=-1) for p in arms], axis=-1)
    ee = ee.astype(np.float32)
    grip = lambda k: gripper_to_unit(np.concatenate([_arr(df, f"{k}.left_gripper"), _arr(df, f"{k}.right_gripper")], 1))
    return {
        "observation.state.ee_base": ee,
        "action.ee_base": np.concatenate([ee[1:], ee[-1:]], axis=0),
        "observation.state.gripper": grip("observation.state").astype(np.float32),
        "action.gripper": grip("action").astype(np.float32),
    }


def gripper_to_unit(raw: np.ndarray) -> np.ndarray:
    """Source 0..100 (checked against wrist video: ~1 = jaws closed, ~97 = open) → [0, 1], 0 = closed."""
    return np.clip(raw / 100.0, 0.0, 1.0)


def prompt_text(sub: str, coarse: str) -> str:
    """``中文@English`` subtask → English half; placeholder subtask → coarse task."""
    text = sub if sub.strip().lower() not in ("", "null", "none") else coarse
    return text.split("@", 1)[1].strip() if "@" in text else text.strip()


def _video_frames(path: Path) -> int:
    import av

    with av.open(str(path)) as c:
        return int(c.streams.video[0].frames)


def _check_episode(src: Path, info: dict, e: dict):
    """Return the source DataFrame or a skip-reason string."""
    ep = e["episode_index"]
    chunk = ep // info["chunks_size"]
    try:
        df = pd.read_parquet(src / info["data_path"].format(episode_chunk=chunk, episode_index=ep))
    except Exception as err:  # noqa: BLE001 - unreadable file is the finding
        return f"bad_parquet:{type(err).__name__}"
    n = len(df)
    if n == 0 or n != e["length"]:
        return "length_mismatch"
    if not (df["frame_index"].to_numpy() == np.arange(n)).all():
        return "bad_frame_index"
    if not {"observation.state.left_ee_pose", "observation.state.right_ee_pose"} <= set(df.columns):
        return "missing_ee_pose"
    q = np.concatenate([_arr(df, f"observation.state.{s}_ee_pose")[:, 3:7] for s in ("left", "right")])
    if not np.isfinite(q).all() or (np.abs(np.linalg.norm(q, axis=1) - 1) > 0.1).any():
        return "bad_quaternion"
    for s in CAMS.values():
        p = src / info["video_path"].format(episode_chunk=chunk, video_key=s, episode_index=ep)
        if not p.is_file():
            return "missing_video"
        try:
            if _video_frames(p) != n:
                return "video_len_mismatch"
        except Exception as err:  # noqa: BLE001
            return f"bad_video:{type(err).__name__}"
    return df


def convert_archive(tar: str, out: str, overwrite: bool = False) -> dict:
    name = Path(tar).name.removesuffix(".tar.gz")
    out_p = Path(out) / name
    if (out_p / "meta" / "info.json").exists() and not overwrite:
        return {"bucket": name, "status": "exists"}
    tmp = Path(out) / ".extract" / name
    shutil.rmtree(tmp, ignore_errors=True)
    shutil.rmtree(out_p, ignore_errors=True)
    tmp.mkdir(parents=True)
    try:
        subprocess.run(["tar", "xzf", tar, "-C", str(tmp)], check=True, capture_output=True)
        return _convert_extracted(tmp, out_p, name)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def _convert_extracted(tmp: Path, out_p: Path, name: str) -> dict:
    src = next(tmp.glob("**/meta/info.json")).parent.parent
    info = json.loads((src / "meta" / "info.json").read_text())
    src_tasks = {t["task_index"]: t["task"] for t in map(json.loads, (src / "meta" / "tasks.jsonl").open())}
    src_eps = [json.loads(line) for line in (src / "meta" / "episodes.jsonl").open()]
    meta_path = next(tmp.glob("**/training_data_set_meta.json"), None)
    bag_meta = json.loads(meta_path.read_text()) if meta_path else {}
    bags = {b["name"]: b for b in bag_meta.get("rawDataList") or []}

    skipped: dict[str, int] = {}
    task_ids: dict[str, int] = {}
    tid = lambda text: task_ids.setdefault(text, len(task_ids))
    rows, cum = [], 0
    for e in src_eps:
        df = _check_episode(src, info, e)
        if isinstance(df, str):
            skipped[df] = skipped.get(df, 0) + 1
            continue
        n, ep = len(df), int(e["episode_index"])
        seq = len(rows)
        chunk, file = divmod(seq, FILES_PER_CHUNK)
        new = {k: list(v) for k, v in episode_columns(df).items()}
        passthrough = [c for c in df.columns if c not in REINDEXED and c not in new]
        coarse = [src_tasks[i] for i in df["coarse_task_index"]]
        prompts = [prompt_text(src_tasks[s], c) for s, c in zip(df["task_index"], coarse)]
        quality = [src_tasks[i] for i in df["quality_index"]]
        coarse_quality = [src_tasks[i] for i in df["coarse_quality_index"]]
        frame = np.arange(n)
        o = pd.DataFrame(new)
        for c in passthrough:
            o[c] = df[c].to_numpy()
        o["task_index"] = [tid(p) for p in prompts]
        o["quality_index"] = [tid(q) for q in quality]
        o["coarse_quality_index"] = [tid(q) for q in coarse_quality]
        o["episode_index"] = ep
        o["frame_index"] = frame
        o["index"] = cum + frame
        o["timestamp"] = (frame / FPS).astype(np.float32)
        data_path = out_p / DATA_PATH.format(chunk_index=chunk, file_index=file)
        data_path.parent.mkdir(parents=True, exist_ok=True)
        o.to_parquet(data_path, index=False)

        bag = bags.get(e.get("raw_file_name"), {})
        row = {
            "episode_index": ep, "length": n, "tasks": sorted(set(prompts)),
            "dataset_from_index": cum, "dataset_to_index": cum + n,
            "data/chunk_index": chunk, "data/file_index": file,
            # only explicit "unqualified": ~0.7% of source labels are misplaced task texts, not verdicts
            "unqualified_frames": int(sum(q == "unqualified" for q in quality)),
            "coarse_quality": ",".join(sorted(set(coarse_quality))),
            "bag_quality": str(bag.get("qualityLabel")), "bag_quality_sub": str(bag.get("qualitySubLabel")),
            "raw_file_name": str(e.get("raw_file_name")),
        }  # fmt: skip
        src_chunk = ep // info["chunks_size"]
        for key, s in CAMS.items():
            dst = out_p / VIDEO_PATH.format(video_key=key, chunk_index=chunk, file_index=file)
            dst.parent.mkdir(parents=True, exist_ok=True)
            os.replace(src / info["video_path"].format(episode_chunk=src_chunk, video_key=s, episode_index=ep), dst)
            row.update({f"videos/{key}/chunk_index": chunk, f"videos/{key}/file_index": file,
                        f"videos/{key}/from_timestamp": 0.0, f"videos/{key}/to_timestamp": n / FPS})  # fmt: skip
        rows.append(row)
        cum += n

    if not rows:
        return {"bucket": name, "status": "empty", "skipped": skipped, "source_episodes": len(src_eps)}
    meta = out_p / "meta"
    (meta / "episodes").mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(meta / "episodes" / "chunk-000.parquet", index=False)
    pd.DataFrame({"task_index": list(task_ids.values())}, index=pd.Index(list(task_ids), name="task")).to_parquet(
        meta / "tasks.parquet"
    )
    features = {k: {"dtype": "float32", "shape": [d]} for k, d in (
        ("observation.state.ee_base", 18), ("action.ee_base", 18),
        ("observation.state.gripper", 2), ("action.gripper", 2))}  # fmt: skip
    features.update({c: info["features"][c] for c in passthrough if c in info["features"]})
    features.update({k: {"dtype": "int64", "shape": [1]} for k in
                     ("task_index", "quality_index", "coarse_quality_index", "episode_index", "frame_index", "index")})  # fmt: skip
    features["timestamp"] = {"dtype": "float32", "shape": [1]}
    features.update({k: {**info["features"][s], "source": s} for k, s in CAMS.items()})
    out_info = {
        "codebase_version": "v3.0",
        "robot_type": info.get("robot_type"),
        "source": "OpenGalaxea/Galaxea-Open-World-Dataset (lerobot/)",
        "source_archive": name,
        "source_version": bag_meta.get("trainingDataSetVersion"),
        "fps": FPS,
        "total_episodes": len(rows),
        "total_frames": cum,
        "total_tasks": len(task_ids),
        "chunks_size": FILES_PER_CHUNK,
        "data_path": DATA_PATH,
        "video_path": VIDEO_PATH,
        "features": features,
        "pose_frame": "torso-end frame (moves with torso, not chassis)",
        "gripper_units": GRIPPER_UNITS,
        "conversion_skipped": skipped,
        "source_episodes": len(src_eps),
    }
    (meta / "info.json").write_text(json.dumps(out_info, indent=2, ensure_ascii=False))
    return {"bucket": name, "status": "ok", "robot_type": info.get("robot_type"), "episodes": len(rows),
            "source_episodes": len(src_eps), "frames": cum, "skipped": skipped}  # fmt: skip


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", required=True, help="directory of <name>.tar.gz archives")
    ap.add_argument("--out", required=True)
    ap.add_argument("--archives", nargs="*", help="archive names (default: all *.tar.gz under --src)")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--overwrite", action="store_true")
    a = ap.parse_args()

    src = Path(a.src)
    tars = [src / (n if n.endswith(".tar.gz") else f"{n}.tar.gz") for n in a.archives] if a.archives else list(src.glob("*.tar.gz"))
    tars.sort(key=lambda p: -p.stat().st_size)  # big first: better packing
    Path(a.out).mkdir(parents=True, exist_ok=True)
    path = Path(a.out) / "conversion_report.jsonl"
    merged = {r["bucket"]: r for r in map(json.loads, path.read_text().splitlines())} if path.exists() else {}
    with ProcessPoolExecutor(a.workers) as pool:
        futs = [pool.submit(convert_archive, str(t), a.out, a.overwrite) for t in tars]
        for f in futs:
            try:
                r = f.result()
            except Exception as err:  # noqa: BLE001 - report and keep going with the other archives
                r = {"bucket": Path(tars[futs.index(f)]).name.removesuffix(".tar.gz"), "status": f"error:{err!r}"[:500]}
            print(json.dumps(r, ensure_ascii=False), flush=True)
            if r["status"] != "exists":
                merged[r["bucket"]] = r
                path.write_text("".join(json.dumps(merged[b], ensure_ascii=False) + "\n" for b in sorted(merged)))


if __name__ == "__main__":
    main()
