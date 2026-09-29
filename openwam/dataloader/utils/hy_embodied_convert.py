#!/usr/bin/env python3
"""Convert Hy-Embodied-0.5-VLA-Data Lance tables into LeRobot v3 buckets.

Input: ``<root>/table_XXX/{table_XXX.lance, meta/}``; every row is one 30 Hz
frame with three per-frame JPEGs and a 16-D state
``[L xyz, L quat xyzw, L grip, R xyz, R quat xyzw, R grip]`` (UMI, motion-
capture frame; no robot base).

Output: one bucket per table in the ``agibotworld_convert`` layout (one data
parquet + one mp4 per camera per episode), so ``episode_quality`` (scan and
``--delete``) applies unchanged. JPEGs are piped straight into ffmpeg
(libx264; no AV1 encoder fast enough for ~700M frames is available).

Columns: ``observation.state.ee_base`` = 18-D ``[L_xyz, L_rot6d, R_xyz,
R_rot6d]`` (absolute pose, lossless; relative poses are derivable later),
Leading unsynced frames (zero pose or 1-byte image; ~17% of episodes start
with exactly one) are trimmed and counted in ``trimmed_leading_frames``.
Frames labeled ``__UNKNOWN__`` (gaps between sub-tasks) take the nearest
labeled task, counted in ``unlabeled_frames``.
``action.ee_base`` = next state (last row repeated), grippers mapped to the
shared ``[0, 1]`` / 0 = closed / 1 = open convention (source: 0 open .. 90 closed). Any remaining zero-pose frames
(tracking loss mid-episode) are kept; ``episode_quality``'s ``zero_pose`` rule
flags them. Remaining placeholder image frames (1-byte entries) are filled from the
nearest valid frame to keep alignment and counted in the episodes table
(``bad_image_frames``) for the ``bad_image`` rule.

    python -m openwam.dataloader.utils.hy_embodied_convert \
        --root /mnt/data/datasets/Hy-Embodied-0.5-VLA-Data --tables table_000 \
        --out /root/Hy-Embodied-lerobotv3 --workers 64
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from openwam.dataloader.utils.eef import quat_xyzw_to_rot6d

FPS = 30
CAMS = {
    "observation.images.cam_high": "observation_images_cam_high",
    "observation.images.cam_left_wrist": "observation_images_cam_left_wrist",
    "observation.images.cam_right_wrist": "observation_images_cam_right_wrist",
}
DATA_PATH = "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet"
VIDEO_PATH = "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4"
FILES_PER_CHUNK = 1000


def state_to_ee18(state: np.ndarray) -> np.ndarray:
    """(T,16) ``[xyz, quat xyzw, grip] x 2`` → (T,18) ``[xyz, rot6d] x 2``."""
    arms = [np.concatenate([state[:, o : o + 3], quat_xyzw_to_rot6d(state[:, o + 3 : o + 7])], axis=-1) for o in (0, 8)]
    return np.concatenate(arms, axis=-1).astype(np.float32)


GRIP_CLOSED_RAW = 90.0  # source grip: 0 = fully open, 90 = fully closed (checked on wrist images)


def gripper_open(raw: np.ndarray) -> np.ndarray:
    """Source grip value → shared convention: [0, 1], 0 = closed, 1 = open."""
    return (1.0 - np.clip(np.asarray(raw, dtype=np.float64) / GRIP_CLOSED_RAW, 0.0, 1.0)).astype(np.float32)


def episode_columns(state: np.ndarray, action: np.ndarray) -> dict[str, np.ndarray]:
    ee = state_to_ee18(state)
    return {
        "observation.state.ee_base": ee,
        "action.ee_base": np.concatenate([ee[1:], ee[-1:]], axis=0),
        "observation.state.gripper": gripper_open(state[:, [7, 15]]),
        "action.gripper": gripper_open(action),
    }


def zero_pose_mask(state: np.ndarray) -> np.ndarray:
    """Frames where either arm's xyz is exactly 0 (motion-capture tracking loss)."""
    return (np.abs(state[:, 0:3]).sum(1) == 0) | (np.abs(state[:, 8:11]).sum(1) == 0)


def _jpeg_ok(b) -> bool:
    return bool(b) and b[:2] == b"\xff\xd8"


def leading_invalid_frames(state: np.ndarray, images: dict[str, list]) -> int:
    """Number of leading frames with a zero pose or any undecodable image."""
    bad = zero_pose_mask(state)
    for v in images.values():
        bad |= ~np.array([_jpeg_ok(b) for b in v])
    return int(np.argmin(bad)) if not bad.all() else len(bad)


def fill_bad_jpegs(jpegs: list[bytes]) -> tuple[list[bytes], int]:
    """Replace placeholder frames (the source has 1-byte ``b"\\x00"`` entries)
    with the nearest valid JPEG so the video keeps one frame per row; ffmpeg
    would silently drop them and shift every later frame against the actions."""
    ok = [_jpeg_ok(b) for b in jpegs]
    if all(ok):
        return jpegs, 0
    if not any(ok):
        raise ValueError("no decodable frame in episode")
    good = [i for i, v in enumerate(ok) if v]
    out = [jpegs[i] if ok[i] else jpegs[min(good, key=lambda g: abs(g - i))] for i in range(len(jpegs))]
    return out, len(jpegs) - len(good)


def encode_jpegs(jpegs: list[bytes], dst: Path, crf: int) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg", "-loglevel", "error", "-y", "-f", "image2pipe", "-c:v", "mjpeg", "-framerate", str(FPS),
        "-i", "-", "-c:v", "libx264", "-preset", "veryfast", "-crf", str(crf), "-g", str(FPS),
        "-pix_fmt", "yuv420p", "-threads", "2", "-movflags", "+faststart", str(dst),
    ]  # fmt: skip
    subprocess.run(cmd, input=b"".join(jpegs), check=True, capture_output=True)


def _convert_episodes(lance_path: str, out: str, jobs: list[dict], crf: int, unknown: int | None) -> list[dict]:
    import lance

    ds = lance.dataset(lance_path)
    cols = ["episode_index", "frame_index", "task_index", "observation_state", "action", *CAMS.values()]
    rows = []
    for j in jobs:
        tb = ds.take(list(range(j["from"], j["to"])), columns=cols)
        ep = tb.column("episode_index").to_numpy()
        fr = tb.column("frame_index").to_numpy()
        n = len(fr)
        if not ((ep == j["episode_index"]).all() and (fr == np.arange(n)).all()):
            raise ValueError(f"episode {j['episode_index']}: rows {j['from']}:{j['to']} are not one ordered episode")
        state = np.asarray(tb.column("observation_state").to_pylist(), dtype=np.float64)
        action = np.asarray(tb.column("action").to_pylist(), dtype=np.float64)
        images = {k: tb.column(c).to_pylist() for k, c in CAMS.items()}
        # Recordings commonly start with one unsynced frame (zero pose, 1-byte image): trim it.
        lead = leading_invalid_frames(state, images)
        state, action = state[lead:], action[lead:]
        images = {k: v[lead:] for k, v in images.items()}
        n = len(state)
        if n == 0:
            continue
        df = pd.DataFrame({k: list(v) for k, v in episode_columns(state, action).items()})
        df["task_index"], n_unlabeled = fill_unknown_tasks(tb.column("task_index").to_numpy()[lead:], unknown)
        df["episode_index"] = int(j["episode_index"])
        df["frame_index"] = np.arange(n)
        df["index"] = j["cum"] + np.arange(n)
        df["timestamp"] = (np.arange(n) / FPS).astype(np.float32)
        chunk, file = divmod(j["seq"], FILES_PER_CHUNK)
        o = Path(out)
        data_path = o / DATA_PATH.format(chunk_index=chunk, file_index=file)
        data_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(data_path, index=False)
        row = {
            "episode_index": int(j["episode_index"]), "length": n, "dataset_from_index": j["cum"],
            "dataset_to_index": j["cum"] + n, "data/chunk_index": chunk, "data/file_index": file,
            "trimmed_leading_frames": lead, "zero_pose_frames": int(zero_pose_mask(state).sum()),
            "unlabeled_frames": n_unlabeled,
        }  # fmt: skip
        row["bad_image_frames"] = 0
        for key, col in CAMS.items():
            jpegs, n_bad = fill_bad_jpegs(images[key])
            row["bad_image_frames"] += n_bad
            encode_jpegs(jpegs, o / VIDEO_PATH.format(video_key=key, chunk_index=chunk, file_index=file), crf)
            row.update({f"videos/{key}/chunk_index": chunk, f"videos/{key}/file_index": file,
                        f"videos/{key}/from_timestamp": 0.0, f"videos/{key}/to_timestamp": n / FPS})  # fmt: skip
        rows.append(row)
    return rows


UNKNOWN_TASK = "__UNKNOWN__"


def fill_unknown_tasks(task_index: np.ndarray, unknown: int | None) -> tuple[np.ndarray, int]:
    """Frames between labeled sub-tasks carry the ``__UNKNOWN__`` task; give
    them the nearest labeled task. An all-unknown episode is left as-is (the
    ``bad_prompt`` rule then flags it)."""
    ti = np.asarray(task_index, dtype=np.int64)
    bad = ti == unknown if unknown is not None else np.zeros(len(ti), bool)
    if not bad.any() or bad.all():
        return ti, 0
    s = pd.Series(np.where(bad, np.nan, ti))
    filled = s.ffill().bfill().to_numpy().astype(np.int64)
    return filled, int(bad.sum())


def write_tasks(src: Path, dst: Path) -> None:
    """Source has columns ``[task_index, task]``; OpenWAM readers expect the
    LeRobot v3 layout (index = task text, single ``task_index`` column)."""
    t = pd.read_parquet(src)
    if "task" in t.columns:
        t = t.set_index("task")[["task_index"]]
    t.to_parquet(dst)


def convert_table(root: Path, table: str, out_root: Path, workers: int, crf: int, limit: int | None) -> dict:
    src, out = root / table, out_root / table
    if (out / "meta" / "info.json").exists():
        return {"table": table, "status": "exists"}
    if out.exists():
        shutil.rmtree(out)
    eps = pd.read_parquet(src / "meta" / "episodes").sort_values("episode_index").reset_index(drop=True)
    if limit:
        eps = eps.head(limit)
    eps["seq"] = np.arange(len(eps))
    eps["cum"] = np.concatenate([[0], np.cumsum(eps["length"].to_numpy())[:-1]])
    jobs = [
        {"episode_index": int(r.episode_index), "from": int(r.dataset_from_index), "to": int(r.dataset_to_index),
         "seq": int(r.seq), "cum": int(r.cum)}
        for r in eps.itertuples()
    ]  # fmt: skip
    batches = [jobs[i : i + 8] for i in range(0, len(jobs), 8)]
    lance_path = str(src / f"{table}.lance")
    tasks = pd.read_parquet(src / "meta" / "tasks.parquet")
    hit = tasks.loc[tasks["task"] == UNKNOWN_TASK, "task_index"]
    unknown = int(hit.iloc[0]) if len(hit) else None
    rows = []
    with ProcessPoolExecutor(workers) as pool:
        for i, r in enumerate(pool.map(_convert_episodes, [lance_path] * len(batches), [str(out)] * len(batches),
                                      batches, [crf] * len(batches), [unknown] * len(batches))):  # fmt: skip
            rows += r
            if i % 50 == 0:
                print(f"{table}: {len(rows)}/{len(jobs)} episodes", flush=True)

    meta = out / "meta"
    (meta / "episodes").mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).sort_values("dataset_from_index").to_parquet(meta / "episodes" / "chunk-000.parquet", index=False)
    write_tasks(src / "meta" / "tasks.parquet", meta / "tasks.parquet")
    frames = int(sum(r["length"] for r in rows))
    features = {k: {"dtype": "float32", "shape": [d]} for k, d in (
        ("observation.state.ee_base", 18), ("action.ee_base", 18),
        ("observation.state.gripper", 2), ("action.gripper", 2))}  # fmt: skip
    features.update({k: {"dtype": "int64", "shape": [1]} for k in ("task_index", "episode_index", "frame_index", "index")})
    features["timestamp"] = {"dtype": "float32", "shape": [1]}
    features.update({k: {"dtype": "video", "shape": [240, 424, 3], "info": {"video.codec": "h264"}} for k in CAMS})
    info = {
        "codebase_version": "v3.0",
        "robot_type": "umi_bimanual",
        "source": "tencent/Hy-Embodied-0.5-VLA-Data",
        "source_table": table,
        "fps": FPS,
        "total_episodes": len(rows),
        "total_frames": frames,
        "total_tasks": int(len(pd.read_parquet(meta / "tasks.parquet"))),
        "chunks_size": FILES_PER_CHUNK,
        "data_path": DATA_PATH,
        "video_path": VIDEO_PATH,
        "features": features,
        "pose_frame": "absolute UMI/motion-capture frame (no robot base)",
        "gripper": {"convention": "0_closed_1_open", "raw_range": [0.0, GRIP_CLOSED_RAW],
                    "raw_semantics": "0_open_90_closed", "transform": "1-clip(x/90,0,1)"},  # fmt: skip
    }
    (meta / "info.json").write_text(json.dumps(info, indent=2, ensure_ascii=False))
    return {"table": table, "status": "ok", "episodes": len(rows), "frames": frames,
            "zero_pose_episodes": int(sum(r["zero_pose_frames"] > 0 for r in rows)),
            "bad_image_episodes": int(sum(r["bad_image_frames"] > 0 for r in rows)),
            "trimmed_episodes": int(sum(r["trimmed_leading_frames"] > 0 for r in rows)),
            "unlabeled_filled_episodes": int(sum(r["unlabeled_frames"] > 0 for r in rows))}  # fmt: skip


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--tables", nargs="*", help="default: all table_* under root")
    ap.add_argument("--workers", type=int, default=64)
    ap.add_argument("--crf", type=int, default=23)
    ap.add_argument("--limit", type=int, help="convert only the first N episodes per table (smoke test)")
    a = ap.parse_args()
    root = Path(a.root)
    for t in a.tables or sorted(p.name for p in root.glob("table_*")):
        print(json.dumps(convert_table(root, t, Path(a.out), a.workers, a.crf, a.limit), ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
