#!/usr/bin/env python3
"""Convert AtomBench-CobotMagic (LeRobot v2.1, AgileX Cobot Magic, two Piper arms) into LeRobot v3
buckets in the ``galaxea_convert`` / ``agibotworld_convert`` layout, so ``episode_quality`` can scan them.

Input: ``<src>/<task>/{meta,data,videos}`` (15 tasks × 100 episodes, 30 fps, one mp4 per episode and
camera, H.264). Output: one bucket per task, ``<out>/<task>/``; one data parquet per episode, the mp4s
copied unchanged (no re-encode). ``episode_index`` is the source one.

Source columns: ``observation.state`` 26-D = per arm ``[joint1..6, gripper, x, y, z, rx, ry, rz]``
(right arm first, then left); ``action`` 14-D = leader ``[joint1..6, gripper]`` right then left.

New columns:

* ``observation.state.ee_base`` 18-D ``[L_xyz, L_rot6d, R_xyz, R_rot6d]`` from the Piper end pose.
  xyz in metres in **each arm's own base frame** (both arms start at ≈(-0.01, 0, 0.28); the lateral
  offset between the two bases is not in the data, so the two halves are NOT in one common frame).
  Euler is extrinsic XYZ (``R = Rz·Ry·Rx``): checked against Piper forward kinematics of the joint
  angles (position error 0.3 mm, rotation 1e-3), so ``eef.euler_xyz_to_rot6d`` applies as is.
* ``action.ee_base`` = next state (last row repeated); the leader has no end pose.
* ``observation.state.gripper`` / ``action.gripper`` ``[L, R]`` in [0, 1], 0 = closed, 1 = open:
  ``clip(raw, 0, 1)`` (raw is ≈0..1 over the whole dataset, action up to 1.06; checked on the wrist
  cameras: 0.78 = jaws released the ball).
* ``observation.state`` / ``action`` (joints) are passed through unchanged.

Episodes whose parquet is unreadable, whose length differs from ``episodes.jsonl`` or any video's,
whose ``frame_index`` is not consecutive, or that hold non-finite values are skipped and reported.
``frame_index`` of dm3/dm4/di5/di6 starts at 43..86 (source head trimmed, index not reset; rows,
timestamps and videos all start at 0 and have ``length`` frames): it is renumbered ``0..n-1`` and the
source start is kept as ``source_frame_offset`` in the episodes table.
``meta/info.json`` is written last (completion marker).

    python -m openwam.dataloader.utils.atombench_convert \
        --src /mnt/data/datasets/AtomBench-CobotMagic --out /root/AtomBench-CobotMagic-lerobotv3
"""

from __future__ import annotations

import argparse
import json
import shutil
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from openwam.dataloader.utils.eef import euler_xyz_to_rot6d

FPS = 30
CAMS = {  # output key → source key (top = fixed head view, left/right = wrist cameras)
    "observation.images.head": "observation.images.image_top",
    "observation.images.hand_left": "observation.images.image_left",
    "observation.images.hand_right": "observation.images.image_right",
}
DATA_PATH = "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet"
VIDEO_PATH = "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4"
FILES_PER_CHUNK = 1000
# observation.state layout: right arm [0:13], left arm [13:26]; each [j1..j6, gripper, x, y, z, rx, ry, rz]
STATE_ARM = {"right": 0, "left": 13}
ACTION_GRIPPER = {"right": 6, "left": 13}
GRIPPER_UNITS = {"range": [0, 1], "convention": "0 = closed, 1 = open", "formula": "clip(raw, 0, 1)",
                 "source": "observation.state[6, 19] / action[6, 13]"}  # fmt: skip
POSE_FRAME = "per-arm base frame (not a common robot frame; left/right base offset unknown)"
REINDEXED = ("timestamp", "frame_index", "episode_index", "index", "task_index")


def _arr(df: pd.DataFrame, col: str) -> np.ndarray:
    return np.stack(df[col].to_numpy()).astype(np.float64)


def episode_columns(df: pd.DataFrame) -> dict[str, np.ndarray]:
    """Source v2.1 episode frame → new per-frame columns (ee_base, gripper)."""
    s, a = _arr(df, "observation.state"), _arr(df, "action")
    arms = []
    for side in ("left", "right"):
        o = STATE_ARM[side]
        arms.append(np.concatenate([s[:, o + 7 : o + 10], euler_xyz_to_rot6d(s[:, o + 10 : o + 13])], axis=-1))
    ee = np.concatenate(arms, axis=-1).astype(np.float32)
    grip = lambda raw: np.clip(raw, 0.0, 1.0).astype(np.float32)  # noqa: E731
    return {
        "observation.state.ee_base": ee,
        "action.ee_base": np.concatenate([ee[1:], ee[-1:]], axis=0),
        "observation.state.gripper": grip(np.stack([s[:, STATE_ARM[k] + 6] for k in ("left", "right")], 1)),
        "action.gripper": grip(np.stack([a[:, ACTION_GRIPPER[k]] for k in ("left", "right")], 1)),
    }


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
    fi = df["frame_index"].to_numpy()
    if not (np.diff(fi) == 1).all():  # starts at 43..86 in dm3/dm4/di5/di6 (head trimmed, index not reset)
        return "bad_frame_index"
    s, a = _arr(df, "observation.state"), _arr(df, "action")
    if s.shape[1] != 26 or a.shape[1] != 14 or not (np.isfinite(s).all() and np.isfinite(a).all()):
        return "bad_state_or_action"
    for key in CAMS.values():
        p = src / info["video_path"].format(episode_chunk=chunk, video_key=key, episode_index=ep)
        if not p.is_file():
            return "missing_video"
        try:
            if _video_frames(p) != n:
                return "video_len_mismatch"
        except Exception as err:  # noqa: BLE001
            return f"bad_video:{type(err).__name__}"
    return df


def convert_task(src: str, out: str, overwrite: bool = False) -> dict:
    src_p, name = Path(src), Path(src).name
    out_p = Path(out) / name
    if (out_p / "meta" / "info.json").exists() and not overwrite:
        return {"bucket": name, "status": "exists"}
    shutil.rmtree(out_p, ignore_errors=True)
    info = json.loads((src_p / "meta" / "info.json").read_text())
    src_tasks = {t["task_index"]: t["task"] for t in map(json.loads, (src_p / "meta" / "tasks.jsonl").open())}
    src_eps = [json.loads(line) for line in (src_p / "meta" / "episodes.jsonl").open()]

    skipped: dict[str, int] = {}
    task_ids: dict[str, int] = {}
    rows, cum = [], 0
    for e in src_eps:
        df = _check_episode(src_p, info, e)
        if isinstance(df, str):
            skipped[df] = skipped.get(df, 0) + 1
            continue
        n, ep = len(df), int(e["episode_index"])
        chunk, file = divmod(len(rows), FILES_PER_CHUNK)
        prompts = [src_tasks[i] for i in df["task_index"]]
        o = pd.DataFrame({k: list(v) for k, v in episode_columns(df).items()})
        for c in df.columns:
            if c not in REINDEXED:  # joints (observation.state / action) kept as in the source
                o[c] = df[c].to_numpy()
        frame = np.arange(n)
        o["task_index"] = [task_ids.setdefault(p, len(task_ids)) for p in prompts]
        o["episode_index"], o["frame_index"], o["index"] = ep, frame, cum + frame
        o["timestamp"] = (frame / FPS).astype(np.float32)
        data_path = out_p / DATA_PATH.format(chunk_index=chunk, file_index=file)
        data_path.parent.mkdir(parents=True, exist_ok=True)
        o.to_parquet(data_path, index=False)

        row = {"episode_index": ep, "length": n, "tasks": sorted(set(prompts)),
               "dataset_from_index": cum, "dataset_to_index": cum + n,
               "data/chunk_index": chunk, "data/file_index": file,
               "source_frame_offset": int(df["frame_index"].iloc[0])}  # fmt: skip
        src_chunk = ep // info["chunks_size"]
        for key, s in CAMS.items():
            dst = out_p / VIDEO_PATH.format(video_key=key, chunk_index=chunk, file_index=file)
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(src_p / info["video_path"].format(episode_chunk=src_chunk, video_key=s, episode_index=ep), dst)
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
    features.update({c: info["features"][c] for c in ("observation.state", "action")})
    features.update({k: {"dtype": "int64", "shape": [1]} for k in ("task_index", "episode_index", "frame_index", "index")})
    features["timestamp"] = {"dtype": "float32", "shape": [1]}
    features.update({k: {**info["features"][s], "source": s} for k, s in CAMS.items()})
    out_info = {
        "codebase_version": "v3.0",
        "robot_type": "agilex_cobot_magic",
        "source": "AtomBench/CobotMagic (LeRobot v2.1)",
        "source_task": name,
        "fps": FPS,
        "total_episodes": len(rows),
        "total_frames": cum,
        "total_tasks": len(task_ids),
        "chunks_size": FILES_PER_CHUNK,
        "data_path": DATA_PATH,
        "video_path": VIDEO_PATH,
        "features": features,
        "pose_frame": POSE_FRAME,
        "euler": "extrinsic XYZ, R = Rz @ Ry @ Rx (verified against Piper FK)",
        "gripper_units": GRIPPER_UNITS,
        "conversion_skipped": skipped,
        "source_episodes": len(src_eps),
    }
    (meta / "info.json").write_text(json.dumps(out_info, indent=2, ensure_ascii=False))
    return {"bucket": name, "status": "ok", "episodes": len(rows), "source_episodes": len(src_eps), "frames": cum,
            "skipped": skipped}  # fmt: skip


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", required=True, help="directory of task folders (each with meta/data/videos)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--tasks", nargs="*", help="task folder names (default: all under --src)")
    ap.add_argument("--workers", type=int, default=4, help="keep low: sources are read from ossfs")
    ap.add_argument("--overwrite", action="store_true")
    a = ap.parse_args()

    src = Path(a.src)
    tasks = [src / t for t in a.tasks] if a.tasks else sorted(p.parent.parent for p in src.glob("*/meta/info.json"))
    Path(a.out).mkdir(parents=True, exist_ok=True)
    path = Path(a.out) / "conversion_report.jsonl"
    merged = {r["bucket"]: r for r in map(json.loads, path.read_text().splitlines())} if path.exists() else {}
    with ProcessPoolExecutor(a.workers) as pool:
        futs = [pool.submit(convert_task, str(t), a.out, a.overwrite) for t in tasks]
        for t, f in zip(tasks, futs):
            try:
                r = f.result()
            except Exception as err:  # noqa: BLE001 - report and keep going with the other tasks
                r = {"bucket": t.name, "status": f"error:{err!r}"[:500]}
            print(json.dumps(r, ensure_ascii=False), flush=True)
            if r["status"] != "exists":
                merged[r["bucket"]] = r
                path.write_text("".join(json.dumps(merged[b], ensure_ascii=False) + "\n" for b in sorted(merged)))


if __name__ == "__main__":
    main()
