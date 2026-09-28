#!/usr/bin/env python3
"""Convert extracted AgiBotWorld-Beta raw episodes into the LeRobot v3 buckets
read by :mod:`openwam.dataloader.agibotworld`.

Input (extracted from the official webdataset tars)::

    <raw>/observations/<task>/<episode>/videos/{head,hand_left,hand_right}_color.mp4
    <raw>/proprio_stats/<task>/<episode>/proprio_stats.h5
    <task_info>/task_<task>.json

Output: one bucket per task, ``<out>/<task>/``; one data parquet per episode,
videos symlinked (AV1, not re-encoded). ``episode_index`` is the raw episode id.

Contract (see agibotworld.py): ``ee_base`` = ``[L_xyz, L_rot6d, R_xyz, R_rot6d]``
flange pose (quat xyzw → rot6d); ``action.ee_base`` is next-state relabeled
(last row repeated, never supervised); ``action.gripper`` raw 0=open/1=closed;
``observation.state.gripper`` mm → m; dex hands keep 12-D joint angles;
``robot_velocity`` is ``[vx, 0, yaw]`` (state has no source → zeros).
Per-frame prompts use the ``label_info`` sub-task text, else ``task_name``.

Episodes missing any of h5 / 3 videos / task_info, or whose video frame count
differs from the h5 length, are skipped and reported. ``meta/info.json`` is
written last, so a bucket without it is incomplete; reruns skip finished
buckets unless ``--overwrite``.

    python -m openwam.dataloader.utils.agibotworld_convert \
        --raw /mnt/data/datasets/agibot_world_beta_extracted \
        --task-info /mnt/data/datasets/agibot_world_beta/task_info \
        --out /mnt/data/datasets/AgiBotWorld-Beta-lerobotv3 --workers 32
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from openwam.dataloader.utils.eef import quat_xyzw_to_rot6d

FPS = 30
CAMS = {
    "observation.images.head": "head_color.mp4",
    "observation.images.hand_left": "hand_left_color.mp4",
    "observation.images.hand_right": "hand_right_color.mp4",
}
DATA_PATH = "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet"
VIDEO_PATH = "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4"
FILES_PER_CHUNK = 1000


def _ee18(pos: np.ndarray, quat: np.ndarray) -> np.ndarray:
    """(N,2,3) xyz + (N,2,4) xyzw → (N,18) [L_xyz, L_rot6d, R_xyz, R_rot6d]."""
    arms = [np.concatenate([pos[:, i], quat_xyzw_to_rot6d(quat[:, i])], axis=-1) for i in (0, 1)]
    return np.concatenate(arms, axis=-1).astype(np.float32)


def _next_state(x: np.ndarray) -> np.ndarray:
    return np.concatenate([x[1:], x[-1:]], axis=0)


def episode_columns(h5) -> dict[str, np.ndarray]:
    """Map one raw ``proprio_stats.h5`` to the reader's per-frame columns."""
    state_ee = _ee18(h5["state/end/position"][()], h5["state/end/orientation"][()])
    n = len(state_ee)
    eff_state = h5["state/effector/position"][()].astype(np.float32)
    eff_action = h5["action/effector/position"][()].astype(np.float32)
    if eff_action.shape != eff_state.shape:
        raise ValueError(f"effector action {eff_action.shape} vs state {eff_state.shape}")
    cols = {
        "observation.state.ee_base": state_ee,
        "action.ee_base": _next_state(state_ee),
    }
    if eff_state.shape[1] == 12:  # dex hand: joint angles (rad)
        cols["observation.state.dex"] = eff_state
        cols["action.dex"] = eff_action
    elif eff_state.shape[1] == 2:  # gripper: action 0=open/1=closed, state mm
        cols["observation.state.gripper"] = eff_state / 1000.0
        cols["action.gripper"] = eff_action
    else:
        raise ValueError(f"unknown effector width {eff_state.shape}")
    vel = h5["action/robot/velocity"][()]
    move = np.zeros((n, 3), dtype=np.float32)
    if vel.size:
        move[:, 0], move[:, 2] = vel[:, 0], vel[:, 1]
    cols["action.robot_velocity"] = move
    cols["observation.state.robot_velocity"] = np.zeros((n, 3), dtype=np.float32)
    for k, v in cols.items():
        if len(v) != n or not np.isfinite(v).all():
            raise ValueError(f"{k}: bad length or non-finite values")
    return cols


def frame_prompts(info: dict, n: int) -> list[str]:
    """Per-frame text: the covering sub-task ``action_text``, else ``task_name``."""
    texts = [info["task_name"].strip()] * n
    for seg in (info.get("label_info") or {}).get("action_config") or []:
        text = (seg.get("action_text") or "").strip()
        if text:
            for t in range(max(0, int(seg["start_frame"])), min(n, int(seg["end_frame"]))):
                texts[t] = text
    return texts


def _video_frames(path: Path) -> int:
    import av

    with av.open(str(path)) as c:
        return int(c.streams.video[0].frames)


def _load_episode(raw: Path, task: str, ep: str, info: dict):
    """Return ``(cols, prompts)`` or a skip-reason string."""
    import h5py

    vdir = raw / "observations" / task / ep / "videos"
    h5_path = raw / "proprio_stats" / task / ep / "proprio_stats.h5"
    if not h5_path.is_file():
        return "missing_h5"
    if not all((vdir / f).is_file() for f in CAMS.values()):
        return "missing_video"
    try:
        with h5py.File(h5_path, "r") as h5:
            cols = episode_columns(h5)
    except (OSError, KeyError, ValueError) as e:
        return f"bad_h5:{type(e).__name__}"
    n = len(cols["observation.state.ee_base"])
    try:
        frames = {_video_frames(vdir / f) for f in CAMS.values()}
    except Exception as e:  # noqa: BLE001 - any decode failure skips the episode
        return f"bad_video:{type(e).__name__}"
    if frames != {n}:
        return "video_len_mismatch"
    return cols, frame_prompts(info, n)


def _stats_block(x: np.ndarray) -> dict:
    return {
        "min": x.min(0).tolist(),
        "max": x.max(0).tolist(),
        "mean": x.mean(0).tolist(),
        "std": x.std(0).tolist(),
        "count": [int(len(x))],
    }


def convert_task(raw: str, task_info_dir: str, out: str, task: str, overwrite: bool = False) -> dict:
    raw_p, out_p = Path(raw), Path(out) / task
    if (out_p / "meta" / "info.json").exists() and not overwrite:
        return {"task": task, "status": "exists"}
    info_path = Path(task_info_dir) / f"task_{task}.json"
    if not info_path.is_file():
        return {"task": task, "status": "no_task_info"}
    ep_info = {str(e["episode_id"]): e for e in json.loads(info_path.read_text())}
    obs_dir = raw_p / "observations" / task
    eps = sorted(os.listdir(obs_dir), key=int) if obs_dir.is_dir() else []
    if out_p.exists():
        shutil.rmtree(out_p)

    skipped: dict[str, int] = {}
    task_ids: dict[str, int] = {}
    ep_rows, move_a, move_s = [], [], []
    cum = 0
    for ep in eps:
        loaded = _load_episode(raw_p, task, ep, ep_info[ep]) if ep in ep_info else "missing_task_info"
        if isinstance(loaded, str):
            skipped[loaded] = skipped.get(loaded, 0) + 1
            continue
        cols, prompts = loaded
        n = len(prompts)
        seq = len(ep_rows)
        chunk, file = divmod(seq, FILES_PER_CHUNK)
        frame = np.arange(n)
        df = pd.DataFrame({k: list(v) for k, v in cols.items()})
        df["task_index"] = [task_ids.setdefault(p, len(task_ids)) for p in prompts]
        df["episode_index"] = int(ep)
        df["frame_index"] = frame
        df["index"] = cum + frame
        df["timestamp"] = (frame / FPS).astype(np.float32)
        data_path = out_p / DATA_PATH.format(chunk_index=chunk, file_index=file)
        data_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(data_path, index=False)

        row = {
            "episode_index": int(ep),
            "length": n,
            "tasks": sorted(set(prompts)),
            "dataset_from_index": cum,
            "dataset_to_index": cum + n,
            "data/chunk_index": chunk,
            "data/file_index": file,
        }
        for key, fname in CAMS.items():
            dst = out_p / VIDEO_PATH.format(video_key=key, chunk_index=chunk, file_index=file)
            dst.parent.mkdir(parents=True, exist_ok=True)
            os.symlink((raw_p / "observations" / task / ep / "videos" / fname).resolve(), dst)
            row.update({f"videos/{key}/chunk_index": chunk, f"videos/{key}/file_index": file,
                        f"videos/{key}/from_timestamp": 0.0, f"videos/{key}/to_timestamp": n / FPS})
        ep_rows.append(row)
        move_a.append(cols["action.robot_velocity"])
        move_s.append(cols["observation.state.robot_velocity"])
        cum += n

    if not ep_rows:
        return {"task": task, "status": "empty", "skipped": skipped}

    meta = out_p / "meta"
    (meta / "episodes").mkdir(parents=True, exist_ok=True)
    pd.DataFrame(ep_rows).to_parquet(meta / "episodes" / "chunk-000.parquet", index=False)
    pd.DataFrame({"task_index": list(task_ids.values())}, index=pd.Index(list(task_ids), name="task")).to_parquet(
        meta / "tasks.parquet"
    )
    (meta / "stats.json").write_text(
        json.dumps(
            {
                "action.robot_velocity": _stats_block(np.concatenate(move_a)),
                "observation.state.robot_velocity": _stats_block(np.concatenate(move_s)),
            }
        )
    )
    is_dex = "action.dex" in cols
    effector = "dex" if is_dex else "gripper"
    feats = {
        "action.ee_base": 18,
        "observation.state.ee_base": 18,
        f"action.{effector}": 12 if is_dex else 2,
        f"observation.state.{effector}": 12 if is_dex else 2,
        "action.robot_velocity": 3,
        "observation.state.robot_velocity": 3,
    }
    features = {k: {"dtype": "float32", "shape": [d]} for k, d in feats.items()}
    features.update({k: {"dtype": "int64", "shape": [1]} for k in ("task_index", "episode_index", "frame_index", "index")})
    features["timestamp"] = {"dtype": "float32", "shape": [1]}
    features.update({k: {"dtype": "video", "shape": [480, 640, 3], "info": {"video.codec": "av1"}} for k in CAMS})
    info = {
        "codebase_version": "v3.0",
        "robot_type": "agibot_g1",
        "source": "AgiBotWorld-Beta",
        "task_name": next(iter(ep_info.values()))["task_name"],
        "fps": FPS,
        "total_episodes": len(ep_rows),
        "total_frames": cum,
        "total_tasks": len(task_ids),
        "chunks_size": FILES_PER_CHUNK,
        "data_path": DATA_PATH,
        "video_path": VIDEO_PATH,
        "features": features,
        "conversion_skipped": skipped,
        "task_info_episodes": len(ep_info),
    }
    (meta / "info.json").write_text(json.dumps(info, indent=2, ensure_ascii=False))
    return {
        "task": task,
        "status": "ok",
        "episodes": len(ep_rows),
        "task_info_episodes": len(ep_info),  # > episodes + skipped → extraction incomplete; rerun --overwrite
        "frames": cum,
        "dex": is_dex,
        "skipped": skipped,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw", required=True)
    ap.add_argument("--task-info", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--tasks", nargs="*", help="task ids (default: all under <raw>/observations)")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    tasks = args.tasks or sorted(os.listdir(Path(args.raw) / "observations"), key=int)
    Path(args.out).mkdir(parents=True, exist_ok=True)
    report = []
    with ProcessPoolExecutor(args.workers) as pool:
        futs = [pool.submit(convert_task, args.raw, args.task_info, args.out, t, args.overwrite) for t in tasks]
        for f in futs:
            r = f.result()
            report.append(r)
            print(json.dumps(r, ensure_ascii=False), flush=True)
    (Path(args.out) / "conversion_report.jsonl").write_text("".join(json.dumps(r) + "\n" for r in report))


if __name__ == "__main__":
    main()
