#!/usr/bin/env python3
"""Convert RoboMIND 1.0 ``h5_ur_1rgb`` (UR5e, single arm) tarballs into LeRobot v3 buckets, streaming.

Source: ``<root>/benchmark1_{0,1}_compressed/h5_ur_1rgb/<task>.tar.gz.part-aa, -ab, ...`` (or one ``<task>.tar.gz``).
The parts are one gzip stream, so each task is read once, in order, with ``tarfile`` in stream mode (no extraction to
disk); every ``success_episodes/**/data/trajectory.hdf5`` member is parsed from memory.

Each HDF5 holds ``master|puppet/joint_position`` (T,7: six joints + gripper), ``puppet/end_effector`` (T,6), JPEG
``observations/rgb_images/camera_top`` (480x640, stored with ordinary RGB colors) and PNG depth (dropped). There are
no timestamps: ``--fps`` (default 10, the cadence RoboMIND's own tooling assumes) is an assumption recorded as
``fps_assumed``.

Many episodes have a **frozen puppet** (its joints and end effector never change while ``master`` moves), so the pose is
always computed from the ``master`` joints with UR5e forward kinematics (standard DH + a 0.174 m tool offset along the
flange z axis; this reproduces the file's own ``puppet/end_effector`` xyz to ~1 mm on episodes where the puppet works).
The gripper (``puppet`` joint 7: 0 open .. ~0.9 closed) is mapped to the shared 0 = closed / 1 = open convention.
Episodes shorter than 10 frames (stubs) or with an undecodable image are skipped and listed in the report.

Output: one bucket per task (``ur1_<task>``) in the ``agibotworld_convert`` layout with the single camera
``observation.images.head``; ``observation.state.ee_base`` is 9-D (xyz + rot6d), ``action.*`` is the next state.

    python -m openwam.dataloader.utils.robomind1_ur_convert --root /mnt/data/datasets/RoboMIND \
        --out /root/RoboMIND1-UR5e-lerobotv3 --workers 4
"""

from __future__ import annotations

import argparse
import io
import json
import re
import shutil
import tarfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from openwam.dataloader.utils.prompt_text import normalize_prompt
from openwam.dataloader.utils.robomind2_convert import encode_jpegs, gripper_open

DATA_PATH = "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet"
VIDEO_PATH = "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4"
FILES_PER_CHUNK = 1000
VIDEO_KEY = "observation.images.head"
MIN_FRAMES = 10
TOOL_OFFSET = 0.174  # m along the flange z axis; fitted against puppet/end_effector (rms ~1 mm)
# UR5e standard DH rows (a, d, alpha)
UR5E_DH = ((0, 0.1625, np.pi / 2), (-0.425, 0, 0), (-0.3922, 0, 0), (0, 0.1333, np.pi / 2),
           (0, 0.0997, -np.pi / 2), (0, 0.0996, 0))  # fmt: skip


def ur5e_fk(q: np.ndarray, tool: float = TOOL_OFFSET) -> tuple[np.ndarray, np.ndarray]:
    """(T,6) joint angles -> TCP xyz (T,3) and rotation matrices (T,3,3) in the robot base frame."""
    T = np.broadcast_to(np.eye(4), (len(q), 4, 4)).copy()
    for (a, d, al), th in zip(UR5E_DH, q.T):
        ct, st, ca, sa = np.cos(th), np.sin(th), np.cos(al), np.sin(al)
        A = np.zeros((len(q), 4, 4))
        A[:, 0, 0], A[:, 0, 1], A[:, 0, 2], A[:, 0, 3] = ct, -st * ca, st * sa, a * ct
        A[:, 1, 0], A[:, 1, 1], A[:, 1, 2], A[:, 1, 3] = st, ct * ca, -ct * sa, a * st
        A[:, 2, 1], A[:, 2, 2], A[:, 2, 3], A[:, 3, 3] = sa, ca, d, 1
        T = T @ A
    return T[:, :3, 3] + tool * T[:, :3, 2], T[:, :3, :3]


class ChainReader(io.RawIOBase):
    """Concatenate split files (``part-aa``, ``part-ab`` ...) into one readable stream."""

    def __init__(self, paths: list[Path]):
        self._paths, self._i, self._f = list(paths), 0, None

    def readable(self) -> bool:
        return True

    def readinto(self, b) -> int:
        while True:
            if self._f is None:
                if self._i >= len(self._paths):
                    return 0
                self._f = open(self._paths[self._i], "rb")
                self._i += 1
            n = self._f.readinto(b)
            if n:
                return n
            self._f.close()
            self._f = None


def task_files(root: Path) -> dict[str, list[Path]]:
    """``{task: [part files in order]}`` over benchmark1_0 / benchmark1_1."""
    out: dict[str, list[Path]] = {}
    for bench in ("benchmark1_0_compressed", "benchmark1_1_compressed"):
        d = root / bench / "h5_ur_1rgb"
        if not d.is_dir():
            continue
        for p in sorted(d.iterdir()):
            m = re.match(r"(.*?)\.tar\.gz(?:\.part-\w+)?$", p.name)
            if m:
                out.setdefault(m.group(1), []).append(p)
    return out


def episode_to_columns(master: np.ndarray, puppet: np.ndarray) -> dict[str, np.ndarray]:
    """Per-frame columns from ``master`` / ``puppet`` joint_position (both (T,7))."""
    xyz, rot = ur5e_fk(master[:, :6])
    ee = np.concatenate([xyz, rot[:, :, 0], rot[:, :, 1]], axis=-1).astype(np.float32)
    g = gripper_open(puppet[:, 6])[:, None]
    return {
        "observation.state.ee_base": ee,
        "action.ee_base": np.concatenate([ee[1:], ee[-1:]], 0),
        "observation.state.gripper": g,
        "action.gripper": np.concatenate([g[1:], g[-1:]], 0),
        "observation.state.joints": master[:, :6].astype(np.float32),
    }


def _good_jpeg(b) -> bool:
    return len(b) > 2 and bytes(b[:2]) == b"\xff\xd8"


def convert_member(raw: bytes, name: str, out: Path, seq: int, fps: int, crf: int) -> dict:
    import h5py

    with h5py.File(io.BytesIO(raw), "r") as f:
        master, puppet = f["master/joint_position"][:], f["puppet/joint_position"][:]
        n = len(master)
        if n < MIN_FRAMES:
            return {"skip": "too_short_stub", "name": name}
        jpegs = [bytes(b) for b in f["observations/rgb_images/camera_top"][:]]
        if len(jpegs) != n or not all(_good_jpeg(b) for b in jpegs):
            return {"skip": "bad_images", "name": name}
        text = normalize_prompt(
            (lambda v: v.decode() if isinstance(v, bytes) else str(v))(f["language_raw"][0]).replace("_", " ")
        )
    cols = episode_to_columns(master.astype(np.float64), puppet.astype(np.float64))
    df = pd.DataFrame({k: list(v) for k, v in cols.items()})
    df["frame_index"] = np.arange(n)
    df["timestamp"] = (np.arange(n) / fps).astype(np.float32)
    chunk, file = divmod(seq, FILES_PER_CHUNK)
    path = out / DATA_PATH.format(chunk_index=chunk, file_index=file)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    encode_jpegs(jpegs, out / VIDEO_PATH.format(video_key=VIDEO_KEY, chunk_index=chunk, file_index=file), fps, "640:480", crf)
    puppet_still = bool(np.ptp(puppet[:, :6], axis=0).max() < 1e-6 and np.ptp(master[:, :6], axis=0).max() > 1e-3)
    parts = name.split("/")
    split = parts[parts.index("success_episodes") + 1] if "success_episodes" in parts and len(parts) > parts.index("success_episodes") + 3 else ""
    return {
        "name": name, "seq": seq, "length": n, "text": text, "puppet_frozen": puppet_still, "split": split,
        "data/chunk_index": chunk, "data/file_index": file,
        f"videos/{VIDEO_KEY}/chunk_index": chunk, f"videos/{VIDEO_KEY}/file_index": file,
        f"videos/{VIDEO_KEY}/from_timestamp": 0.0, f"videos/{VIDEO_KEY}/to_timestamp": n / fps,
    }  # fmt: skip


def finalize_bucket(out: Path, task: str, rows: list[dict], fps: int) -> None:
    rows = sorted(rows, key=lambda r: r["seq"])
    texts = sorted({r["text"] for r in rows})
    tindex = {t: i for i, t in enumerate(texts)}
    cum = 0
    for ep, r in enumerate(rows):
        path = out / DATA_PATH.format(chunk_index=r["data/chunk_index"], file_index=r["data/file_index"])
        df = pd.read_parquet(path)
        df["task_index"], df["episode_index"], df["index"] = tindex[r["text"]], ep, cum + np.arange(len(df))
        df.to_parquet(path, index=False)
        r.update(episode_index=ep, dataset_from_index=cum, dataset_to_index=cum + len(df))
        cum += len(df)
    meta = out / "meta"
    (meta / "episodes").mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).drop(columns=["text"]).to_parquet(meta / "episodes" / "chunk-000.parquet", index=False)
    pd.DataFrame({"task_index": range(len(texts))}, index=pd.Index(texts)).to_parquet(meta / "tasks.parquet")
    feats = {k: {"dtype": "float32", "shape": [d]} for k, d in (
        ("observation.state.ee_base", 9), ("action.ee_base", 9), ("observation.state.gripper", 1),
        ("action.gripper", 1), ("observation.state.joints", 6))}  # fmt: skip
    feats.update({k: {"dtype": "int64", "shape": [1]} for k in ("task_index", "episode_index", "frame_index", "index")})
    feats["timestamp"] = {"dtype": "float32", "shape": [1]}
    feats[VIDEO_KEY] = {"dtype": "video", "shape": [480, 640, 3], "info": {"video.codec": "h264"}}
    info = {
        "codebase_version": "v3.0", "robot_type": "robomind1_ur5e", "source": "RoboMIND 1.0 h5_ur_1rgb",
        "source_task": task, "fps": fps, "fps_assumed": True, "total_episodes": len(rows), "total_frames": cum,
        "total_tasks": len(texts), "chunks_size": FILES_PER_CHUNK, "data_path": DATA_PATH, "video_path": VIDEO_PATH,
        "features": feats,
        "pose_frame": "UR5e base frame, TCP = flange + 0.174 m along z, from master joints (puppet is often frozen)",
        "gripper": {"convention": "0_closed_1_open", "raw_semantics": "0_open_~0.9_closed", "transform": "1-clip(x,0,1)"},
    }  # fmt: skip
    (meta / "info.json").write_text(json.dumps(info, indent=2, ensure_ascii=False))


def convert_task(task: str, files: list[str], out_root: str, fps: int, crf: int, limit: int | None) -> dict:
    out = Path(out_root) / f"ur1_{task}"
    if (out / "meta" / "info.json").exists():
        return {"bucket": out.name, "status": "exists"}
    shutil.rmtree(out, ignore_errors=True)
    rows, skipped, seq = [], [], 0
    try:
        with tarfile.open(fileobj=io.BufferedReader(ChainReader([Path(p) for p in files]), 1 << 22), mode="r|gz") as t:
            for m in t:
                if not (m.isfile() and m.name.endswith("trajectory.hdf5")) or "success_episodes" not in m.name:
                    continue
                try:
                    r = convert_member(t.extractfile(m).read(), m.name, out, seq, fps, crf)
                except Exception as e:  # noqa: BLE001 - one broken member is reported, not fatal
                    r = {"skip": f"error: {type(e).__name__}: {str(e)[:100]}", "name": m.name}
                (skipped if "skip" in r else rows).append(r)
                seq += 1
                if limit and len(rows) >= limit:
                    break
    except Exception as e:  # noqa: BLE001 - truncated tarball: keep what was read, report it
        skipped.append({"skip": f"stream_error: {type(e).__name__}: {str(e)[:100]}", "name": ""})
    if rows:
        finalize_bucket(out, task, rows, fps)
    reasons = pd.Series([s["skip"].split(":")[0] for s in skipped], dtype=object).value_counts().to_dict()
    return {"bucket": out.name, "status": "ok" if rows else "empty", "episodes": len(rows), "frames": int(sum(r["length"] for r in rows)),
            "puppet_frozen": int(sum(r["puppet_frozen"] for r in rows)), "skipped": len(skipped), "skip_reasons": reasons}  # fmt: skip


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--tasks", nargs="*", help="only these task names")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--fps", type=int, default=10)
    ap.add_argument("--crf", type=int, default=26)
    ap.add_argument("--limit", type=int, help="stop after N converted episodes per task (smoke test)")
    a = ap.parse_args()
    tasks = task_files(Path(a.root))
    if a.tasks:
        tasks = {k: v for k, v in tasks.items() if k in a.tasks}
    Path(a.out).mkdir(parents=True, exist_ok=True)
    with ProcessPoolExecutor(a.workers) as pool:
        futs = {t: pool.submit(convert_task, t, [str(p) for p in f], a.out, a.fps, a.crf, a.limit) for t, f in tasks.items()}
        for t, fut in futs.items():
            try:
                res = fut.result()
            except Exception as e:  # noqa: BLE001
                res = {"bucket": f"ur1_{t}", "status": f"error: {type(e).__name__}: {str(e)[:120]}"}
            print(json.dumps(res, ensure_ascii=False), flush=True)
            with open(Path(a.out) / "conversion_report.jsonl", "a") as f:
                f.write(json.dumps(res, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
