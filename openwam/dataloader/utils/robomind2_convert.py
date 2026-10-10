#!/usr/bin/env python3
"""Convert the single-arm tasks of RoboMIND2.0 Franka / UR5 into LeRobot v3 buckets.

Source: ``<root>/RoboMIND2.0-{Franka-Part-N,UR5}/data/{franka,ur}/<task>/success_episodes/<ts>/data/trajectory.hdf5``,
one HDF5 per episode on a *bimanual* platform (``puppet|master/{arm,end_effector}_{left,right}_*_align``,
JPEG byte arrays for 6 cameras + depth). Only tasks where exactly one arm moves are converted (``select_tasks``
on a first-episode survey, names with "both"/"two" arms dropped) and every episode is re-checked: the other arm must
stay still, otherwise the episode is skipped and listed in ``conversion_report.jsonl``.

Output: one bucket per task (``franka_<task>`` / ``ur5_<task>``) in the ``agibotworld_convert`` layout (one data parquet
+ one mp4 per camera per episode), so ``episode_quality`` applies unchanged.

Columns (active arm only; ``arm_side`` is recorded per episode):
``observation.state.ee_base`` 9-D ``[xyz, rot6d]`` from the puppet pose (quaternion read as xyzw),
``action.ee_base`` = next state (last row repeated), ``observation.state.gripper`` / ``action.gripper`` in the shared
``[0, 1]`` / 0 = closed / 1 = open convention (source: 0 = open .. 1 = closed, checked on wrist images; the action
gripper is the *master* command, state is the puppet), ``observation.state.joints`` = puppet arm joints.
Cameras: ``observation.images.head`` = ``camera_front`` (scaled to 640x360), ``observation.images.wrist`` = the
active arm's wrist camera (640x480).
FPS is an estimate: source timestamps are whole seconds, so frames / span gives ~14 Hz (Franka) and ~7 Hz (UR5).

    python -m openwam.dataloader.utils.robomind2_convert --survey /root/robomind_survey.jsonl \
        --root /mnt/data/datasets/RoboMIND2.0 --out /root/RoboMIND2.0-single-lerobotv3 --workers 4
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from openwam.dataloader.utils.eef import quat_xyzw_to_rot6d
from openwam.dataloader.utils.prompt_text import normalize_prompt

FPS = {"franka": 14, "ur5": 7}  # estimated, see module docstring
DATA_PATH = "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet"
VIDEO_PATH = "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4"
FILES_PER_CHUNK = 1000
MOVE_XYZ, MOVE_GRIP = 0.02, 0.05  # an arm "moves" above these ranges (m / gripper units)
VIDEO_KEYS = ("observation.images.head", "observation.images.wrist")
SCALE = {"observation.images.head": "640:360", "observation.images.wrist": "640:480"}


def arm_moves(xyz: np.ndarray, grip: np.ndarray) -> bool:
    return bool(np.ptp(xyz, axis=0).max() > MOVE_XYZ or np.ptp(grip) > MOVE_GRIP)


def active_side(left: tuple[np.ndarray, np.ndarray], right: tuple[np.ndarray, np.ndarray]) -> str | None:
    """'left' / 'right' when exactly one arm moves, else None. Each arm is ``(xyz (T,3), grip (T,))``."""
    ml, mr = arm_moves(*left), arm_moves(*right)
    return "left" if ml and not mr else "right" if mr and not ml else None


def gripper_open(raw: np.ndarray) -> np.ndarray:
    """Source grip (0 open .. 1 closed) → shared convention: [0, 1], 0 = closed, 1 = open."""
    return (1.0 - np.clip(np.asarray(raw, dtype=np.float64), 0.0, 1.0)).astype(np.float32)


def select_tasks(rows: list[dict]) -> list[dict]:
    """Survey rows whose first episode moves exactly one arm; names that say both/two arms are dropped."""
    out = []
    for r in rows:
        if "arms" not in r or re.search(r"both|two_arm|dual", r["task"], re.I):
            continue
        a = r["arms"]
        side = active_side(
            (np.array([[0, 0, 0], [a["left"]["ee_xyz_ptp"], 0, 0]]), np.array([0, a["left"]["grip_ptp"]])),
            (np.array([[0, 0, 0], [a["right"]["ee_xyz_ptp"], 0, 0]]), np.array([0, a["right"]["grip_ptp"]])),
        )
        if side:
            out.append({**r, "side": side})
    return out


def _jpeg_ok(b) -> bool:
    return len(b) > 2 and bytes(b[:2]) == b"\xff\xd8"


def encode_jpegs(jpegs: list[bytes], dst: Path, fps: int, scale: str, crf: int) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg", "-loglevel", "error", "-y", "-f", "image2pipe", "-c:v", "mjpeg", "-framerate", str(fps),
        "-i", "-", "-vf", f"scale={scale}", "-c:v", "libx264", "-preset", "veryfast", "-crf", str(crf), "-g", str(fps),
        "-pix_fmt", "yuv420p", "-threads", "2", "-movflags", "+faststart", str(dst),
    ]  # fmt: skip
    subprocess.run(cmd, input=b"".join(bytes(j) for j in jpegs), check=True, capture_output=True)


def _text(v) -> str:
    return v.decode() if isinstance(v, bytes) else str(v)


def convert_episode(src: str, out: str, kind: str, seq: int, crf: int) -> dict:
    """Convert one HDF5; returns an episodes-table row, or ``{"skip": reason}``."""
    import h5py

    fps = FPS[kind]
    with h5py.File(src, "r") as f:
        if "puppet" not in f or "color_images" not in f.get("camera_observations", {}):
            return {"skip": "no_puppet_or_images", "src": src}
        p, m = f["puppet"], f["master"]

        def arm(side: str):
            pose = p[f"end_effector_{side}_pose_align"]["data"][:].astype(np.float64)
            grip = p[f"end_effector_{side}_position_align"]["data"][:, 0].astype(np.float64)
            return pose, grip

        (lp, lg), (rp, rg) = arm("left"), arm("right")
        side = active_side((lp[:, :3], lg), (rp[:, :3], rg))
        if side is None:
            return {"skip": "not_single_arm", "src": src}
        pose, grip = (lp, lg) if side == "left" else (rp, rg)
        n = len(pose)
        if n < 2:
            return {"skip": "too_short", "src": src}
        ee = np.concatenate([pose[:, :3], quat_xyzw_to_rot6d(pose[:, 3:7])], axis=-1).astype(np.float32)
        g_act = m[f"end_effector_{side}_position_align"]["data"][:, 0]
        joints = p[f"arm_{side}_position_align"]["data"][:].astype(np.float32)
        text = normalize_prompt(_text(f["metadata"].attrs.get("language_instruction", "")).replace("_", " "))
        cams = {"observation.images.head": "camera_front", "observation.images.wrist": f"camera_wrist_{side}"}
        images = {}
        bad = 0
        for key, cam in cams.items():
            if cam not in f["camera_observations/color_images"]:
                return {"skip": f"missing_camera:{cam}", "src": src}
            jp = list(f["camera_observations/color_images"][cam][:])
            ok = [_jpeg_ok(b) for b in jp]
            if len(jp) != n or not any(ok):
                return {"skip": "image_length_mismatch" if len(jp) != n else "no_valid_image", "src": src}
            if not all(ok):  # keep one frame per row: borrow the nearest valid JPEG
                good = [i for i, v in enumerate(ok) if v]
                jp = [jp[i] if ok[i] else jp[min(good, key=lambda g: abs(g - i))] for i in range(n)]
                bad += len(ok) - len(good)
            images[key] = jp
    df = pd.DataFrame({
        "observation.state.ee_base": list(ee),
        "action.ee_base": list(np.concatenate([ee[1:], ee[-1:]], 0)),
        "observation.state.gripper": list(gripper_open(grip)[:, None]),
        "action.gripper": list(gripper_open(g_act)[:, None]),
        "observation.state.joints": list(joints),
        "frame_index": np.arange(n),
        "timestamp": (np.arange(n) / fps).astype(np.float32),
    })
    chunk, file = divmod(seq, FILES_PER_CHUNK)
    o = Path(out)
    data_path = o / DATA_PATH.format(chunk_index=chunk, file_index=file)
    data_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(data_path, index=False)
    row = {"src": src, "seq": seq, "length": n, "arm_side": side, "text": text, "bad_image_frames": bad,
           "data/chunk_index": chunk, "data/file_index": file}  # fmt: skip
    for key in VIDEO_KEYS:
        encode_jpegs(images[key], o / VIDEO_PATH.format(video_key=key, chunk_index=chunk, file_index=file), fps, SCALE[key], crf)
        row.update({f"videos/{key}/chunk_index": chunk, f"videos/{key}/file_index": file,
                    f"videos/{key}/from_timestamp": 0.0, f"videos/{key}/to_timestamp": n / fps})  # fmt: skip
    return row


def _convert_batch(jobs: list[tuple], out: str, kind: str, crf: int) -> list[dict]:
    rows = []
    for src, seq in jobs:
        try:
            rows.append(convert_episode(src, out, kind, seq, crf))
        except Exception as e:  # noqa: BLE001 - a broken file is reported, not fatal
            rows.append({"skip": f"error: {type(e).__name__}: {str(e)[:120]}", "src": src})
    return rows


def finalize_bucket(out: Path, kind: str, task: str, rows: list[dict]) -> None:
    """Assign episode / task / global indices (rewriting the small data parquets) and write meta."""
    rows = sorted(rows, key=lambda r: r["seq"])
    texts = sorted({r["text"] for r in rows})
    tindex = {t: i for i, t in enumerate(texts)}
    cum = 0
    for ep, r in enumerate(rows):
        path = out / DATA_PATH.format(chunk_index=r["data/chunk_index"], file_index=r["data/file_index"])
        df = pd.read_parquet(path)
        n = len(df)
        df["task_index"] = tindex[r["text"]]
        df["episode_index"] = ep
        df["index"] = cum + np.arange(n)
        df.to_parquet(path, index=False)
        r.update(episode_index=ep, dataset_from_index=cum, dataset_to_index=cum + n)
        cum += n
    meta = out / "meta"
    (meta / "episodes").mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).drop(columns=["text"]).to_parquet(meta / "episodes" / "chunk-000.parquet", index=False)
    pd.DataFrame({"task_index": range(len(texts))}, index=pd.Index(texts)).to_parquet(meta / "tasks.parquet")
    feats = {k: {"dtype": "float32", "shape": [d]} for k, d in (
        ("observation.state.ee_base", 9), ("action.ee_base", 9), ("observation.state.gripper", 1),
        ("action.gripper", 1), ("observation.state.joints", 8 if kind == "franka" else 6))}  # fmt: skip
    feats.update({k: {"dtype": "int64", "shape": [1]} for k in ("task_index", "episode_index", "frame_index", "index")})
    feats["timestamp"] = {"dtype": "float32", "shape": [1]}
    feats.update({k: {"dtype": "video", "shape": [360 if "head" in k else 480, 640, 3], "info": {"video.codec": "h264"}}
                  for k in VIDEO_KEYS})  # fmt: skip
    info = {
        "codebase_version": "v3.0", "robot_type": f"robomind2_{kind}_single_arm", "source": "RoboMIND2.0",
        "source_task": task, "fps": FPS[kind], "fps_estimated": True, "total_episodes": len(rows), "total_frames": cum,
        "total_tasks": len(texts), "chunks_size": FILES_PER_CHUNK, "data_path": DATA_PATH, "video_path": VIDEO_PATH,
        "features": feats,
        "pose_frame": "active-arm base frame, quaternion read as xyzw",
        "gripper": {"convention": "0_closed_1_open", "raw_semantics": "0_open_1_closed", "transform": "1-clip(x,0,1)"},
    }  # fmt: skip
    (meta / "info.json").write_text(json.dumps(info, indent=2, ensure_ascii=False))


def episode_dirs(task_dir: Path) -> list[Path]:
    return sorted(p for p in (task_dir / "success_episodes").iterdir() if p.is_dir())


def task_dir(root: Path, r: dict) -> Path:
    base = root / r["part"] / "data" / ("franka" if r["kind"] == "franka" else "ur")
    return base / r["task"]


def convert_task(root: Path, r: dict, out_root: Path, workers: int, crf: int, limit: int | None) -> dict:
    kind, name = r["kind"], f"{r['kind']}_{r['task']}"
    out = out_root / name
    if (out / "meta" / "info.json").exists():
        return {"bucket": name, "status": "exists"}
    shutil.rmtree(out, ignore_errors=True)
    eps = episode_dirs(task_dir(root, r))
    if limit:
        eps = eps[:limit]
    jobs = [(str(e / "data" / "trajectory.hdf5"), i) for i, e in enumerate(eps)]
    batches = [jobs[i : i + 4] for i in range(0, len(jobs), 4)]
    results = []
    with ProcessPoolExecutor(workers) as pool:
        for rows in pool.map(_convert_batch, batches, [str(out)] * len(batches), [kind] * len(batches), [crf] * len(batches)):
            results += rows
    ok = [x for x in results if "skip" not in x]
    skipped = [x for x in results if "skip" in x]
    if ok:
        finalize_bucket(out, kind, r["task"], ok)
    reasons = pd.Series([x["skip"].split(":")[0] for x in skipped]).value_counts().to_dict()
    return {"bucket": name, "status": "ok" if ok else "empty", "episodes": len(ok), "skipped": len(skipped),
            "skip_reasons": reasons, "frames": int(sum(x["length"] for x in ok)),
            "sides": pd.Series([x["arm_side"] for x in ok]).value_counts().to_dict(), "src_episodes": len(eps)}  # fmt: skip


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--survey", required=True, help="jsonl with one first-episode summary per task")
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--tasks", nargs="*", help="only these 'kind_task' names")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--crf", type=int, default=26)
    ap.add_argument("--limit", type=int, help="convert only the first N episodes per task (smoke test)")
    a = ap.parse_args()
    rows = select_tasks([json.loads(x) for x in Path(a.survey).read_text().splitlines()])
    if a.tasks:
        rows = [r for r in rows if f"{r['kind']}_{r['task']}" in a.tasks]
    out_root = Path(a.out)
    out_root.mkdir(parents=True, exist_ok=True)
    for r in rows:
        res = convert_task(Path(a.root), r, out_root, a.workers, a.crf, a.limit)
        print(json.dumps(res, ensure_ascii=False), flush=True)
        with open(out_root / "conversion_report.jsonl", "a") as f:
            f.write(json.dumps(res, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
