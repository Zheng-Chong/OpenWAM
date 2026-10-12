#!/usr/bin/env python3
"""Convert RoboMIND 1.0 Franka tarballs (real ``h5_franka_3rgb`` and ``h5_sim_franka_3rgb``) into LeRobot v3 buckets.

Same streaming scheme as ``robomind1_ur_convert`` (split ``.tar.gz.part-*`` read as one gzip stream, one HDF5 parsed per
member in memory). Source folders: ``benchmark1_{0,1,2}_compressed/{h5_franka_3rgb,h5_sim_franka_3rgb}``.

Real (``--mode real``): ``puppet/joint_position`` (T,8 = 7 joints + gripper, 0 open .. ~1 closed),
``puppet/end_effector`` (T,6 = xyz + euler, extrinsic ``xyz``: its relative rotations match Franka forward kinematics
exactly), ``master/joint_position`` (the gripper command), cameras ``camera_{top,left,right}`` (7 tasks only have
left/right) stored either as JPEG (1.1/1.2 and part of 1.0) or as raw 640x480 BGR (23 tasks of 1.0). Episodes whose
puppet is frozen while master moves are skipped (not seen in the sampled episodes, kept as a guard). fps is not stored
anywhere: ``--fps`` (default 10) is an assumption recorded as ``fps_assumed``.

Sim (``--mode sim``, Isaac Sim): ``franka/end_effector`` is **constant over every episode** (it never moves while the joints
do), so the pose is computed from ``franka/joint_position`` (T,8; gripper joint 0 open .. 0.785 closed) with Panda forward
kinematics, in the robot base frame, cameras front_external / left_external / right_external / handeye (JPEG). States are
sampled ~4.2x faster than the cameras, so they are resampled to the image count; fps comes from ``sim_time_seq``
(images / simulated seconds) scatters between 30 and 34 Hz per task, so the nominal 30 fps is used for all sim data.

Language: ``language_raw``; tasks without it (version 1.2) use ``static/RoboMIND_v1_2_instr.csv``, then the task name.
Output: one bucket per task in the ``agibotworld_convert`` layout; ``observation.state.ee_base`` 9-D (xyz + rot6d),
``action.*`` = next state (real gripper action = master command).

    python -m openwam.dataloader.utils.robomind1_franka_convert --mode real --root /mnt/data/datasets/RoboMIND \
        --out /root/RoboMIND1-Franka3rgb-lerobotv3 --workers 8
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import re
import shutil
import subprocess
import tarfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from openwam.dataloader.utils.prompt_text import normalize_prompt
from openwam.dataloader.utils.robomind1_ur_convert import ChainReader
from openwam.dataloader.utils.robomind2_convert import gripper_open

DATA_PATH = "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet"
VIDEO_PATH = "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4"
FILES_PER_CHUNK = 1000
MIN_FRAMES = 10
SIM_GRIPPER_CLOSED = 0.785  # rad, Isaac Franka sim gripper joint at fully closed
BENCHES = ("benchmark1_0_compressed", "benchmark1_1_compressed", "benchmark1_2_compressed")
KINDS = {"real": "h5_franka_3rgb", "sim": "h5_sim_franka_3rgb"}
CAM_NAMES = {  # source camera key -> feature key
    "camera_top": "top", "camera_left": "left", "camera_right": "right",
    "camera_front_external": "front", "camera_left_external": "left", "camera_right_external": "right",
    "camera_handeye": "wrist",
}  # fmt: skip


def task_files(root: Path, mode: str) -> dict[str, list[Path]]:
    out: dict[str, list[Path]] = {}
    for bench in BENCHES:
        d = root / bench / KINDS[mode]
        if not d.is_dir():
            continue
        for p in sorted(d.iterdir()):
            m = re.match(r"(.*?)\.tar\.gz(?:\.part-\w+)?$", p.name)
            if m:
                out.setdefault(m.group(1), []).append(p)
    return out


def load_instructions(root: Path) -> dict[str, str]:
    p = root / "static" / "RoboMIND_v1_2_instr.csv"
    if not p.exists():
        return {}
    rows = list(csv.reader(p.open(encoding="utf-8")))
    return {r[1]: r[2] for r in rows[1:] if len(r) >= 3}


def euler_pose_to_ee9(pose: np.ndarray) -> np.ndarray:
    """(T,6) xyz + extrinsic-xyz euler -> (T,9) xyz + rot6d."""
    from scipy.spatial.transform import Rotation

    m = Rotation.from_euler("xyz", np.nan_to_num(pose[:, 3:6])).as_matrix()
    return np.concatenate([pose[:, :3], m[:, :, 0], m[:, :, 1]], -1).astype(np.float32)


def franka_joints_to_ee9(q: np.ndarray) -> np.ndarray:
    """(T,7) Panda joint angles -> (T,9) TCP xyz + rot6d via forward kinematics."""
    from openwam.dataloader.utils.episode_quality import franka_fk

    xyz, mat = franka_fk(np.nan_to_num(q))
    return np.concatenate([xyz, mat[:, :, 0], mat[:, :, 1]], -1).astype(np.float32)


def resample_index(n_src: int, n_dst: int) -> np.ndarray:
    """Indices that map ``n_dst`` image frames onto a ``n_src``-long state stream (first and last frame kept)."""
    return np.round(np.linspace(0, n_src - 1, n_dst)).astype(int)


def next_state(x: np.ndarray) -> np.ndarray:
    return np.concatenate([x[1:], x[-1:]], 0)


def _is_jpeg(b) -> bool:
    return len(b) > 2 and bytes(b[:2]) == b"\xff\xd8"


def encode_frames(frames: list, dst: Path, fps: int, crf: int) -> tuple[int, int]:
    """JPEG or raw 640x480 BGR frames -> mp4 (width capped at 640); returns the written (height, width)."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    if _is_jpeg(frames[0]):
        from PIL import Image

        w, h = Image.open(io.BytesIO(bytes(frames[0]))).size
        src = ["-f", "image2pipe", "-c:v", "mjpeg", "-framerate", str(fps), "-i", "-"]
    else:
        w, h = 640, 480
        if len(frames[0]) != w * h * 3:
            raise ValueError(f"unexpected raw frame size {len(frames[0])}")
        src = ["-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{w}x{h}", "-framerate", str(fps), "-i", "-"]
    ow = min(640, w)
    oh = int(round(h * ow / w / 2) * 2)
    cmd = ["ffmpeg", "-loglevel", "error", "-y", *src, "-vf", f"scale={ow}:{oh}", "-c:v", "libx264", "-preset", "veryfast",
           "-crf", str(crf), "-g", str(fps), "-pix_fmt", "yuv420p", "-threads", "2", "-movflags", "+faststart", str(dst)]  # fmt: skip
    subprocess.run(cmd, input=b"".join(bytes(f) for f in frames), check=True, capture_output=True)
    return oh, ow


def _text(v) -> str:
    return v.decode() if isinstance(v, bytes) else str(v)


def parse_real(f) -> dict | None:
    puppet, master = f["puppet"], f["master"]
    pj, ee = puppet["joint_position"][:], puppet["end_effector"][:]
    mj = master["joint_position"][:]
    if np.ptp(pj[:, :7], axis=0).max() < 1e-6 and np.ptp(mj[:, :7], axis=0).max() > 1e-3:
        return {"skip": "puppet_frozen"}
    return {"ee": euler_pose_to_ee9(ee), "joints": pj[:, :7].astype(np.float32),
            "grip": gripper_open(pj[:, 7]), "grip_act": gripper_open(mj[:, 7]), "n": len(pj), "idx": None}  # fmt: skip


def parse_sim(f, n_img: int) -> dict | None:
    jp = f["franka/joint_position"][:]
    ij = resample_index(len(jp), n_img)
    grip = np.clip(jp[ij, 7] / SIM_GRIPPER_CLOSED, 0.0, 1.0)
    st = f["sim_time_seq"][:] if "sim_time_seq" in f else None
    fps = float(n_img / (st[-1] - st[0])) if st is not None and len(st) > 1 and st[-1] > st[0] else None
    g = (1.0 - grip).astype(np.float32)
    return {"ee": franka_joints_to_ee9(jp[ij, :7]), "joints": jp[ij, :7].astype(np.float32), "grip": g, "grip_act": next_state(g),
            "n": n_img, "idx": None, "fps_est": fps}  # fmt: skip


def convert_member(raw: bytes, name: str, out: Path, seq: int, mode: str, fps: int, crf: int, cams0: list | None,
                   instr: dict, task: str) -> dict:
    import h5py

    with h5py.File(io.BytesIO(raw), "r") as f:
        cams = sorted(f["observations/rgb_images"].keys())
        if cams0 is not None and cams != cams0:
            return {"skip": "camera_mismatch", "name": name}
        imgs = {c: list(f[f"observations/rgb_images/{c}"][:]) for c in cams}
        n_img = len(imgs[cams[0]])
        if mode == "real":
            if "puppet" not in f:
                return {"skip": "no_puppet", "name": name}
            d = parse_real(f)
        else:
            if "franka" not in f:
                return {"skip": "no_franka_group", "name": name}
            d = parse_sim(f, n_img)
        if "skip" in d:
            return {"skip": d["skip"], "name": name}
        n = d["n"]
        if n < MIN_FRAMES or any(len(v) != n for v in imgs.values()):
            return {"skip": "too_short" if n < MIN_FRAMES else "image_length_mismatch", "name": name}
        text = _text(f["language_raw"][0]) if "language_raw" in f else instr.get(task, task)
    text = normalize_prompt(text.replace("_", " "))
    ep_fps = fps  # dataset-wide: real = assumed 10, sim = nominal 30 (sim_time_seq estimates scatter 30-34)
    df = pd.DataFrame({
        "observation.state.ee_base": list(d["ee"]), "action.ee_base": list(next_state(d["ee"])),
        "observation.state.gripper": list(d["grip"][:, None]), "action.gripper": list(d["grip_act"][:, None]),
        "observation.state.joints": list(d["joints"]),
        "frame_index": np.arange(n), "timestamp": (np.arange(n) / ep_fps).astype(np.float32),
    })  # fmt: skip
    chunk, file = divmod(seq, FILES_PER_CHUNK)
    path = out / DATA_PATH.format(chunk_index=chunk, file_index=file)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    row = {"name": name, "seq": seq, "length": n, "text": text, "fps_ep": ep_fps, "data/chunk_index": chunk,
           "data/file_index": file, "cams": cams}  # fmt: skip
    shapes = {}
    for c in cams:
        key = f"observation.images.{CAM_NAMES[c]}"
        h, w = encode_frames(imgs[c], out / VIDEO_PATH.format(video_key=key, chunk_index=chunk, file_index=file), ep_fps, crf)
        shapes[key] = [h, w, 3]
        row.update({f"videos/{key}/chunk_index": chunk, f"videos/{key}/file_index": file,
                    f"videos/{key}/from_timestamp": 0.0, f"videos/{key}/to_timestamp": n / ep_fps})  # fmt: skip
    row["shapes"] = shapes
    return row


def finalize_bucket(out: Path, task: str, rows: list[dict], mode: str, fps: int) -> None:
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
    pd.DataFrame(rows).drop(columns=["text", "cams", "shapes"]).to_parquet(meta / "episodes" / "chunk-000.parquet", index=False)
    pd.DataFrame({"task_index": range(len(texts))}, index=pd.Index(texts)).to_parquet(meta / "tasks.parquet")
    feats = {k: {"dtype": "float32", "shape": [d]} for k, d in (
        ("observation.state.ee_base", 9), ("action.ee_base", 9), ("observation.state.gripper", 1),
        ("action.gripper", 1), ("observation.state.joints", 7))}  # fmt: skip
    feats.update({k: {"dtype": "int64", "shape": [1]} for k in ("task_index", "episode_index", "frame_index", "index")})
    feats["timestamp"] = {"dtype": "float32", "shape": [1]}
    feats.update({k: {"dtype": "video", "shape": s, "info": {"video.codec": "h264"}} for k, s in rows[0]["shapes"].items()})
    sim = mode == "sim"
    info_fps = fps
    info = {
        "codebase_version": "v3.0", "robot_type": f"robomind1_franka_{'sim' if sim else '3rgb'}",
        "source": f"RoboMIND 1.0 {KINDS[mode]}", "source_task": task, "fps": info_fps, "fps_assumed": True, "total_episodes": len(rows), "total_frames": cum, "total_tasks": len(texts),
        "chunks_size": FILES_PER_CHUNK, "data_path": DATA_PATH, "video_path": VIDEO_PATH, "features": feats,
        "pose_frame": "Franka base frame, TCP by forward kinematics from joint_position (sim end_effector is constant)"
        if sim else "Franka base frame; xyz + extrinsic-xyz euler from puppet/end_effector",
        "gripper": {"convention": "0_closed_1_open", "transform": "1-clip(x/0.785,0,1)" if sim else "1-clip(x,0,1)"},
    }  # fmt: skip
    (meta / "info.json").write_text(json.dumps(info, indent=2, ensure_ascii=False))


def convert_task(task: str, files: list[str], out_root: str, mode: str, fps: int, crf: int, limit: int | None,
                 instr: dict) -> dict:
    out = Path(out_root) / f"{'frsim' if mode == 'sim' else 'fr3'}_{task}"
    if (out / "meta" / "info.json").exists():
        return {"bucket": out.name, "status": "exists"}
    shutil.rmtree(out, ignore_errors=True)
    rows, skipped, seq, cams0 = [], [], 0, None
    try:
        with tarfile.open(fileobj=io.BufferedReader(ChainReader([Path(p) for p in files]), 1 << 22), mode="r|gz") as t:
            for m in t:
                if not (m.isfile() and m.name.endswith(".hdf5")) or "success_episodes" not in m.name:  # 1.2 sim names it <ts>.hdf5
                    continue
                try:
                    r = convert_member(t.extractfile(m).read(), m.name, out, seq, mode, fps, crf, cams0, instr, task)
                except Exception as e:  # noqa: BLE001 - one broken member is reported, not fatal
                    r = {"skip": f"error: {type(e).__name__}: {str(e)[:100]}", "name": m.name}
                if "skip" not in r and cams0 is None:
                    cams0 = r["cams"]
                (skipped if "skip" in r else rows).append(r)
                seq += 1
                if limit and len(rows) >= limit:
                    break
    except Exception as e:  # noqa: BLE001 - truncated tarball: keep what was read, report it
        skipped.append({"skip": f"stream_error: {type(e).__name__}: {str(e)[:100]}", "name": ""})
    if rows:
        finalize_bucket(out, task, rows, mode, fps)
    reasons = pd.Series([s["skip"].split(":")[0] for s in skipped], dtype=object).value_counts().to_dict()
    return {"bucket": out.name, "status": "ok" if rows else "empty", "episodes": len(rows),
            "frames": int(sum(r["length"] for r in rows)), "cams": rows[0]["cams"] if rows else None,
            "fps": rows[0]["fps_ep"] if rows else None, "skipped": len(skipped), "skip_reasons": reasons}  # fmt: skip


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", choices=("real", "sim"), required=True)
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--tasks", nargs="*", help="only these task names")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--fps", type=int, help="default: 10 for real (assumed), 30 for sim (nominal, ~sim_time_seq estimate)")
    ap.add_argument("--crf", type=int, default=26)
    ap.add_argument("--limit", type=int, help="stop after N converted episodes per task (smoke test)")
    a = ap.parse_args()
    a.fps = a.fps or (30 if a.mode == "sim" else 10)
    root = Path(a.root)
    tasks = task_files(root, a.mode)
    if a.tasks:
        tasks = {k: v for k, v in tasks.items() if k in a.tasks}
    instr = load_instructions(root)
    Path(a.out).mkdir(parents=True, exist_ok=True)
    with ProcessPoolExecutor(a.workers) as pool:
        futs = {t: pool.submit(convert_task, t, [str(p) for p in f], a.out, a.mode, a.fps, a.crf, a.limit, instr)
                for t, f in tasks.items()}
        for t, fut in futs.items():
            try:
                res = fut.result()
            except Exception as e:  # noqa: BLE001
                res = {"bucket": t, "status": f"error: {type(e).__name__}: {str(e)[:120]}"}
            print(json.dumps(res, ensure_ascii=False), flush=True)
            with open(Path(a.out) / "conversion_report.jsonl", "a") as fh:
                fh.write(json.dumps(res, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
