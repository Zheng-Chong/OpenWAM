#!/usr/bin/env python3
"""Add ``ee_base`` / ``gripper`` columns to GenieSim 3.0 LeRobot v3 tasks, in place.

Input: unpacked ``dataset_lerobot3.0/<task>/g2_omnipicker/full`` (already v3: many episodes per data
parquet, shared mp4s sliced by ``from/to_timestamp``; robot G2 + OmniPicker, 30 fps, sim). The
``observation.state`` (186-D) and ``action`` (40-D) have **no field names** and no official layout was
found, so the slices below are inferred from the data and are NOT verified against G2 kinematics:

* ``state[14:17]`` / ``state[17:20]``: left / right end-effector xyz (m). Frame: fixed to the robot (the
  start pose is the same across tasks, x≈0.6, y≈±0.4, z≈1.0, while the robot's world position
  ``state[118:121]`` varies by metres), origin on the floor under the base, z up. Waist pose changes
  the start height / reach in some tasks.
* ``state[126:135]`` / ``state[135:144]``: left / right EE rotation matrix (orthonormal, det +1 on
  every frame). **Row-major is assumed** (``R = reshape(3, 3)``); transpose is not excluded.
* ``state[0]`` / ``state[1]``: left / right gripper opening 0..120 (OmniPicker width, mm; official
  ``omnipicker_reverse_relabel_gripper``), 0 = closed. Left/right order assumed to follow the EE order.
* ``action`` has no EE pose; ``action[0:2]`` is a gripper command in a different scale, so the action
  columns below are the next state (last row of each episode repeated).

Columns added to every data parquet (originals kept):

* ``observation.state.ee_base`` 18-D ``[L_xyz, L_rot6d, R_xyz, R_rot6d]``, rot6d = first two columns of R.
* ``action.ee_base`` = next state.
* ``observation.state.gripper`` / ``action.gripper`` ``[L, R]`` = ``clip(opening / 120, 0, 1)``, 0 = closed.

``meta/info.json`` gets the new features and ``ee_pose_unverified`` notes and is rewritten last
(``geniesim_ee`` marks a finished task; reruns skip it).

    python -m openwam.dataloader.utils.geniesim_convert --root /root/GenieSim3.0-unpacked --workers 8
"""

from __future__ import annotations

import argparse
import json
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

GRIPPER_MAX = 120.0
NEW_COLS = ("observation.state.ee_base", "action.ee_base", "observation.state.gripper", "action.gripper")
LAYOUT = {
    "ee_pos": {"left": [14, 17], "right": [17, 20]},
    "ee_rot_matrix_row_major": {"left": [126, 135], "right": [135, 144]},
    "gripper_opening_mm": {"left": 0, "right": 1},
}
NOTE = ("ee pose inferred from the unnamed 186-D observation.state, not verified with G2 kinematics: "
        "frame fixed to the robot (floor under base, z up); rotation assumed row-major; left/right assumed")  # fmt: skip


def episode_columns(state: np.ndarray, episode: np.ndarray) -> dict[str, np.ndarray]:
    """Per-frame ``state`` (n, 186) and ``episode`` ids (n,) → the four new columns."""
    state = state.astype(np.float64)
    arms = []
    for side in ("left", "right"):
        a, b = LAYOUT["ee_pos"][side]
        r0, r1 = LAYOUT["ee_rot_matrix_row_major"][side]
        rot = state[:, r0:r1].reshape(-1, 3, 3)
        arms.append(np.concatenate([state[:, a:b], rot[:, :, 0], rot[:, :, 1]], axis=-1))
    ee = np.concatenate(arms, axis=-1).astype(np.float32)
    grip = np.clip(state[:, [LAYOUT["gripper_opening_mm"][s] for s in ("left", "right")]] / GRIPPER_MAX, 0, 1)
    grip = grip.astype(np.float32)
    last = np.r_[episode[1:] != episode[:-1], True]  # last row of each episode
    nxt = np.where(last, np.arange(len(state)), np.arange(len(state)) + 1)
    return {"observation.state.ee_base": ee, "action.ee_base": ee[nxt],
            "observation.state.gripper": grip, "action.gripper": grip[nxt]}  # fmt: skip


def _fixed_list(a: np.ndarray) -> pa.Array:
    return pa.FixedSizeListArray.from_arrays(pa.array(a.reshape(-1)), a.shape[1])


def convert_task(task: str, overwrite: bool = False) -> dict:
    root = Path(task)
    info_p = root / "meta" / "info.json"
    info = json.loads(info_p.read_text())
    if info.get("geniesim_ee") and not overwrite:
        return {"task": root.name, "status": "exists"}
    frames = bad_rot = 0
    for p in sorted((root / "data").glob("chunk-*/file-*.parquet")):
        t = pq.read_table(p)
        t = t.drop_columns([c for c in NEW_COLS if c in t.column_names])
        state = np.stack(t["observation.state"].to_numpy(zero_copy_only=False))
        ep = t["episode_index"].to_numpy()
        cols = episode_columns(state, ep)
        for k in NEW_COLS:
            t = t.append_column(k, _fixed_list(cols[k]))
        rot = state[:, 126:144].reshape(-1, 2, 3, 3).astype(np.float64)
        bad_rot += int((np.abs(np.einsum("nbij,nbkj->nbik", rot, rot) - np.eye(3)).max(axis=(1, 2, 3)) > 1e-2).sum())
        frames += len(t)
        tmp = p.with_suffix(".parquet.tmp")
        pq.write_table(t, tmp)
        os.replace(tmp, p)
    info["features"].update({k: {"dtype": "float32", "shape": [18 if "ee" in k else 2]} for k in NEW_COLS})
    info.update({"ee_pose_unverified": NOTE, "state_layout_inferred": LAYOUT, "gripper_units": {
        "range": [0, 1], "convention": "0 = closed, 1 = open", "formula": "clip(state[0|1] / 120, 0, 1)"}})  # fmt: skip
    info["geniesim_ee"] = True
    info_p.write_text(json.dumps(info, ensure_ascii=False))
    return {"task": root.name, "status": "ok", "frames": frames, "episodes": info["total_episodes"], "bad_rotation_frames": bad_rot}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", required=True, help="directory of unpacked task folders")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--overwrite", action="store_true")
    a = ap.parse_args()
    tasks = sorted(str(p.parent.parent) for p in Path(a.root).glob("*/meta/info.json"))
    with ProcessPoolExecutor(a.workers) as pool:
        for r in pool.map(convert_task, tasks, [a.overwrite] * len(tasks)):
            print(json.dumps(r, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
