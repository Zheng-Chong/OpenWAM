"""Build the openpi training copy of a RoboDojo G1-D LeRobot v3 export (schema sim16: [left arm 7, left gripper,
right arm 7, right gripper]): meta copied, videos symlinked, data parquet rewritten with the gripper actions (the
expert's 0/1 commands) replaced by the next frame's measured opening, the episode's last frame by its own. That is the
label the OpenWAM G1D sim runs train on (`g1d_self_convert` sim16) and what remote_policy.py's --close-below expects.
Arm actions are unchanged.

    python3 prepare_data.py --src <export dir> --dst $HF_LEROBOT_HOME/g1d_sz_pour_beans_cup_or_mug_500
"""

import argparse
import os
from pathlib import Path
import shutil

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

GRIPPERS = [7, 15]


def relabel(table: pa.Table) -> pa.Table:
    ep = table["episode_index"].to_numpy()
    frame = table["frame_index"].to_numpy()
    starts = np.r_[True, ep[1:] != ep[:-1]]
    if not (np.all(frame[starts] == 0) and np.all(np.diff(frame)[~starts[1:]] == 1)):
        raise ValueError("rows are not whole episodes in frame order")
    state = np.stack(table["observation.state"].to_numpy(zero_copy_only=False))
    action = np.stack(table["action"].to_numpy(zero_copy_only=False))
    nxt = np.arange(len(ep)) + 1
    nxt[np.r_[starts[1:], True]] -= 1
    action[:, GRIPPERS] = state[nxt][:, GRIPPERS]
    col = table.schema.get_field_index("action")
    return table.set_column(col, table.schema.field(col), pa.array(list(action.astype(np.float32)), type=table.schema.field(col).type))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--src", type=Path, required=True)
    p.add_argument("--dst", type=Path, required=True)
    args = p.parse_args()
    src, dst = args.src.resolve(), args.dst
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src / "meta", dst / "meta")
    dst.mkdir(parents=True, exist_ok=True)
    os.symlink(src / "videos", dst / "videos")
    files = sorted((src / "data").rglob("*.parquet"))
    tables = [pq.read_table(f) for f in files]
    table = relabel(pa.concat_tables(tables))  # an episode may span two data files
    offset = 0
    for f, t in zip(files, tables):
        out = dst / f.relative_to(src)
        out.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(table.slice(offset, len(t)), out)
        offset += len(t)
    grip = np.stack(table["action"].to_numpy(zero_copy_only=False))[:, GRIPPERS]
    print(f"{len(files)} data files, {len(table)} frames, {len(np.unique(table['episode_index']))} episodes; "
          f"gripper action range {grip.min():.3f}..{grip.max():.3f}")


if __name__ == "__main__":
    main()
