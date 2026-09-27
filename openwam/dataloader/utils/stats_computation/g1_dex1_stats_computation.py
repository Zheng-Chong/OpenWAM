"""Compute pooled G1-Dex1 EEF20 normalization stats over all task buckets.

Action and state rows of every train-split episode (``episode_index %
val_every != 0``) with ``recomputed_ee_valid == 1`` are pooled after the exact
reader conversion (see :mod:`openwam.dataloader.g1_dex1`). rot6d and gripper
dims are pinned to identity. The output is deploy-compatible
(``{"eef": {...}}``) and is copied into checkpoints by the trainer.

Example::

    python -m openwam.dataloader.utils.stats_computation.g1_dex1_stats_computation \
      --config configs/dataloader/g1_dex1.yaml \
      --output /path/to/g1_dex1_normalization_stats.npy
"""

from __future__ import annotations

import argparse
import os
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from omegaconf import OmegaConf

from openwam.dataloader.g1_dex1 import (
    ACTION_MODE,
    ACTION_PREFIX,
    EEF20_DIM,
    GRIPPER_DIMS_EEF20,
    ROT6D_DIMS,
    STATE_PREFIX,
    STATS_CONTRACT,
    VALID_COL,
    discover_buckets,
    needed_columns,
    window_to_eef20,
)
from openwam.dataloader.utils.normalization import pin_rot6d_identity
from openwam.dataloader.utils.stats_computation.robocoin_stats_computation import Accumulator


def _bucket_rows(bucket: Path, val_every: int):
    """Yield ``(action_eef20, state_eef20, n_invalid)`` per parquet shard of one bucket."""
    cols = list(needed_columns()) + ["episode_index"]
    for path in sorted((bucket / "data").rglob("*.parquet")):
        df = pq.read_table(path, columns=cols).to_pandas()
        valid = np.stack([np.atleast_1d(v) for v in df[VALID_COL].values])[:, 0] != 0
        keep = valid & (df["episode_index"].to_numpy() % val_every != 0)
        df = df[keep]
        yield window_to_eef20(df, ACTION_PREFIX), window_to_eef20(df, STATE_PREFIX), int((~valid).sum())


def compute_stats(buckets, val_every: int, reservoir_cap: int, workers: int) -> dict:
    acc = Accumulator(dim=EEF20_DIM, reservoir_cap=reservoir_cap)
    lock = threading.Lock()
    counts = {"rows": 0, "invalid": 0}

    def run(bucket: Path):
        for action, state, n_invalid in _bucket_rows(bucket, val_every):
            with lock:
                acc.update_batch(action)
                acc.update_batch(state)
                counts["rows"] += len(action)
                counts["invalid"] += n_invalid
        print(f"  done {bucket.name}", flush=True)

    with ThreadPoolExecutor(workers) as pool:
        list(pool.map(run, buckets))
    if counts["rows"] == 0:
        raise ValueError("G1Dex1 stats: no rows")

    stats = acc.finalize()
    pin_rot6d_identity(stats, ROT6D_DIMS)
    pin_rot6d_identity(stats, GRIPPER_DIMS_EEF20)  # gripper is already [-1, 1]: identity too
    stats.update(STATS_CONTRACT)
    stats.update(
        {
            "num_timesteps": acc.count,
            "pool": "action_and_state",
            "train_rows": counts["rows"],
            "invalid_rows_skipped": counts["invalid"],
            "val_every": val_every,
            "tasks": [b.name for b in buckets],
        }
    )
    return stats


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="configs/dataloader/g1_dex1.yaml")
    parser.add_argument("--dataset-dir", default=None, help="Override config dataset_dir")
    parser.add_argument("--output", required=True)
    parser.add_argument("--reservoir-cap", type=int, default=1_000_000)
    parser.add_argument("--workers", type=int, default=16)
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config)
    root = Path(args.dataset_dir or cfg.dataset_dir)
    buckets = discover_buckets(root, cfg.get("tasks"), cfg.get("exclude_tasks"))
    print(f"G1Dex1 stats over {len(buckets)} buckets under {root}", flush=True)
    stats = compute_stats(buckets, int(cfg.get("val_every", 50)), args.reservoir_cap, args.workers)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    tmp = output.with_name(f".{output.name}.{os.getpid()}.tmp.npy")
    np.save(tmp, {ACTION_MODE: stats}, allow_pickle=True)
    os.replace(tmp, output)
    print(
        f"wrote {output}: rows={stats['train_rows']} invalid_skipped={stats['invalid_rows_skipped']}\n"
        f"  min={np.round(stats['min'], 3).tolist()}\n  max={np.round(stats['max'], 3).tolist()}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
