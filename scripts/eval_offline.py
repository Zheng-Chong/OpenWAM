"""Open-loop offline evaluation of a checkpoint on its dataloader's val split.

For each sampled val window the model gets exactly what the policy server
would get — the first (multiview) frame, the prompt and the raw proprio — and
predicts one action chunk. The chunk is compared with the recorded actions in
physical units (bimanual EEF20 layout: per arm xyz, rot6d, gripper open-scale).
A "hold" baseline (repeat the current pose for the whole chunk) is scored on
the same windows as a reference.

Run one process per GPU, then summarize::

    for i in 0 1 2 3 4 5 6 7; do
      CUDA_VISIBLE_DEVICES=$i python scripts/eval_offline.py --ckpt-dir CKPT \
        --out OUT/shard$i.jsonl --shard $i --num-shards 8 &
    done; wait
    python scripts/eval_offline.py --summarize OUT/shard*.jsonl
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

HORIZON_BUCKETS = ((0, 8), (8, 16), (16, 24), (24, 32))
ARMS = {"left": 0, "right": 10}


def rot6d_to_matrix(r6: np.ndarray) -> np.ndarray:
    """``(..., 6)`` rot6d (first two rotation-matrix columns) → ``(..., 3, 3)``."""
    a1, a2 = r6[..., :3], r6[..., 3:6]
    b1 = a1 / np.linalg.norm(a1, axis=-1, keepdims=True)
    b2 = a2 - (b1 * a2).sum(-1, keepdims=True) * b1
    b2 = b2 / np.linalg.norm(b2, axis=-1, keepdims=True)
    return np.stack([b1, b2, np.cross(b1, b2)], axis=-1)


def chunk_errors(pred: np.ndarray, gt: np.ndarray) -> dict:
    """Per-step errors ``{metric: (T,)}`` for EEF20 chunks ``(T, 20)``."""
    out = {}
    for arm, o in ARMS.items():
        out[f"{arm}_pos_mm"] = np.linalg.norm(pred[:, o : o + 3] - gt[:, o : o + 3], axis=-1) * 1000.0
        rel = np.swapaxes(rot6d_to_matrix(pred[:, o + 3 : o + 9]), -1, -2) @ rot6d_to_matrix(gt[:, o + 3 : o + 9])
        cos = np.clip((np.trace(rel, axis1=-2, axis2=-1) - 1.0) / 2.0, -1.0, 1.0)
        out[f"{arm}_rot_deg"] = np.degrees(np.arccos(cos))
        out[f"{arm}_grip_abs"] = np.abs(pred[:, o + 9] - gt[:, o + 9])
        # Open/closed agreement, scored only where the recorded command is decisive.
        decisive = np.abs(gt[:, o + 9]) > 0.5
        out[f"{arm}_grip_acc"] = np.where(decisive, np.sign(pred[:, o + 9]) == np.sign(gt[:, o + 9]), np.nan)
    return out


def summarize_errors(errors: dict, valid_steps: int) -> dict:
    row = {}
    for name, per_step in errors.items():
        per_step = per_step[:valid_steps].astype(np.float64)
        row[name] = float(np.nanmean(per_step)) if np.isfinite(per_step).any() else None
        for lo, hi in HORIZON_BUCKETS:
            seg = per_step[lo : min(hi, valid_steps)]
            row[f"{name}@{lo}-{hi}"] = float(np.nanmean(seg)) if seg.size and np.isfinite(seg).any() else None
    return row


def select_indices(dataset, per_task: int):
    """``[(global_idx, task)]``: ``per_task`` evenly spaced windows per bucket."""
    buckets = getattr(dataset, "buckets", None) or [dataset]
    picks, start = [], 0
    for bucket in buckets:
        n = len(bucket)
        task = getattr(bucket, "_dataset_id", type(bucket).__name__)
        for local in np.unique(np.linspace(0, n - 1, num=min(per_task, n)).astype(int)):
            picks.append((start + int(local), task))
        start += n
    return picks


def run(args) -> None:
    from omegaconf import OmegaConf

    from openwam.dataloader.registry import build_dataset
    from openwam.deploy import JointInferenceEngine
    from openwam.deploy.model_loader import load_from_checkpoint_dir
    from openwam.deploy.server import merge_deploy_cfg

    train_cfg, architecture = load_from_checkpoint_dir(args.ckpt_dir, device="cuda")
    deploy_cfg = OmegaConf.load(args.deploy_config)
    OmegaConf.update(deploy_cfg, "inference.denoise_steps", args.denoise_steps)
    OmegaConf.update(deploy_cfg, "optimization.compile.enabled", args.compile)
    cfg = merge_deploy_cfg(train_cfg, deploy_cfg)
    engine = JointInferenceEngine(cfg=cfg, architecture=architecture)
    normalizer = architecture.normalizer
    if normalizer is None:
        raise RuntimeError("checkpoint has no action normalizer; offline metrics need physical units")

    dl_cfg = cfg.dataloader
    if args.dataset_dir:
        dl_cfg.dataset_dir = args.dataset_dir
    dataset = build_dataset(dl_cfg, split="val")
    picks = select_indices(dataset, args.per_task)[args.shard :: args.num_shards]

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as fh:
        for n, (idx, task) in enumerate(picks):
            sample = dataset[idx]
            valid_steps = int(sample["action_mask"].numpy().any(axis=1).sum())
            gt = normalizer.unnormalize(sample["action"].numpy())
            proprio = normalizer.unnormalize(sample["proprio"].numpy())[0]
            t0 = time.time()
            result = engine.generate(
                {
                    "first_frame_image": [sample["video"][0]],
                    "prompt": sample["prompt"],
                    "proprio": proprio,
                    "seed": args.seed,
                }
            )
            latency = time.time() - t0
            pred = np.asarray(result["actions"])
            hold = np.repeat(proprio[None], len(gt), axis=0)
            record = {
                "task": task,
                "idx": idx,
                "prompt": sample["prompt"],
                "valid_steps": valid_steps,
                "latency_s": latency,
                "model": summarize_errors(chunk_errors(pred, gt), valid_steps),
                "hold": summarize_errors(chunk_errors(hold, gt), valid_steps),
                # raw EEF20 chunks (physical units) for signed / bias analysis
                "proprio": np.round(proprio[:20], 5).tolist(),
                "pred": np.round(pred[:valid_steps, :20], 5).tolist(),
                "gt": np.round(gt[:valid_steps, :20], 5).tolist(),
            }
            fh.write(json.dumps(record) + "\n")
            fh.flush()
            print(f"[{n + 1}/{len(picks)}] {task} idx={idx} {latency:.2f}s", flush=True)


def _mean(rows, key):
    vals = [r[key] for r in rows if r.get(key) is not None]
    return float(np.mean(vals)) if vals else None


def summarize(paths) -> dict:
    records = [json.loads(line) for p in paths for line in Path(p).read_text().splitlines() if line]
    keys = list(records[0]["model"].keys())
    by_task = defaultdict(list)
    for r in records:
        by_task[r["task"]].append(r)
    summary = {
        "num_windows": len(records),
        "num_tasks": len(by_task),
        "latency_s_median": float(np.median([r["latency_s"] for r in records])),
        "model": {k: _mean([r["model"] for r in records], k) for k in keys},
        "hold": {k: _mean([r["hold"] for r in records], k) for k in keys},
        "per_task": {
            t: {k: _mean([r["model"] for r in rs], k) for k in keys if "@" not in k} for t, rs in sorted(by_task.items())
        },
    }
    headline = [k for k in keys if "@" not in k]
    print(f"{len(records)} windows / {len(by_task)} tasks, median latency {summary['latency_s_median']:.2f}s")
    print(f"{'metric':<18}{'model':>10}{'hold':>10}")
    for k in headline:
        m, h = summary["model"][k], summary["hold"][k]
        print(f"{k:<18}{m if m is None else f'{m:.3f}':>10}{h if h is None else f'{h:.3f}':>10}")
    return summary


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--summarize", nargs="+", help="jsonl shards to aggregate (skips inference)")
    p.add_argument("--summary-out", default=None, help="write the aggregate as json")
    p.add_argument("--ckpt-dir")
    p.add_argument("--out")
    p.add_argument("--dataset-dir", default=None, help="override the checkpoint's dataloader.dataset_dir")
    p.add_argument("--deploy-config", default="configs/deploy.yaml")
    p.add_argument("--per-task", type=int, default=20)
    p.add_argument("--shard", type=int, default=0)
    p.add_argument("--num-shards", type=int, default=1)
    p.add_argument("--denoise-steps", type=int, default=10)
    p.add_argument("--compile", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    if args.summarize:
        summary = summarize(args.summarize)
        if args.summary_out:
            Path(args.summary_out).write_text(json.dumps(summary, indent=2))
        return
    if not (args.ckpt_dir and args.out):
        p.error("--ckpt-dir and --out are required unless --summarize is given")
    run(args)
    # ponytail: lingering non-daemon threads/workers kept the process alive after results were written; hard exit
    sys.stdout.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
