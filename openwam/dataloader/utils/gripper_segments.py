#!/usr/bin/env python3
"""Split two-arm gripper episodes into sub-task segments at gripper close/open events.

Reads only the action gripper channels of LeRobot v3 buckets (``action.{left,right}_gripper``,
or the flat 16-D G1 joint layout ``action[14:16]``) and writes per-episode event lists.
Episodes in ``meta/excluded_episodes.json`` are skipped. No data is modified.

Rules (calibrated on G1-Dex1 and the G1D self-collected tabletop data):

* Open level = per-episode gripper max; a closure starts below ``open - 0.3`` and ends above
  ``open - 0.15`` (thresholds scale with the bucket's gripper range, 4.5 or 5.4 on G1).
  Closures shorter than ``MIN_CLOSED`` frames are chatter.
* The boundary is where the gripper *starts* moving (walked back from the threshold crossing),
  for closing and opening alike.
* A closure that starts in the first ``IDLE_START`` frames and closes fully (min < 10 % of
  range) is the operator's idle pose: it and its opening are not events. An early closure that
  stops part-way is holding an object (e.g. trimmed episodes starting mid-grasp) and is kept.
* Segments shorter than ``--min-seg`` frames are merged: a short open gap ``X- .. X+`` on one
  hand is a re-grasp (both go) unless the gripper opened fully in the gap, in which case the
  release is real and the brief closure after it goes; any other short segment joins the
  previous one (the first joins the next).

Events are ``(frame, "L+" | "L-" | "R+" | "R-")``; ``+`` = start closing, ``-`` = start opening.

    python -m openwam.dataloader.utils.gripper_segments \\
        --root /root/g1d_data/eef --out /root/seg/g1d --workers 8
"""

from __future__ import annotations

import argparse
import collections
import fnmatch
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

OPEN = 4.5          # reference open level the thresholds below are written for
MIN_CLOSED = 10     # frames
MAX_BACK = 30       # frames walked back from a threshold crossing to the motion onset
IDLE_START = 30     # frames
HANDS = ("left", "right")


def closed_spans(g: np.ndarray, scale: float = OPEN) -> list[tuple[int, int]]:
    """``(close_onset, open_onset)`` frame pairs for one gripper; open_onset = len(g) if never reopened."""
    top = g.max()
    close_th, open_th = top - 0.3 * scale / OPEN, top - 0.15 * scale / OPEN
    spans, closed, start = [], False, 0
    for i, v in enumerate(g):
        if not closed and v < close_th:
            closed, start = True, i
        elif closed and v > open_th:
            closed = False
            spans.append((start, i))
    if closed:
        spans.append((start, len(g)))
    spans = [p for p in spans if p[1] - p[0] >= MIN_CLOSED]
    merged = []  # closures separated by a few open frames are one closure
    for p in spans:
        if merged and p[0] - merged[-1][1] < MIN_CLOSED:
            merged[-1] = (merged[-1][0], p[1])
        else:
            merged.append(p)

    eps = 0.01 * scale / OPEN

    def onset(i: int, sign: int) -> int:  # walk back while the gripper still moves this way
        j = i
        while j > 0 and i - j < MAX_BACK and sign * (g[j] - g[j - 1]) > eps:
            j -= 1
        return j

    spans = [(a2, max(onset(b, +1), a2 + 1) if b < len(g) else b) for a, b in merged for a2 in [onset(a, -1)]]
    return [p for p in spans if p[0] >= IDLE_START or g[p[0]:p[1]].min() > 0.1 * scale]


def merge_short(ev: list, n: int, g: dict | None = None, open_level: dict | None = None,
                min_seg: int = 15) -> list:
    """Drop boundaries until every segment between events (and episode ends) is >= ``min_seg`` frames."""
    ev = list(ev)
    while ev:
        lengths = np.diff([0] + [f for f, _ in ev] + [n])
        k = int(np.argmin(lengths))
        if lengths[k] >= min_seg:
            break
        if 0 < k < len(ev) and ev[k - 1][1][1] == "-" and ev[k][1] == ev[k - 1][1][0] + "+":
            hand = "left" if ev[k][1][0] == "L" else "right"
            if g is not None and g[hand][ev[k - 1][0]:ev[k][0] + 1].max() >= open_level[hand]:
                # real release, then a brief closure: drop that closure and its opening
                j = next((i for i in range(k + 1, len(ev)) if ev[i][1] == ev[k][1][0] + "-"), None)
                del ev[k]
                if j is not None:
                    del ev[j - 1]
            else:
                del ev[k - 1:k + 1]  # re-grasp
        else:
            del ev[0 if k == 0 else k - 1]
    return ev


def segment_episode(grips: dict[str, np.ndarray], scale: float = OPEN, min_seg: int = 15) -> tuple[list, list]:
    """Return ``(events, raw_events)`` for one episode; raw = before short-segment merging."""
    n = len(grips["left"])
    raw = sorted(
        (f, h[0].upper() + s)
        for h in HANDS
        for a, b in closed_spans(grips[h], scale)
        for f, s in ((a, "+"), (b, "-"))
        if f < n
    )
    open_level = {h: grips[h].max() - 0.15 * scale / OPEN for h in HANDS}
    return merge_short(raw, n, grips, open_level, min_seg), raw


def load_grippers(bucket: Path) -> pd.DataFrame:
    """episode_index, task_index, action.left_gripper, action.right_gripper (floats), excluded dropped."""
    files = sorted((bucket / "data").rglob("*.parquet"))
    if "action" in pq.read_schema(files[0]).names:  # flat 16-D: 7 L joints, 7 R joints, L grip, R grip
        df = pd.concat(pd.read_parquet(p, columns=["episode_index", "task_index", "action"]) for p in files)
        a = np.stack(df.pop("action").to_numpy())
        df["action.left_gripper"], df["action.right_gripper"] = a[:, 14], a[:, 15]
    else:
        cols = ["episode_index", "task_index", "action.left_gripper", "action.right_gripper"]
        df = pd.concat(pd.read_parquet(p, columns=cols) for p in files)
        for h in HANDS:  # scalar column, or one-element lists in the converted G1D data
            df[f"action.{h}_gripper"] = np.hstack(df[f"action.{h}_gripper"].to_numpy()).astype(float)
    excluded = bucket / "meta" / "excluded_episodes.json"
    if excluded.exists():
        df = df[~df.episode_index.isin(json.loads(excluded.read_text())["episode_indices"])]
    return df


def segment_bucket(bucket: Path, min_seg: int = 15) -> dict:
    df = load_grippers(bucket)
    scale = float(np.percentile(np.r_[df["action.left_gripper"], df["action.right_gripper"]], 99))
    episodes = []
    for ep, x in df.groupby("episode_index", sort=True):
        grips = {h: x[f"action.{h}_gripper"].to_numpy(float) for h in HANDS}
        ev, raw = segment_episode(grips, scale, min_seg)
        episodes.append(dict(ep=int(ep), task=int(x.task_index.iloc[0]), len=len(x),
                             pattern=" ".join(e for _, e in ev),
                             events=[[int(f), e] for f, e in ev], raw_events=[[int(f), e] for f, e in raw]))
    return dict(bucket=bucket.name, grip_scale=scale, min_seg=min_seg, episodes=episodes)


def summarize(rep: dict) -> dict:
    eps = rep["episodes"]
    patterns = collections.Counter(e["pattern"] for e in eps)
    n_raw = sum(len(e["raw_events"]) for e in eps)
    seg = np.concatenate([np.diff([0] + [f for f, _ in e["events"]] + [e["len"]]) for e in eps])
    return dict(
        episodes=len(eps), grip_scale=rep["grip_scale"],
        median_events=float(np.median([len(e["events"]) for e in eps])),
        merged_frac=1 - sum(len(e["events"]) for e in eps) / max(n_raw, 1),
        median_segment_frames=float(np.median(seg)),
        top_patterns=[[p, c] for p, c in patterns.most_common(5)],
    )


def _run(args: tuple[str, str, int]) -> tuple[str, dict]:
    bucket, out, min_seg = args
    rep = segment_bucket(Path(bucket), min_seg)
    Path(out, f"{rep['bucket']}.json").write_text(json.dumps(rep))
    return rep["bucket"], summarize(rep)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", required=True, help="directory of LeRobot v3 buckets")
    ap.add_argument("--out", required=True, help="writes <bucket>.json per bucket and summary.json")
    ap.add_argument("--buckets", default="*", help="glob on bucket names")
    ap.add_argument("--min-seg", type=int, default=15, help="frames; shorter segments are merged")
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    buckets = sorted(p for p in Path(a.root).iterdir()
                     if (p / "meta" / "info.json").exists() and fnmatch.fnmatch(p.name, a.buckets))
    Path(a.out).mkdir(parents=True, exist_ok=True)
    with ProcessPoolExecutor(a.workers) as pool:
        summary = dict(pool.map(_run, [(str(b), a.out, a.min_seg) for b in buckets]))
    Path(a.out, "summary.json").write_text(json.dumps(summary, indent=1, ensure_ascii=False))
    for name, s in summary.items():
        top = s["top_patterns"][0]
        print(f"{name:48s} n={s['episodes']:6d} merged={s['merged_frac']:4.0%} "
              f"top={top[1] / s['episodes']:4.0%} [{top[0][:40]}]")


if __name__ == "__main__":
    main()
