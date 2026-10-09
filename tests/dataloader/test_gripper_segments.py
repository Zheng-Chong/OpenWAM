"""Gripper-event sub-task segmentation rules and the per-bucket CLI path."""

import json

import numpy as np
import pandas as pd

from openwam.dataloader.utils.gripper_segments import closed_spans, merge_short, segment_bucket, segment_episode

OPEN = np.full(600, 4.5)
LEVEL = {"left": 4.35, "right": 4.35}


def ramp(a, b, n):
    return np.linspace(a, b, n)


def test_boundaries_are_motion_onsets():
    g = np.r_[np.full(50, 4.5), ramp(4.5, 3.0, 6), np.full(60, 3.0), ramp(3.0, 4.5, 6), np.full(30, 4.5)]
    assert closed_spans(g) == [(50, 116)]


def test_early_full_closure_is_idle_but_early_partial_closure_is_a_grasp():
    idle = np.r_[np.full(5, 4.5), ramp(4.5, 0, 8), np.full(150, 0.0), ramp(0, 4.5, 6), np.full(40, 4.5),
                 ramp(4.5, 3.0, 6), np.full(60, 3.0), ramp(3.0, 4.5, 6), np.full(30, 4.5)]
    assert closed_spans(idle) == [(209, 275)]
    holding = np.r_[np.full(5, 4.5), ramp(4.5, 3.6, 8), np.full(150, 3.6), ramp(3.6, 4.5, 6), np.full(40, 4.5)]
    assert closed_spans(holding) == [(5, 163)]


def test_thresholds_follow_the_bucket_gripper_range():
    g = np.r_[np.full(50, 5.4), ramp(5.4, 3.0, 6), np.full(60, 3.0), ramp(3.0, 5.4, 6), np.full(30, 5.4)]
    assert closed_spans(g, scale=5.4) == [(50, 116)]


def test_regrasp_merges_but_full_release_survives():
    gL = OPEN.copy(); gL[100:400] = 2.0; gL[200:205] = 3.0          # gap never reopens fully
    ev = [(100, "L+"), (200, "L-"), (205, "L+"), (400, "L-")]
    assert merge_short(ev, 600, {"left": gL, "right": OPEN}, LEVEL) == [(100, "L+"), (400, "L-")]
    gR = OPEN.copy(); gR[100:200] = 2.0; gR[210:300] = 0.0          # opens fully, then closes briefly
    ev = [(100, "R+"), (200, "R-"), (210, "R+"), (300, "R-")]
    assert merge_short(ev, 600, {"left": OPEN, "right": gR}, LEVEL) == [(100, "R+"), (200, "R-")]


def test_short_segments_join_neighbours():
    assert merge_short([(10, "R+"), (300, "R-")], 600) == [(300, "R-")]
    assert merge_short([(100, "R+"), (590, "R-")], 600) == [(100, "R+")]
    assert merge_short([(100, "R+"), (110, "L+"), (400, "R-")], 600) == [(110, "L+"), (400, "R-")]


def test_segment_episode_two_hands():
    gR = np.r_[np.full(60, 4.5), ramp(4.5, 3.2, 6), np.full(300, 3.2), ramp(3.2, 4.5, 6), np.full(100, 4.5)]
    gL = np.r_[np.full(150, 4.5), ramp(4.5, 3.6, 6), np.full(250, 3.6), ramp(3.6, 4.5, 6), np.full(60, 4.5)]
    ev, raw = segment_episode({"left": gL, "right": gR})
    assert [e for _, e in ev] == ["R+", "L+", "R-", "L-"] and ev == raw


def test_segment_bucket_reads_both_layouts_and_skips_excluded(tmp_path):
    g = np.r_[np.full(50, 4.5), ramp(4.5, 3.0, 6), np.full(60, 3.0), ramp(3.0, 4.5, 6), np.full(30, 4.5)]
    n = len(g)
    named = pd.DataFrame({"episode_index": np.repeat([0, 1], n), "task_index": 0,
                          "action.left_gripper": [[v] for v in np.r_[g, g]],
                          "action.right_gripper": 4.5})
    flat = np.zeros((2 * n, 16)); flat[:, 14] = np.r_[g, g]; flat[:, 15] = 4.5
    layouts = {"named": named,
               "flat": pd.DataFrame({"episode_index": np.repeat([0, 1], n), "task_index": 0, "action": list(flat)})}
    for name, df in layouts.items():
        b = tmp_path / name
        (b / "data" / "chunk-000").mkdir(parents=True); (b / "meta").mkdir()
        df.to_parquet(b / "data" / "chunk-000" / "file-000.parquet")
        (b / "meta" / "info.json").write_text("{}")
        (b / "meta" / "excluded_episodes.json").write_text(json.dumps({"episode_indices": [1]}))
        rep = segment_bucket(b)
        assert [e["ep"] for e in rep["episodes"]] == [0]
        assert rep["episodes"][0]["events"] == [[50, "L+"], [116, "L-"]]
