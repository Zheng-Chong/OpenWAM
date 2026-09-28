from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd

from openwam.dataloader.utils import episode_quality as eq
from openwam.dataloader.utils.lerobotv3 import load_excluded_episodes_snapshot


def _pose(xyz: np.ndarray) -> np.ndarray:
    pose = np.zeros((len(xyz), 18))
    pose[:, 0:3] = xyz
    pose[:, [3, 7, 12, 16]] = 1.0  # identity rot6d, both arms
    return pose


def test_pose_metrics():
    xyz = np.zeros((10, 3))
    xyz[:, 0] = np.linspace(0, 0.09, 10)  # 1 cm/step
    m = eq.pose_metrics(_pose(xyz))
    assert np.isclose(m["max_step_m"], 0.01) and np.isclose(m["path_m"], 0.09)
    assert m["max_step_deg"] < 1e-3 and m["rot6d_err"] < 1e-9

    xyz[5:, 0] += 0.3  # teleop glitch
    assert eq.pose_metrics(_pose(xyz))["max_step_m"] > 0.3

    p = _pose(np.zeros((3, 3)))
    p[2, 3:9] = [0, 1, 0, -1, 0, 0]  # 90° about z
    assert np.isclose(eq.pose_metrics(p)["max_step_deg"], 90.0)


def test_prompt_ok():
    assert eq.prompt_ok("Pick the cucumber.")
    for bad in ("", "  ", "N/A", "task", "do something", "123", "go"):
        assert not eq.prompt_ok(bad), bad


def test_flag_and_apply(tmp_path):
    base = dict(bucket="b", nonfinite=False, max_step_m=0.01, max_step_deg=1.0, rot6d_err=0.0,
                path_m=1.0, effector_range=1.0, prompt_ok=True)
    df = pd.DataFrame([
        {**base, "episode_index": 0, "length": 300},
        {**base, "episode_index": 1, "length": 30},  # < 2 s
        {**base, "episode_index": 2, "length": 300, "max_step_m": 0.2},
        {**base, "episode_index": 3, "length": 300, "path_m": 0.0, "effector_range": 0.0},
        {**base, "episode_index": 4, "length": 300, "prompt_ok": False},
    ])
    a = argparse.Namespace(min_seconds=2.0, max_len_x_median=5.0, max_step_m=0.05, max_step_deg=20.0,
                           min_path_m=0.05, min_effector_range=0.05)
    df["reasons"] = eq.flag(df, a, {"b": 30.0})
    assert df["reasons"].tolist() == ["", "too_short", "pos_jump", "static", "bad_prompt"]

    meta = tmp_path / "b" / "meta"
    meta.mkdir(parents=True)
    (meta / "excluded_episodes.json").write_text(json.dumps({"episode_indices": [9]}))
    eq.apply_exclusions(tmp_path, df)
    assert load_excluded_episodes_snapshot(tmp_path / "b").episode_indices == (1, 2, 3, 4, 9)
    assert json.loads((meta / "excluded_episodes.json").read_text())["episode_quality"]["3"] == "static"
