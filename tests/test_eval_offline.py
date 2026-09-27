"""Metric math of scripts/eval_offline.py (no model, no data)."""

import importlib.util
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

_spec = importlib.util.spec_from_file_location(
    "eval_offline", Path(__file__).resolve().parents[1] / "scripts" / "eval_offline.py"
)
eval_offline = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(eval_offline)


def _eef20(rotvec_left, pos_left, grip_left):
    m = Rotation.from_rotvec(rotvec_left).as_matrix()
    arm_l = np.concatenate([pos_left, m[:, 0], m[:, 1], [grip_left]])
    arm_r = np.array([0.4, -0.1, 0.2, 1, 0, 0, 0, 1, 0, -1.0])
    return np.concatenate([arm_l, arm_r])[None].repeat(32, 0)


def test_errors_match_known_offsets():
    gt = _eef20([0, 0, 0], [0.3, 0.1, 0.2], 1.0)
    pred = _eef20([0, 0, np.radians(10)], [0.3, 0.1, 0.23], -1.0)
    row = eval_offline.summarize_errors(eval_offline.chunk_errors(pred, gt), valid_steps=32)
    assert abs(row["left_pos_mm"] - 30.0) < 1e-6
    assert abs(row["left_rot_deg"] - 10.0) < 1e-4
    assert row["left_grip_acc"] == 0.0 and row["right_grip_acc"] == 1.0
    assert row["right_pos_mm"] == 0.0 and row["right_rot_deg"] < 1e-4


def test_valid_steps_truncate():
    gt = _eef20([0, 0, 0], [0.3, 0.1, 0.2], 1.0)
    pred = gt.copy()
    pred[8:, 0] += 1.0
    row = eval_offline.summarize_errors(eval_offline.chunk_errors(pred, gt), valid_steps=8)
    assert row["left_pos_mm"] == 0.0 and row["left_pos_mm@8-16"] is None
