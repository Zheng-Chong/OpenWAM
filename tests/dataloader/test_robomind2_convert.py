from __future__ import annotations

import numpy as np

from openwam.dataloader.utils import robomind2_convert as rc


def _row(task, left, right, grip_l=0.0, grip_r=0.0, kind="franka"):
    arm = lambda x, g: {"ee_xyz_ptp": x, "grip_ptp": g}  # noqa: E731
    return {"kind": kind, "part": "p", "task": task, "arms": {"left": arm(left, grip_l), "right": arm(right, grip_r)}}


def test_select_tasks():
    rows = [
        _row("open_drawer", 0.3, 0.0),
        _row("press_button", 0.0, 0.0, grip_r=0.8),  # gripper-only motion counts
        _row("move_both", 0.3, 0.3),
        _row("lift_with_both_arms", 0.3, 0.0),  # name says both arms
        _row("idle", 0.0, 0.0),
        {"kind": "ur5", "part": "p", "task": "broken"},  # no puppet group in the survey
    ]
    sel = {r["task"]: r["side"] for r in rc.select_tasks(rows)}
    assert sel == {"open_drawer": "left", "press_button": "right"}


def test_active_side_and_gripper():
    still = (np.zeros((5, 3)), np.zeros(5))
    moving = (np.c_[np.linspace(0, 0.2, 5), np.zeros(5), np.zeros(5)], np.zeros(5))
    assert rc.active_side(moving, still) == "left" and rc.active_side(still, moving) == "right"
    assert rc.active_side(moving, moving) is None and rc.active_side(still, still) is None
    assert np.allclose(rc.gripper_open(np.array([0.0, 0.8, 1.2, -0.1])), [1.0, 0.2, 0.0, 1.0])
