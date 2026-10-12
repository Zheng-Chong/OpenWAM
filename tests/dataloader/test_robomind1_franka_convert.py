from __future__ import annotations

import numpy as np

from openwam.dataloader.utils import robomind1_franka_convert as rc


def test_pose_conversions_and_resample():
    ee = rc.euler_pose_to_ee9(np.array([[0.5, 0.0, 0.3, 0.0, 0.0, np.pi / 2]]))
    assert ee.shape == (1, 9) and np.allclose(ee[0, :3], [0.5, 0, 0.3]) and np.allclose(ee[0, 3:9], [0, 1, 0, -1, 0, 0], atol=1e-6)
    ready = np.array([[0, -np.pi / 4, 0, -3 * np.pi / 4, 0, np.pi / 2, np.pi / 4]])
    fk = rc.franka_joints_to_ee9(ready)  # Panda "ready" pose: TCP (0.307, 0, 0.487), pointing down
    assert fk.shape == (1, 9) and np.allclose(fk[0, :3], [0.307, 0, 0.487], atol=2e-3)
    idx = rc.resample_index(473, 114)
    assert idx[0] == 0 and idx[-1] == 472 and len(idx) == 114 and (np.diff(idx) >= 4).all()
    assert np.allclose(rc.next_state(np.arange(4.0)[:, None])[:, 0], [1, 2, 3, 3])
