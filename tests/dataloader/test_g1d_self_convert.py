"""FK used to convert self-collected joint-space G1 data into the official torso-EEF schema."""

import glob
import os

import numpy as np
import pandas as pd
import pytest
from scipy.spatial.transform import Rotation

from openwam.dataloader.utils.g1d_self_convert import G1ArmFK


def test_fk_arms_are_mirror_images_at_zero():
    fk = G1ArmFK()
    left, right = fk.pose("left", np.zeros((1, 7))), fk.pose("right", np.zeros((1, 7)))
    assert np.allclose(left[0, [0, 2]], right[0, [0, 2]], atol=1e-4)
    assert left[0, 1] > 0.1 and np.isclose(left[0, 1], -right[0, 1], atol=1e-4)


@pytest.mark.skipif(not os.environ.get("G1_DEX1_ROOT"), reason="set G1_DEX1_ROOT to the official dataset root")
def test_fk_reproduces_official_torso_pose():
    path = sorted(glob.glob(os.path.join(os.environ["G1_DEX1_ROOT"], "G1_Dex1_ZipUp", "data", "**", "*.parquet"), recursive=True))[0]
    df = pd.read_parquet(path).iloc[::50]
    fk = G1ArmFK()
    for side in ("left", "right"):
        pose = fk.pose(side, np.stack(df[f"observation.state.{side}_arm"]))
        ref = np.stack(df[f"observation.state.{side}_ee_pose_gripper_torso"])
        assert np.abs(pose[:, :3] - ref[:, :3]).max() < 1e-4
        rel = Rotation.from_euler("xyz", pose[:, 3:]).inv() * Rotation.from_euler("xyz", ref[:, 3:])
        assert np.degrees(rel.magnitude()).max() < 1e-2
