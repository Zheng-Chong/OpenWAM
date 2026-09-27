"""Pure-numpy checks for the G1-Dex1 EEF20 conversion and split rule."""

import os

import numpy as np
import pandas as pd
import pytest
from omegaconf import OmegaConf
from scipy.spatial.transform import Rotation

from openwam.dataloader.g1_dex1 import (
    STATE_PREFIX,
    gripper_to_open_scale,
    split_episodes,
    window_to_eef20,
)


def _window(pose_l, pose_r, grip_l, grip_r):
    row = {
        f"{STATE_PREFIX}.left_ee_pose_gripper_torso": np.asarray(pose_l, np.float32),
        f"{STATE_PREFIX}.right_ee_pose_gripper_torso": np.asarray(pose_r, np.float32),
        f"{STATE_PREFIX}.left_gripper": np.asarray([grip_l], np.float32),
        f"{STATE_PREFIX}.right_gripper": np.asarray([grip_r], np.float32),
    }
    return pd.DataFrame([row])


def test_gripper_open_scale():
    np.testing.assert_allclose(gripper_to_open_scale([0.0, 2.25, 4.5, 4.8, -0.1]), [-1, 0, 1, 1, -1])


def test_eef20_layout_and_rotation():
    rpy = [0.3, -0.2, 1.1]
    out = window_to_eef20(_window([0.3, 0.1, 0.2, *rpy], [0.4, -0.1, 0.25, 0, 0, 0], 4.5, 0.0), STATE_PREFIX)
    assert out.shape == (1, 20)
    np.testing.assert_allclose(out[0, :3], [0.3, 0.1, 0.2], atol=1e-6)
    matrix = Rotation.from_euler("xyz", rpy).as_matrix()  # extrinsic XYZ == Rz @ Ry @ Rx
    np.testing.assert_allclose(out[0, 3:9], np.concatenate([matrix[:, 0], matrix[:, 1]]), atol=1e-6)
    assert out[0, 9] == 1.0 and out[0, 19] == -1.0
    np.testing.assert_allclose(out[0, 13:19], [1, 0, 0, 0, 1, 0], atol=1e-6)


def test_split_is_disjoint_and_complete():
    eps = pd.DataFrame({"episode_index": np.arange(120)})
    train, val = split_episodes(eps, "train", 50), split_episodes(eps, "val", 50)
    assert val["episode_index"].tolist() == [0, 50, 100]
    assert len(train) + len(val) == 120


@pytest.mark.skipif(not os.environ.get("G1_DEX1_ROOT"), reason="set G1_DEX1_ROOT to the real dataset root")
def test_real_data_samples():
    from openwam.dataloader.registry import build_dataset

    cfg = OmegaConf.load("configs/dataloader/g1_dex1.yaml")
    cfg.dataset_dir = os.environ["G1_DEX1_ROOT"]
    cfg.tasks = ["G1_Dex1_ArrangePlates", "G1_Dex1_Arrange_Flowers"]
    stats = os.environ.get("G1_DEX1_STATS")
    cfg.normalize_mode = "min-max" if stats else None
    cfg.normalization_stats_path = stats
    ds = build_dataset(cfg, split="val")
    assert len(ds.buckets) == 2
    for idx in (0, len(ds) - 1):
        s = ds[idx]
        assert s["action"].shape == (32, 80) and s["proprio"].shape == (1, 80)
        valid = s["action_mask"][0].numpy()
        assert valid.sum() == 20 and valid[0:10].all() and valid[34:44].all()
        assert len(s["video"]) == 9 and s["video"][0].size == (320, 384)
        assert "@" not in s["prompt"] and s["prompt"]
        grip = s["action"][:, [9, 43]].numpy()
        assert grip.min() >= -1 and grip.max() <= 1
        if stats:
            assert np.abs(s["action"].numpy()).max() <= 1.0 + 1e-6
