"""Raw AgiBotWorld-Beta episode → LeRobot v3 bucket → AgiBotWorldDataset round trip."""

from __future__ import annotations

import json

import numpy as np
import pytest
from PIL import Image

h5py = pytest.importorskip("h5py")

from openwam.dataloader import agibotworld  # noqa: E402
from openwam.dataloader.utils import agibotworld_convert as conv  # noqa: E402

N = 40
TASK = "373"
EP = "750062"


def _write_raw(root):
    ep_dir = root / "raw" / "proprio_stats" / TASK / EP
    ep_dir.mkdir(parents=True)
    t = np.arange(N, dtype=np.float64)
    pos = np.zeros((N, 2, 3))
    pos[:, 0, 0] = t  # left x encodes the frame index
    quat = np.zeros((N, 2, 4))
    quat[..., 3] = 1.0  # identity, xyzw
    with h5py.File(ep_dir / "proprio_stats.h5", "w") as f:
        f["state/end/position"] = pos
        f["state/end/orientation"] = quat
        f["state/effector/position"] = np.full((N, 2), 125.0)  # mm, fully closed
        f["action/effector/position"] = np.ones((N, 2))  # 1 = closed
        f["action/robot/velocity"] = np.tile([0.5, 0.2], (N, 1))
        f["timestamp"] = np.arange(N, dtype=np.int64)
    vdir = root / "raw" / "observations" / TASK / EP / "videos"
    vdir.mkdir(parents=True)
    for name in conv.CAMS.values():
        (vdir / name).write_bytes(b"")
    info_dir = root / "task_info"
    info_dir.mkdir()
    info = [
        {
            "episode_id": int(EP),
            "task_name": "Pickup items",
            "label_info": {"action_config": [{"start_frame": 10, "end_frame": 30, "action_text": "Pick the cucumber."}]},
        }
    ]
    (info_dir / f"task_{TASK}.json").write_text(json.dumps(info))


def test_convert_round_trip(tmp_path, monkeypatch):
    _write_raw(tmp_path)
    monkeypatch.setattr(conv, "_video_frames", lambda path: N)
    monkeypatch.setattr(
        "openwam.dataloader.bases.lerobot_v3_reader._decode_video_frames",
        lambda path, idx, h, w: [Image.new("RGB", (w, h)) for _ in idx],
    )
    out = tmp_path / "out"
    r = conv.convert_task(str(tmp_path / "raw"), str(tmp_path / "task_info"), str(out), TASK)
    assert r["status"] == "ok" and r["episodes"] == 1 and r["frames"] == N
    assert conv.convert_task(str(tmp_path / "raw"), str(tmp_path / "task_info"), str(out), TASK)["status"] == "exists"

    ds = agibotworld.AgiBotWorldDataset(
        dataset_dir=str(out / TASK), num_frames=33, video_stride=4, height=384, width=320, multiview=True,
        unify_action=True,
    )
    s = ds[5]  # window starts at frame 5
    a, p = s["action"].numpy(), s["proprio"].numpy().reshape(1, -1)
    assert s["prompt"] == "Pickup items"
    assert a[0, 0] == 6.0  # next-state relabel: action[t] = state[t+1]
    assert p[0, 0] == 5.0
    assert a[0, 9] == 0.0 and p[0, 9] == 0.0  # closed in the 0=closed/1=open convention
    np.testing.assert_allclose(a[0, 3:9], [1, 0, 0, 0, 1, 0])  # identity rot6d
    np.testing.assert_allclose(a[0, [68, 70]], [0.5, 0.2])  # base vx / yaw
    assert ds[12]["prompt"] == "Pick the cucumber."
