from __future__ import annotations

import gzip
import io
import tarfile
from pathlib import Path

import numpy as np

from openwam.dataloader.utils import robomind1_ur_convert as rc


def test_ur5e_fk_home_and_tool():
    xyz, rot = rc.ur5e_fk(np.zeros((1, 6)), tool=0.0)
    assert np.allclose(xyz[0], [-0.8172, -0.2329, 0.0628], atol=1e-3)  # UR5e at all-zero joints
    assert np.allclose(rot[0] @ rot[0].T, np.eye(3), atol=1e-9)
    xyz_t, rot_t = rc.ur5e_fk(np.zeros((1, 6)))
    assert np.allclose(xyz_t - xyz, rc.TOOL_OFFSET * rot_t[:, :, 2])  # tool offset is along the flange z axis


def test_episode_columns_and_chain_reader(tmp_path: Path):
    master = np.zeros((5, 7))
    master[:, 0] = np.linspace(0, 0.5, 5)
    puppet = np.zeros((5, 7))
    puppet[:, 6] = [0.0, 0.0, 0.8, 0.8, 0.0]  # source: 0 open .. 0.8 closed
    c = rc.episode_to_columns(master, puppet)
    assert c["observation.state.ee_base"].shape == (5, 9) and c["observation.state.joints"].shape == (5, 6)
    assert np.allclose(c["observation.state.gripper"][:, 0], [1, 1, 0.2, 0.2, 1])
    assert np.allclose(c["action.ee_base"][:-1], c["observation.state.ee_base"][1:])

    # one gzip tar split into two parts must read back through ChainReader in order
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w|gz") as t:
        data = b"x" * 5000
        ti = tarfile.TarInfo("task/success_episodes/a/data/trajectory.hdf5")
        ti.size = len(data)
        t.addfile(ti, io.BytesIO(data))
    raw = buf.getvalue()
    (tmp_path / "t.tar.gz.part-aa").write_bytes(raw[:30])
    (tmp_path / "t.tar.gz.part-ab").write_bytes(raw[30:])
    parts = sorted(tmp_path.iterdir())
    with tarfile.open(fileobj=io.BufferedReader(rc.ChainReader(parts)), mode="r|gz") as t:
        names = [m.name for m in t]
    assert names == ["task/success_episodes/a/data/trajectory.hdf5"]
    assert gzip is not None
