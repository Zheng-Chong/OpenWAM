import numpy as np

from openwam.dataloader.utils import hy_embodied_convert as hy


def test_episode_columns():
    state = np.zeros((3, 16))
    state[:, 0] = [0.1, 0.2, 0.3]  # left x
    state[:, [6, 14]] = 1.0  # identity quats (xyzw)
    state[:, 7], state[:, 15] = 40.0, 60.0
    action = np.full((3, 2), 5.0)
    c = hy.episode_columns(state, action)
    np.testing.assert_allclose(c["observation.state.ee_base"][:, 0], [0.1, 0.2, 0.3], rtol=1e-6)
    np.testing.assert_allclose(c["action.ee_base"][:, 0], [0.2, 0.3, 0.3], rtol=1e-6)  # next state, last repeated
    np.testing.assert_allclose(c["observation.state.ee_base"][0, 3:9], [1, 0, 0, 0, 1, 0])
    np.testing.assert_allclose(c["observation.state.gripper"][0], [40, 60])
    np.testing.assert_allclose(c["action.gripper"][0], [5, 5])


def test_fill_bad_jpegs():
    j = [b"\x00", b"\xff\xd8A", b"\xff\xd8B", b"", b"\xff\xd8C"]
    out, n = hy.fill_bad_jpegs(j)
    assert n == 2 and out == [b"\xff\xd8A", b"\xff\xd8A", b"\xff\xd8B", b"\xff\xd8B", b"\xff\xd8C"]
    assert hy.fill_bad_jpegs(j[1:3]) == (j[1:3], 0)


def test_leading_invalid_frames():
    state = np.ones((4, 16))
    state[0, 0:3] = 0  # tracking loss on frame 0
    imgs = {"a": [b"\xff\xd8", b"\x00", b"\xff\xd8", b"\x00"]}  # frame 1 bad too; frame 3 bad mid/late
    assert hy.leading_invalid_frames(state, imgs) == 2
    assert hy.leading_invalid_frames(np.ones((2, 16)), {"a": [b"\xff\xd8"] * 2}) == 0


def test_write_tasks(tmp_path):
    import pandas as pd

    pd.DataFrame({"task_index": [0, 1], "task": ["丢垃圾", "抓取方块"]}).to_parquet(tmp_path / "src.parquet")
    hy.write_tasks(tmp_path / "src.parquet", tmp_path / "dst.parquet")
    t = pd.read_parquet(tmp_path / "dst.parquet")
    assert list(t.columns) == ["task_index"] and t.index.tolist() == ["丢垃圾", "抓取方块"]


def test_fill_unknown_tasks():
    ti, n = hy.fill_unknown_tasks(np.array([9, 3, 3, 9, 9, 5, 9]), unknown=9)
    assert n == 4 and ti.tolist() == [3, 3, 3, 3, 3, 5, 5]
    assert hy.fill_unknown_tasks(np.array([9, 9]), 9)[1] == 0  # all unknown: untouched, bad_prompt flags it
    assert hy.fill_unknown_tasks(np.array([1, 2]), None)[1] == 0
