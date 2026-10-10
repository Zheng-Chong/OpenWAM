import json

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from openwam.dataloader.utils import geniesim_convert as gc


def _state(n: int) -> np.ndarray:
    s = np.zeros((n, 186), dtype=np.float32)
    s[:, 0], s[:, 1] = 120.0, 30.0  # left open, right 1/4
    s[:, 14] = np.arange(n) * 0.1  # left x
    s[:, 17:20] = [0.6, -0.4, 1.0]
    rot_l = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1.0]])  # 90° about z
    s[:, 126:135] = rot_l.reshape(-1)
    s[:, 135:144] = np.eye(3).reshape(-1)
    return s


def test_episode_columns():
    c = gc.episode_columns(np.concatenate([_state(3), _state(2)]), np.array([0, 0, 0, 1, 1]))
    ee = c["observation.state.ee_base"]
    np.testing.assert_allclose(ee[:3, 0], [0, 0.1, 0.2], atol=1e-6)
    np.testing.assert_allclose(ee[0, 3:9], [0, 1, 0, -1, 0, 0])  # columns of the row-major matrix
    np.testing.assert_allclose(ee[0, 9:18], [0.6, -0.4, 1.0, 1, 0, 0, 0, 1, 0])
    np.testing.assert_allclose(c["action.ee_base"][:, 0], [0.1, 0.2, 0.2, 0.1, 0.1], atol=1e-6)  # next, per episode
    np.testing.assert_allclose(c["observation.state.gripper"][0], [1.0, 0.25])


def test_convert_task(tmp_path):
    root = tmp_path / "T"
    (root / "meta").mkdir(parents=True)
    (root / "data" / "chunk-000").mkdir(parents=True)
    (root / "meta" / "info.json").write_text(json.dumps({"total_episodes": 2, "features": {}}))
    s = np.concatenate([_state(3), _state(2)])
    t = pa.table({"observation.state": pa.FixedSizeListArray.from_arrays(pa.array(s.reshape(-1)), 186),
                  "episode_index": pa.array([0, 0, 0, 1, 1])})  # fmt: skip
    pq.write_table(t, root / "data/chunk-000/file-000.parquet")
    r = gc.convert_task(str(root))
    assert r["frames"] == 5 and r["bad_rotation_frames"] == 0
    out = pq.read_table(root / "data/chunk-000/file-000.parquet")
    assert set(gc.NEW_COLS) <= set(out.column_names) and "observation.state" in out.column_names
    assert json.loads((root / "meta/info.json").read_text())["features"]["observation.state.ee_base"]["shape"] == [18]
    assert gc.convert_task(str(root))["status"] == "exists"
