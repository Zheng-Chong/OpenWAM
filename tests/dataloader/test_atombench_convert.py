import json

import numpy as np
import pandas as pd

from openwam.dataloader.utils import atombench_convert as ac


def _episode(n: int, ep: int) -> pd.DataFrame:
    s, a = np.zeros((n, 26)), np.zeros((n, 14))
    s[:, 6], s[:, 19] = 1.04, 0.25  # right / left gripper
    s[:, 7] = np.arange(n) * 0.1  # right x
    s[:, 20] = 0.5  # left x
    s[:, 10] = np.pi  # right roll π: rot6d = [1,0,0, 0,-1,0]
    a[:, 6], a[:, 13] = 1.06, -0.01
    return pd.DataFrame({"index": np.arange(n), "action": list(a), "observation.state": list(s), "episode_index": ep,
                         "frame_index": np.arange(n), "timestamp": np.arange(n) / 30, "task_index": 0})  # fmt: skip


def test_episode_columns():
    c = ac.episode_columns(_episode(3, 0))
    ee = c["observation.state.ee_base"]
    np.testing.assert_allclose(ee[0, :9], [0.5, 0, 0, 1, 0, 0, 0, 1, 0], atol=1e-6)  # left: identity rotation
    np.testing.assert_allclose(ee[:, 9], [0, 0.1, 0.2], atol=1e-6)  # right x
    np.testing.assert_allclose(ee[0, 12:18], [1, 0, 0, 0, -1, 0], atol=1e-6)  # right: roll π
    np.testing.assert_allclose(c["action.ee_base"][:, 9], [0.1, 0.2, 0.2], atol=1e-6)  # next state
    np.testing.assert_allclose(c["observation.state.gripper"][0], [0.25, 1.0])  # [L, R], clipped
    np.testing.assert_allclose(c["action.gripper"][0], [0.0, 1.0])


def test_convert_task(tmp_path, monkeypatch):
    src = tmp_path / "dm1_task"
    (src / "meta").mkdir(parents=True)
    info = {"chunks_size": 10000,
            "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
            "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
            "features": {**{k: {"dtype": "video"} for k in ac.CAMS.values()},
                         "observation.state": {"dtype": "float32", "shape": [26]}, "action": {"dtype": "float32", "shape": [14]}}}  # fmt: skip
    (src / "meta" / "info.json").write_text(json.dumps(info))
    (src / "meta" / "tasks.jsonl").write_text(json.dumps({"task_index": 0, "task": "Put the ball in the basket."}) + "\n")
    lengths = {0: 5, 1: 4}  # episode 1: video has 5 frames → skipped
    (src / "meta" / "episodes.jsonl").write_text("".join(json.dumps({"episode_index": e, "length": n}) + "\n" for e, n in lengths.items()))
    for e, n in lengths.items():
        p = src / info["data_path"].format(episode_chunk=0, episode_index=e)
        p.parent.mkdir(parents=True, exist_ok=True)
        d = _episode(n, e)
        d["frame_index"] += 7 * (e == 0)  # source head trimmed, index not reset
        d.to_parquet(p)
        for k in ac.CAMS.values():
            v = src / info["video_path"].format(episode_chunk=0, video_key=k, episode_index=e)
            v.parent.mkdir(parents=True, exist_ok=True)
            v.write_bytes(b"mp4")
    monkeypatch.setattr(ac, "_video_frames", lambda p: 5)

    r = ac.convert_task(str(src), str(tmp_path / "out"))
    assert r["episodes"] == 1 and r["skipped"] == {"video_len_mismatch": 1}
    out = tmp_path / "out" / "dm1_task"
    df = pd.read_parquet(out / "data/chunk-000/file-000.parquet")
    assert df["frame_index"].tolist() == [0, 1, 2, 3, 4]
    assert pd.read_parquet(out / "meta/episodes/chunk-000.parquet")["source_frame_offset"].tolist() == [7]
    assert len(df) == 5 and {"observation.state", "action", "observation.state.ee_base"} <= set(df.columns)
    assert (out / "videos/observation.images.hand_right/chunk-000/file-000.mp4").read_bytes() == b"mp4"
    meta = json.loads((out / "meta" / "info.json").read_text())
    assert (meta["total_frames"], meta["fps"], meta["codebase_version"]) == (5, 30, "v3.0")
