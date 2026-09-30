import json

import numpy as np
import pandas as pd

from openwam.dataloader.utils import galaxea_convert as gc


def _episode(n: int, ep: int) -> pd.DataFrame:
    pose = np.zeros((n, 7))
    pose[:, 0] = np.arange(n) * 0.1
    pose[:, 6] = 1.0  # identity quat xyzw
    return pd.DataFrame({
        "observation.state.left_ee_pose": list(pose), "observation.state.right_ee_pose": list(pose),
        "observation.state.left_gripper": [[50.0]] * n, "observation.state.right_gripper": [[100.0]] * n,
        "action.left_gripper": [[0.0]] * n, "action.right_gripper": [[80.0]] * n,
        "observation.state.torso": [[0.1, 0.2, 0.3, 0.0]] * n,
        "timestamp": np.arange(n) / 15, "frame_index": np.arange(n), "episode_index": ep, "index": np.arange(n),
        "task_index": [0] * (n - 1) + [1], "coarse_task_index": 2, "quality_index": [3] * (n - 1) + [4],
        "coarse_quality_index": 3,
    })  # fmt: skip


def test_episode_columns_and_prompt():
    c = gc.episode_columns(_episode(3, 0))
    np.testing.assert_allclose(c["observation.state.ee_base"][:, 0], [0, 0.1, 0.2], atol=1e-6)
    np.testing.assert_allclose(c["action.ee_base"][:, 0], [0.1, 0.2, 0.2], atol=1e-6)
    np.testing.assert_allclose(c["observation.state.ee_base"][0, 3:9], [1, 0, 0, 0, 1, 0])
    np.testing.assert_allclose(c["observation.state.gripper"][0], [0.5, 1.0])
    np.testing.assert_allclose(c["action.gripper"][0], [0.0, 0.8])
    np.testing.assert_allclose(gc.gripper_to_unit(np.array([-2.0, 50.0, 103.0])), [0.0, 0.5, 1.0])
    assert gc.prompt_text("左手开灯@Turn on the light.", "x") == "Turn on the light."
    assert gc.prompt_text("null", "pick up garbage") == "pick up garbage"


def test_convert_extracted(tmp_path, monkeypatch):
    src = tmp_path / "x" / "T"
    (src / "meta").mkdir(parents=True)
    info = {"robot_type": "r1lite", "chunks_size": 1000,
            "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
            "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
            "features": {**{s: {"dtype": "video"} for s in gc.CAMS.values()},
                         "observation.state.torso": {"dtype": "float64", "shape": [4]}}}  # fmt: skip
    (src / "meta" / "info.json").write_text(json.dumps(info))
    tasks = ["左手开灯@Turn on the light.", "null", "turn on off the light", "qualified", "unqualified"]
    (src / "meta" / "tasks.jsonl").write_text("".join(json.dumps({"task_index": i, "task": t}) + "\n" for i, t in enumerate(tasks)))
    lengths = {0: 5, 1: 4}  # episode 1: video has 5 frames → skipped
    (src / "meta" / "episodes.jsonl").write_text(
        "".join(json.dumps({"episode_index": e, "length": n, "raw_file_name": f"b{e}"}) + "\n" for e, n in lengths.items())
    )
    (tmp_path / "x" / "training_data_set_meta.json").write_text(json.dumps(
        {"trainingDataSetVersion": "v1", "rawDataList": [{"name": "b0", "qualityLabel": "不合格", "qualitySubLabel": "s"}]}
    ))  # fmt: skip
    for e, n in lengths.items():
        p = src / info["data_path"].format(episode_chunk=0, episode_index=e)
        p.parent.mkdir(parents=True, exist_ok=True)
        _episode(n, e).to_parquet(p)
        for s in gc.CAMS.values():
            v = src / info["video_path"].format(episode_chunk=0, video_key=s, episode_index=e)
            v.parent.mkdir(parents=True, exist_ok=True)
            v.write_bytes(b"mp4")
    monkeypatch.setattr(gc, "_video_frames", lambda p: 5)

    out = tmp_path / "out" / "T"
    r = gc._convert_extracted(tmp_path / "x", out, "T")
    assert r["episodes"] == 1 and r["skipped"] == {"video_len_mismatch": 1}
    eps = pd.read_parquet(out / "meta" / "episodes" / "chunk-000.parquet")
    row = eps.iloc[0]
    assert (row["unqualified_frames"], row["bag_quality"], row["coarse_quality"]) == (1, "不合格", "qualified")
    df = pd.read_parquet(out / "data/chunk-000/file-000.parquet")
    texts = dict(zip(pd.read_parquet(out / "meta" / "tasks.parquet")["task_index"], pd.read_parquet(out / "meta" / "tasks.parquet").index))
    assert [texts[i] for i in df["task_index"]] == ["Turn on the light."] * 4 + ["turn on off the light"]
    assert texts[df["quality_index"].iloc[-1]] == "unqualified"
    assert "observation.state.torso" in df and len(df) == 5
    assert (out / "videos/observation.images.head/chunk-000/file-000.mp4").read_bytes() == b"mp4"
    assert json.loads((out / "meta" / "info.json").read_text())["total_frames"] == 5
