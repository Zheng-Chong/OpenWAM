from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd

from openwam.dataloader.utils import episode_quality as eq
from openwam.dataloader.utils.lerobotv3 import load_excluded_episodes_snapshot


def _pose(xyz: np.ndarray) -> np.ndarray:
    pose = np.zeros((len(xyz), 18))
    pose[:, 0:3] = xyz
    pose[:, [3, 7, 12, 16]] = 1.0  # identity rot6d, both arms
    return pose


def test_pose_metrics():
    xyz = np.zeros((10, 3))
    xyz[:, 0] = np.linspace(0, 0.09, 10)  # 1 cm/step
    m = eq.pose_metrics(_pose(xyz))
    assert np.isclose(m["max_step_m"], 0.01) and np.isclose(m["path_m"], 0.09)
    assert m["max_step_deg"] < 1e-3 and m["rot6d_err"] < 1e-9

    xyz[5:, 0] += 0.3  # teleop glitch
    assert eq.pose_metrics(_pose(xyz))["max_step_m"] > 0.3

    p = _pose(np.zeros((3, 3)))
    p[2, 3:9] = [0, 1, 0, -1, 0, 0]  # 90° about z
    assert np.isclose(eq.pose_metrics(p)["max_step_deg"], 90.0)


def test_prompt_ok():
    assert eq.prompt_ok("Pick the cucumber.")
    assert eq.prompt_ok("丢垃圾") and eq.prompt_ok("使用键盘打字：I am a robot.")
    assert not eq.prompt_ok("丢")
    for bad in ("", "  ", "N/A", "task", "do something", "123", "go"):
        assert not eq.prompt_ok(bad), bad


def test_flag_and_apply(tmp_path):
    base = dict(bucket="b", nonfinite=False, max_step_m=0.01, max_step_deg=1.0, rot6d_err=0.0,
                path_m=1.0, effector_range=1.0, prompt_ok=True)
    df = pd.DataFrame([
        {**base, "episode_index": 0, "length": 300},
        {**base, "episode_index": 1, "length": 30},  # < 2 s
        {**base, "episode_index": 2, "length": 300, "max_step_m": 0.2},
        {**base, "episode_index": 3, "length": 300, "path_m": 0.0, "effector_range": 0.0},
        {**base, "episode_index": 4, "length": 300, "prompt_ok": False},
        {**base, "episode_index": 5, "length": 300, "zero_pose_frames": 3},
    ]).fillna({"zero_pose_frames": 0})
    a = argparse.Namespace(min_seconds=2.0, max_len_x_median=5.0, max_step_m=0.05, max_step_deg=20.0,
                           min_path_m=0.05, min_effector_range=0.05)
    df["reasons"] = eq.flag(df, a, {"b": 30.0})
    assert df["reasons"].tolist() == ["", "too_short", "pos_jump", "static", "bad_prompt", "zero_pose"]

    meta = tmp_path / "b" / "meta"
    meta.mkdir(parents=True)
    (meta / "excluded_episodes.json").write_text(json.dumps({"episode_indices": [9]}))
    eq.apply_exclusions(tmp_path, df)
    assert load_excluded_episodes_snapshot(tmp_path / "b").episode_indices == (1, 2, 3, 4, 5, 9)
    assert json.loads((meta / "excluded_episodes.json").read_text())["episode_quality"]["3"] == "static"


def test_delete_converter_layout(tmp_path):
    from openwam.dataloader.utils.lerobotv3 import load_episodes_parquet, resolve_lerobot_v3_data_population

    b = tmp_path / "b"
    (b / "meta" / "episodes").mkdir(parents=True)
    info = {"fps": 30, "total_episodes": 3, "total_frames": 30,
            "data_path": "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet",
            "video_path": "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4"}
    (b / "meta" / "info.json").write_text(json.dumps(info))
    rows = []
    for i, ep in enumerate((7, 8, 9)):
        (b / "data" / "chunk-000").mkdir(parents=True, exist_ok=True)
        pd.DataFrame({"x": range(10)}).to_parquet(b / f"data/chunk-000/file-{i:03d}.parquet")
        v = b / f"videos/cam/chunk-000/file-{i:03d}.mp4"
        v.parent.mkdir(parents=True, exist_ok=True)
        v.write_bytes(b"")
        rows.append({"episode_index": ep, "length": 10, "dataset_from_index": 10 * i, "data/chunk_index": 0,
                     "data/file_index": i, "videos/cam/chunk_index": 0, "videos/cam/file_index": i})
    pd.DataFrame(rows).to_parquet(b / "meta/episodes/chunk-000.parquet")

    df = pd.DataFrame({"bucket": ["b"] * 3, "episode_index": [7, 8, 9], "reasons": ["", "pos_jump", ""]})
    eq.delete_episodes(tmp_path, df)

    assert load_episodes_parquet(b)["episode_index"].tolist() == [7, 9]
    assert not (b / "data/chunk-000/file-001.parquet").exists() and not (b / "videos/cam/chunk-000/file-001.mp4").exists()
    assert json.loads((b / "meta/info.json").read_text())["total_frames"] == 20
    assert json.loads((b / "meta/deleted_episodes.json").read_text()) == {"8": "pos_jump"}
    assert resolve_lerobot_v3_data_population(b).total_rows == 20  # reader's manifest checks still pass

    eq.delete_episodes(tmp_path, pd.DataFrame({"bucket": ["b"] * 2, "episode_index": [7, 9], "reasons": ["static"] * 2}))
    assert not b.exists()  # emptied bucket removed
    assert json.loads((tmp_path / "deleted_buckets.json").read_text())["b"]["episodes"] == 2


def test_flag_official_quality_and_body_motion():
    base = dict(bucket="b", nonfinite=False, max_step_m=0.01, max_step_deg=1.0, rot6d_err=0.0, path_m=1.0,
                effector_range=1.0, prompt_ok=True, length=300, unqualified_frames=0, coarse_quality="qualified",
                bag_quality="合格", chassis_cmd_frac=0.0, torso_range=0.0)
    df = pd.DataFrame([base, {**base, "unqualified_frames": 5}, {**base, "coarse_quality": "unqualified"},
                       {**base, "bag_quality": "不合格"}, {**base, "chassis_cmd_frac": 0.3}, {**base, "torso_range": 0.4}])
    a = argparse.Namespace(min_seconds=2.0, max_len_x_median=5.0, max_step_m=0.05, max_step_deg=20.0,
                           min_path_m=0.05, min_effector_range=0.05, max_chassis_cmd_frac=None, max_torso_range=None)
    assert eq.flag(df, a, {"b": 15.0}).tolist() == ["", "official_unqualified", "official_unqualified",
                                                    "bag_unqualified", "", ""]  # body_motion is opt-in
    a.max_chassis_cmd_frac, a.max_torso_range = 0.01, 0.05
    assert eq.flag(df, a, {"b": 15.0}).tolist()[4:] == ["body_motion", "body_motion"]


def test_scan_interndata_bucket(tmp_path):
    """Nested InternData-A1 bucket: xyz + quat(wxyz) per arm, gripper in metres."""
    b = tmp_path / "basic_tasks" / "split_aloha" / "task"
    (b / "meta" / "episodes" / "chunk-000").mkdir(parents=True)
    (b / "data" / "chunk-000").mkdir(parents=True)
    info = {"codebase_version": "v3.0", "robot_type": "AgileX Split Aloha", "fps": 30,
            "total_episodes": 2, "total_frames": 200, "features": {},
            "data_path": "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet",
            "video_path": "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4"}
    (b / "meta" / "info.json").write_text(json.dumps(info))
    pd.DataFrame({"task_index": [0]}, index=["Pick up the cup."]).to_parquet(b / "meta" / "tasks.parquet")
    pd.DataFrame({"episode_index": [0, 1], "length": [100, 100], "data/chunk_index": [0, 0],
                  "data/file_index": [0, 0], "dataset_from_index": [0, 100], "dataset_to_index": [100, 200]}
                 ).to_parquet(b / "meta" / "episodes" / "chunk-000" / "file-000.parquet")
    pose = np.tile([0.3, 0.0, 0.2, 1.0, 0.0, 0.0, 0.0], (200, 1))  # identity quaternion, w first
    pose[:100, 0] += np.linspace(0, 0.2, 100)  # ep 0: smooth 20 cm reach
    grip = np.zeros((200, 1))
    grip[:100, 0] = np.linspace(0, 0.1, 100)  # ep 0 opens fully (0.1 m stroke); ep 1 static
    pose[150:, 1] += 0.3  # ep 1: 30 cm teleport
    cols = {"task_index": np.zeros(200, dtype=np.int64)}
    for side in ("left", "right"):
        cols[f"states.{side}_ee_to_robot_pose"] = list(pose.astype(np.float32))
        cols[f"states.{side}_gripper.position"] = list(grip.astype(np.float32))
    # two shards, one episode each, but the manifest (like A1's) puts both in file-000
    pd.DataFrame(cols).iloc[:100].to_parquet(b / "data" / "chunk-000" / "file-000.parquet")
    pd.DataFrame(cols).iloc[100:].to_parquet(b / "data" / "chunk-000" / "file-001.parquet")

    assert eq.find_buckets(tmp_path) == ["basic_tasks/split_aloha/task"]
    df = eq.scan_bucket(str(b))
    assert np.isclose(df.loc[0, "effector_range"], 1.0) and df.loc[0, "rot6d_err"] < 1e-6
    assert df.loc[0, "max_step_deg"] < 1e-3 and df.loc[1, "max_step_m"] > 0.29
    a = argparse.Namespace(min_seconds=2.0, max_len_x_median=5.0, max_step_m=0.10, max_step_deg=20.0,
                           min_path_m=0.05, min_effector_range=0.05)
    assert eq.flag(df, a, {"task": 30.0}).tolist() == ["", "pos_jump"]
