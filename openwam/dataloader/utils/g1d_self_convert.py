"""Convert self-collected G1-Dex1 (joint-space) LeRobot v3 buckets to the torso-EEF schema.

The official ``unitreerobotics/G1_Dex1_*`` data that OpenWAM was mid-trained on
stores ``*_ee_pose_gripper_torso`` (torso-frame xyz + extrinsic-XYZ Euler) and a
Dex1 gripper scalar (0 closed … ~4.5 open). Our own recordings only carry arm
joint angles, so this script recomputes the same pose with forward kinematics:

    URDF chain torso_link → <side>_dex1_base_link, then +TOOL_OFFSET_X along the
    local x axis.

Verified on official G1_Dex1_ZipUp / Arrange_Flowers (joint → pose): position
error 0.000 mm and rotation error 0.000° once the 63.5 mm tool offset is applied.

Gripper: our Dex1 firmware reads ~5.4 when open (episode starts sit at 5.37–5.38),
so raw values are rescaled by 4.5/5.4 onto the official scale; the reader's
``clip(g/4.5,0,1)*2-1`` then applies unchanged.

Episodes are dropped when marked bad or when the chassis moves / the lift changes
(tabletop-only post-training). Only parquet + meta are written; videos are
referenced in place through an absolute ``video_path`` and per-episode
``from_timestamp`` (the reader derives frame offsets from it).

Usage::

    python -m openwam.dataloader.utils.g1d_self_convert \
        --src-root "/mnt/data/datasets/G1D-自采数据" --out-root /mnt/data/datasets/G1D-自采数据-eef
"""

from __future__ import annotations

import argparse
import json
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation

URDF_PATH = Path(__file__).resolve().parents[1] / "assets" / "unitree_g1_dex1.urdf"
TOOL_OFFSET_X = 0.0635  # dex1_base_link → official "gripper" point, metres along local x
RAW_OPEN_SELF = 5.4
RAW_OPEN_OFFICIAL = 4.5
ARM_JOINTS = ("shoulder_pitch", "shoulder_roll", "shoulder_yaw", "elbow", "wrist_roll", "wrist_pitch", "wrist_yaw")
SIDES = ("left", "right")

# Tabletop filter: drop the whole episode if any of these is exceeded.
MAX_BASE_XY_M = 0.01
MAX_BASE_YAW_DEG = 1.0
MAX_LIFT_RANGE = 1e-3
MAX_CHASSIS_CMD = 0.02

# Output bucket name → (source dir, schema). PourBeansEps380 is superseded by its
# _desk trim (chassis-motion prefix removed), so only the trim is converted.
SOURCES = {
    "G1D_PickKettle": ("G1D-PickKettleEps89", "flat16"),
    "G1D_PourBeans": ("G1D-PourBeansEps380_desk", "flat23"),
    "G1D_PourBeansPlus": ("G1D-PourBeansPlusEps222", "flat23"),
    "G1D_CapybaraToBox": ("G1D-capybara_to_box", "split"),
    "G1D_Pick3ObjectsToBasket": ("G1D-pick_3objects_to_basket", "split"),
    "G1D_PickBottleToBasket": ("G1D-pick_bottle_to_basket", "split"),
}

# English paraphrases ("@"-joined, same convention as the official tasks.parquet)
# for sources that ship a single instruction.
PARAPHRASES = {
    "put the capybara plush into the box": [
        "Put the capybara plush into the box.",
        "Place the capybara toy in the box.",
        "Pick up the capybara plush and drop it into the box.",
        "Move the capybara plush into the box.",
        "Put the plush capybara in the box.",
    ],
    "pick up the bottle and put it into the basket": [
        "Pick up the bottle and put it into the basket.",
        "Place the bottle in the basket.",
        "Grab the bottle and drop it into the basket.",
        "Move the bottle into the basket.",
        "Put the bottle in the basket.",
    ],
    "pick up the marker and put it into the basket": [
        "Pick up the marker and put it into the basket.",
        "Place the marker in the basket.",
        "Grab the marker and drop it into the basket.",
        "Move the marker into the basket.",
        "Put the marker pen in the basket.",
    ],
    "pick up the capybara plush and put it into the basket": [
        "Pick up the capybara plush and put it into the basket.",
        "Place the capybara toy in the basket.",
        "Grab the capybara plush and drop it into the basket.",
        "Move the capybara plush into the basket.",
        "Put the plush capybara in the basket.",
    ],
}


class G1ArmFK:
    """Numpy forward kinematics torso_link → <side> gripper point, from the URDF."""

    def __init__(self, urdf_path: Path = URDF_PATH):
        self._joints = {}
        for j in ET.parse(urdf_path).getroot().findall("joint"):
            o, a = j.find("origin"), j.find("axis")
            self._joints[j.find("child").get("link")] = (
                j.get("name"),
                j.find("parent").get("link"),
                j.get("type"),
                np.array((o.get("xyz") if o is not None else "0 0 0").split(), float),
                np.array((o.get("rpy") if o is not None else "0 0 0").split(), float),
                np.array((a.get("xyz") if a is not None else "0 0 1").split(), float),
            )

    def _chain(self, tip: str, base: str = "torso_link"):
        out = []
        while tip != base:
            out.append(self._joints[tip])
            tip = self._joints[tip][1]
        return out[::-1]

    def pose(self, side: str, q: np.ndarray) -> np.ndarray:
        """``q`` (N, 7) arm joints → (N, 6) torso-frame ``[xyz, extrinsic-XYZ rpy]``."""
        q = np.asarray(q, np.float64)
        n = len(q)
        by_name = {f"{side}_{name}_joint": q[:, i] for i, name in enumerate(ARM_JOINTS)}
        T = np.tile(np.eye(4), (n, 1, 1))
        for name, _, typ, xyz, rpy, axis in self._chain(f"{side}_dex1_base_link"):
            M = np.eye(4)
            M[:3, :3] = Rotation.from_euler("xyz", rpy).as_matrix()
            M[:3, 3] = xyz
            T = T @ M
            if typ == "revolute":
                Rj = np.tile(np.eye(4), (n, 1, 1))
                Rj[:, :3, :3] = Rotation.from_rotvec(axis[None] * by_name[name][:, None]).as_matrix()
                T = T @ Rj
        pos = T[:, :3, 3] + T[:, :3, 0] * TOOL_OFFSET_X
        return np.concatenate([pos, Rotation.from_matrix(T[:, :3, :3]).as_euler("xyz")], axis=1)


def _stack(df: pd.DataFrame, col: str) -> np.ndarray:
    return np.stack([np.atleast_1d(v) for v in df[col].values]).astype(np.float64)


def arm_and_gripper(df: pd.DataFrame, schema: str, prefix: str) -> dict:
    """``{side: (q (N,7), raw gripper (N,))}`` for ``prefix`` in {observation.state, action}."""
    if schema in ("flat16", "flat23"):
        v = _stack(df, prefix)
        grip = (14, 15) if schema == "flat16" else (17, 18)
        return {"left": (v[:, 0:7], v[:, grip[0]]), "right": (v[:, 7:14], v[:, grip[1]])}
    return {s: (_stack(df, f"{prefix}.{s}_arm.qpos"), _stack(df, f"{prefix}.{s}_ee.qpos")[:, 0]) for s in SIDES}


def episode_is_tabletop(ep: pd.DataFrame, schema: str) -> tuple[bool, str]:
    if schema == "flat16":
        return True, ""
    if schema == "flat23":
        s, a = _stack(ep, "observation.state"), _stack(ep, "action")
        xy = max(np.ptp(s[:, 19]), np.ptp(s[:, 20]))
        yaw = np.ptp(np.degrees(np.unwrap(np.arctan2(s[:, 22], s[:, 21]))))
        lift = max(np.ptp(s[:, 14]), np.ptp(a[:, 14]))
        cmd = np.abs(a[:, 15:17]).max()
    else:
        if _stack(ep, "complementary_info.is_bad").max() > 0:
            return False, "bad"
        xy, yaw = 0.0, 0.0
        lift = max(np.ptp(_stack(ep, "observation.state.torso.height")), np.abs(_stack(ep, "action.torso.qvel")).max())
        cmd = np.abs(_stack(ep, "action.chassis.qvel")).max()
    for ok, why in ((xy <= MAX_BASE_XY_M, "base_xy"), (yaw <= MAX_BASE_YAW_DEG, "base_yaw"),
                    (lift <= MAX_LIFT_RANGE, "lift"), (cmd <= MAX_CHASSIS_CMD, "chassis_cmd")):
        if not ok:
            return False, why
    return True, ""


def _task_text(text: str) -> str:
    if "@" in text:
        return text
    return "@".join(PARAPHRASES.get(text.strip(), [text]))


def convert_bucket(src: Path, out: Path, schema: str, fk: G1ArmFK) -> dict:
    info = json.loads((src / "meta" / "info.json").read_text())
    eps = pd.concat([pd.read_parquet(p) for p in sorted((src / "meta" / "episodes").rglob("*.parquet"))])
    eps = eps[[c for c in eps.columns if not c.startswith("stats/")]].sort_values("episode_index")
    data = pd.concat([pd.read_parquet(p) for p in sorted((src / "data").rglob("*.parquet"))])

    # task_index → text; tasks.parquet stores text as the index (or not at all for the _desk trim,
    # whose per-episode "tasks" column carries the paraphrases).
    tasks = pd.read_parquet(src / "meta" / "tasks.parquet")
    if "task" in tasks.columns:
        idx_to_text = dict(zip(tasks["task_index"], tasks["task"]))
    elif isinstance(tasks.index[0], str):
        idx_to_text = dict(zip(tasks["task_index"], tasks.index))
    else:
        idx_to_text = None
    if idx_to_text is None:
        texts = sorted({t for ts in eps["tasks"] for t in ts})
        ep_task = {int(e): texts.index(ts[0]) for e, ts in zip(eps["episode_index"], eps["tasks"])}
        data["task_index"] = data["episode_index"].map(ep_task)
        idx_to_text = dict(enumerate(texts))

    kept, dropped, frames = [], {}, []
    for e, ep in data.groupby("episode_index", sort=True):
        ok, why = episode_is_tabletop(ep, schema)
        if not ok:
            dropped[why] = dropped.get(why, 0) + 1
            continue
        cols = {k: ep[k].to_numpy() for k in ("timestamp", "frame_index", "episode_index", "task_index")}
        for prefix in ("observation.state", "action"):
            for side, (q, g) in arm_and_gripper(ep, schema, prefix).items():
                cols[f"{prefix}.{side}_ee_pose_gripper_torso"] = list(fk.pose(side, q).astype(np.float32))
                cols[f"{prefix}.{side}_gripper"] = list((g * RAW_OPEN_OFFICIAL / RAW_OPEN_SELF).astype(np.float32)[:, None])
        cols["observation.state.recomputed_ee_valid"] = list(np.ones((len(ep), 1), np.float32))
        frames.append(pd.DataFrame(cols))
        kept.append(int(e))
    new = pd.concat(frames, ignore_index=True)
    new["index"] = np.arange(len(new))

    if out.exists():
        shutil.rmtree(out)
    (out / "data" / "chunk-000").mkdir(parents=True)
    (out / "meta" / "episodes" / "chunk-000").mkdir(parents=True)
    new.to_parquet(out / "data" / "chunk-000" / "file-000.parquet")

    eps = eps[eps["episode_index"].isin(kept)].copy()
    lengths = new.groupby("episode_index").size()
    eps["length"] = eps["episode_index"].map(lengths).to_numpy()
    eps["data/chunk_index"], eps["data/file_index"] = 0, 0
    eps["dataset_to_index"] = eps["length"].cumsum()
    eps["dataset_from_index"] = eps["dataset_to_index"] - eps["length"]
    eps.to_parquet(out / "meta" / "episodes" / "chunk-000" / "file-000.parquet")

    used = sorted(new["task_index"].unique())
    pd.DataFrame({"task_index": used}, index=pd.Index([_task_text(idx_to_text[i]) for i in used], name="task")).to_parquet(
        out / "meta" / "tasks.parquet"
    )

    video_path = Path(info["video_path"]) if Path(info["video_path"]).is_absolute() else (src / info["video_path"]).resolve()
    image_features = {k: v for k, v in info["features"].items() if "images" in k}
    if not image_features:  # _desk trim drops image features; take them from the source it points at
        orig = Path(str(video_path).split("/videos/")[0])
        image_features = {k: v for k, v in json.loads((orig / "meta" / "info.json").read_text())["features"].items() if "images" in k}
    features = {k: v for k, v in info["features"].items() if k in ("timestamp", "frame_index", "episode_index", "index", "task_index")}
    for prefix in ("observation.state", "action"):
        for side in SIDES:
            features[f"{prefix}.{side}_ee_pose_gripper_torso"] = {"dtype": "float32", "shape": [6], "names": None}
            features[f"{prefix}.{side}_gripper"] = {"dtype": "float32", "shape": [1], "names": None}
    features["observation.state.recomputed_ee_valid"] = {"dtype": "float32", "shape": [1], "names": None}
    features.update(image_features)
    info.update(
        features=features,
        video_path=str(video_path),
        data_path="data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet",
        total_episodes=len(kept),
        total_frames=len(new),
        total_tasks=len(used),
        g1d_conversion={
            "source": str(src),
            "schema": schema,
            "fk": "urdf torso_link->dex1_base_link + 0.0635 m local x",
            "gripper_rescale": f"{RAW_OPEN_OFFICIAL}/{RAW_OPEN_SELF}",
            "dropped_episodes": dropped,
        },
    )
    info.pop("splits", None)
    (out / "meta" / "info.json").write_text(json.dumps(info, indent=2, ensure_ascii=False))
    return {"kept": len(kept), "dropped": dropped, "frames": len(new), "tasks": len(used)}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--src-root", required=True)
    p.add_argument("--out-root", required=True)
    p.add_argument("--only", nargs="*", help="subset of output bucket names")
    args = p.parse_args()
    fk = G1ArmFK()
    for name, (src, schema) in SOURCES.items():
        if args.only and name not in args.only:
            continue
        print(name, convert_bucket(Path(args.src_root) / src, Path(args.out_root) / name, schema, fk), flush=True)


if __name__ == "__main__":
    main()
