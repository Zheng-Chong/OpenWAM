"""Unitree G1 + Dex1 gripper LeRobot v3 dataloader (multi-task).

Source: the ``unitreerobotics/G1_Dex1_*`` LeRobot v3.0 datasets whose schema
carries torso-frame end-effector poses (``*_ee_pose_gripper_torso``). Each
task is one bucket under ``dataset_dir``; incompatible buckets (LeRobot v2.1,
joint-only) are skipped at discovery time.

Per arm the raw columns are::

    <prefix>.<side>_ee_pose_gripper_torso  [x, y, z, roll, pitch, yaw]
    <prefix>.<side>_gripper                [gripper_pos]

with ``prefix`` = ``action`` (commanded target at row t) or
``observation.state`` (achieved). Positions are metres in the torso frame;
Euler angles are extrinsic XYZ radians (``R = Rz @ Ry @ Rx``) — verified by
composing ``state_torso`` with the torso-frame pose, which reproduces the
base-frame pose exactly only under this convention. Dex1 ``gripper_pos`` is
``0 = closed`` … ``~4.5 = open``.

Both streams are converted to the canonical bimanual EEF20::

    [L_xyz3, L_rot6d6, L_open1, R_xyz3, R_rot6d6, R_open1]

where ``open = clip(gripper_pos / 4.5, 0, 1) * 2 - 1`` (-1 closed, +1 open).
EEF20 is min-max normalized with one pooled stats file shared by every task
(rot6d and gripper dims pinned to identity), then scattered into the α 80-D
slots ``["0-9", "34-43"]``.

Split: episodes with ``episode_index % val_every == 0`` form ``val``; the rest
form ``train``. Normalization stats are computed on ``train`` only.
"""

from __future__ import annotations

import json
import logging
import random
from pathlib import Path
from typing import Any, ClassVar, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from openwam.dataloader.bases import LeRobotV3Reader
from openwam.dataloader.bases.lerobot_v3_reader import _CONFIG_MISSING
from openwam.dataloader.bases.multi_lerobot_v3_reader import MultiLeRobotV3Reader
from openwam.dataloader.utils.eef import euler_xyz_to_rot6d
from openwam.dataloader.utils.lerobotv3 import build_multibucket, load_episodes_parquet
from openwam.dataloader.utils.normalization import (
    ROT6D_DIMS_EEF20,
    apply_normalization,
    load_stats_file,
    load_stats_metadata,
)

logger = logging.getLogger(__name__)

ACTION_MODE = "eef"
EEF20_DIM = 20
GRIPPER_DIMS_EEF20 = (9, 19)
RAW_GRIPPER_OPEN = 4.5
SIDES = ("left", "right")
POSE_COL = "{prefix}.{side}_ee_pose_gripper_torso"
GRIP_COL = "{prefix}.{side}_gripper"
VALID_COL = "observation.state.recomputed_ee_valid"
ACTION_PREFIX = "action"
STATE_PREFIX = "observation.state"

# Written into the stats file and checked on load, so stats built under a
# different conversion can never be silently reused.
STATS_CONTRACT = {
    "pose_frame": "torso",
    "rotation_convention": "extrinsic_xyz_radians_r_equals_rz_ry_rx",
    "gripper_transform": "clip(gripper_pos/4.5,0,1)*2-1",
    "gripper_convention": "minus1_closed_plus1_open",
}


def gripper_to_open_scale(gripper_pos: np.ndarray) -> np.ndarray:
    """Dex1 ``gripper_pos`` (0 closed … ~4.5 open) → ``[-1 closed, +1 open]``."""
    return (np.clip(np.asarray(gripper_pos, np.float32) / RAW_GRIPPER_OPEN, 0.0, 1.0) * 2.0 - 1.0).astype(np.float32)


def _column(win: pd.DataFrame, name: str, width: int) -> np.ndarray:
    values = np.stack([np.atleast_1d(v) for v in win[name].values]).astype(np.float32)
    if values.shape[1] != width:
        raise ValueError(f"G1Dex1 column {name!r} must be {width}-D, got {values.shape}")
    return values


def window_to_eef20(win: pd.DataFrame, prefix: str) -> np.ndarray:
    """Convert one parquet window to raw (un-normalized) ``(T, 20)`` EEF20."""
    arms = []
    for side in SIDES:
        pose = _column(win, POSE_COL.format(prefix=prefix, side=side), 6)
        grip = _column(win, GRIP_COL.format(prefix=prefix, side=side), 1)
        arms += [pose[:, :3], euler_xyz_to_rot6d(pose[:, 3:6]).astype(np.float32), gripper_to_open_scale(grip)]
    out = np.concatenate(arms, axis=-1)
    if not np.isfinite(out).all():
        raise ValueError("G1Dex1 EEF window contains NaN/inf")
    return out


def needed_columns() -> Tuple[str, ...]:
    cols = [VALID_COL, "task_index"]
    for prefix in (ACTION_PREFIX, STATE_PREFIX):
        for side in SIDES:
            cols += [POSE_COL.format(prefix=prefix, side=side), GRIP_COL.format(prefix=prefix, side=side)]
    return tuple(cols)


def is_compatible_bucket(path: Path) -> bool:
    """LeRobot v3.0 bucket with the torso-frame EEF schema this reader consumes."""
    info_path = path / "meta" / "info.json"
    if not info_path.is_file():
        return False
    info = json.loads(info_path.read_text())
    features = info.get("features", {})
    return info.get("codebase_version") == "v3.0" and all(c in features for c in needed_columns())


def split_episodes(eps: pd.DataFrame, split: str, val_every: int) -> pd.DataFrame:
    is_val = (eps["episode_index"].to_numpy() % int(val_every)) == 0
    if split == "val":
        keep = is_val
    elif split == "train":
        keep = ~is_val
    else:
        raise ValueError(f"G1Dex1 split must be 'train' or 'val', got {split!r}")
    return eps[keep].reset_index(drop=True)


class G1Dex1Dataset(LeRobotV3Reader):
    """Single-task bucket reader; ``from_config`` on a root builds all tasks."""

    DATASET_NAME = "G1Dex1"
    ACTION_DIM = EEF20_DIM
    NEEDED_COLS = needed_columns()
    DEPLOY_ACTION_MODE = ACTION_MODE
    HEAD_CAMERA = "observation.images.head_stereo_left"
    LEFT_WRIST_CAMERA = "observation.images.wrist_left"
    RIGHT_WRIST_CAMERA = "observation.images.wrist_right"
    CONFIG_KEYS: ClassVar[Tuple[str, ...]] = LeRobotV3Reader.CONFIG_KEYS + (
        "action_mode",
        "normalization_stats_path",
        "val_every",
    )

    def __init__(
        self,
        dataset_dir: str,
        *,
        action_mode: str = ACTION_MODE,
        normalization_stats_path: Optional[str] = None,
        val_every: int = 50,
        **kwargs: Any,
    ):
        if action_mode != ACTION_MODE:
            raise ValueError(f"G1Dex1 supports only action_mode='eef', got {action_mode!r}")
        self.action_mode = action_mode
        self._val_every = int(val_every)
        self._stats_path = normalization_stats_path
        super().__init__(dataset_dir=dataset_dir, **kwargs)

    def _build_episode_index(self, info: dict) -> pd.DataFrame:
        eps = load_episodes_parquet(self._dataset_dir)
        self._add_episode_offsets(eps)
        return split_episodes(eps, self._split, self._val_every)

    def _post_init(self, info: dict) -> None:
        if not is_compatible_bucket(self._dataset_dir):
            raise ValueError(f"G1Dex1({self._dataset_id}): not a v3.0 torso-EEF bucket")
        if float(info["fps"]) != 30.0:
            raise ValueError(f"G1Dex1({self._dataset_id}): expected 30 fps, got {info['fps']}")

    def _load_stats(self, info: dict) -> Optional[dict]:
        if not self._normalize_mode or self._normalize_mode in ("none", "null"):
            return None
        if not self._stats_path:
            raise ValueError(
                "G1Dex1 needs normalization_stats_path; build it with "
                "python -m openwam.dataloader.utils.stats_computation.g1_dex1_stats_computation"
            )
        meta = load_stats_metadata(self._stats_path, action_mode=ACTION_MODE)
        bad = {k: (meta.get(k), v) for k, v in STATS_CONTRACT.items() if meta.get(k) != v}
        if bad:
            raise ValueError(f"G1Dex1 stats contract mismatch in {self._stats_path}: {bad}")
        self.normalization_stats_path = str(self._stats_path)
        return load_stats_file(
            self._stats_path, action_mode=ACTION_MODE, normalize_mode=self._normalize_mode, dim=EEF20_DIM
        )

    def _resolve_prompt(self, row, win: pd.DataFrame) -> str:
        # tasks.parquet stores "@"-joined paraphrases: sample one for train, first for val.
        options = [s.strip() for s in super()._resolve_prompt(row, win).split("@") if s.strip()]
        return random.choice(options) if self._split == "train" else options[0]

    def _check_valid(self, win: pd.DataFrame) -> None:
        # ponytail: invalid-EE windows raise so _safe_get moves to the next index;
        # fine while such rows are rare, switch to per-step masking if they are not.
        if (_column(win, VALID_COL, 1) == 0).any():
            raise ValueError("G1Dex1 window contains recomputed_ee_valid == 0 rows")

    def _action_20d(self, win: pd.DataFrame) -> np.ndarray:
        self._check_valid(win)
        return apply_normalization(window_to_eef20(win, ACTION_PREFIX), self._normalization_stats, self._normalize_mode)

    def _proprio_20d(self, win: pd.DataFrame) -> np.ndarray:
        raw = window_to_eef20(win.iloc[:1], STATE_PREFIX)
        return apply_normalization(raw, self._normalization_stats, self._normalize_mode)

    @classmethod
    def _multibucket_wrapper(cls):
        return MultiG1Dex1Dataset

    @classmethod
    def from_config(cls, config, split: str = "train"):
        from openwam.dataloader.utils import get_cfg

        root = Path(get_cfg(config, "dataset_dir"))
        if (root / "meta" / "info.json").is_file():
            return super().from_config(config, split=split)
        sub_dirs = discover_buckets(root, get_cfg(config, "tasks"), get_cfg(config, "exclude_tasks"))
        # Same key-presence rule as LeRobotV3Reader.from_config: an explicit
        # normalize_mode=null disables normalization, an absent key keeps the default.
        common: Dict[str, Any] = {"split": split}
        for key in cls.CONFIG_KEYS:
            value = get_cfg(config, key, _CONFIG_MISSING)
            if value is _CONFIG_MISSING or (value is None and key != "normalize_mode"):
                continue
            common[key] = value
        dataset = build_multibucket(
            cls,
            sub_dirs,
            common,
            base_seed=int(get_cfg(config, "seed", 42)),
            total_hours=get_cfg(config, "total_hours"),
            wrapper_cls=MultiG1Dex1Dataset,
            source_name=cls.__name__,
        )
        loaded = {b._dataset_id for b in dataset.buckets}  # noqa: SLF001
        missing = [d.name for d in sub_dirs if d.name not in loaded]
        if missing:
            raise RuntimeError(f"G1Dex1: {len(missing)} bucket(s) failed to load or were empty: {missing}")
        return dataset


def discover_buckets(
    root: Path, tasks: Optional[Sequence[str]] = None, exclude_tasks: Optional[Sequence[str]] = None
) -> List[Path]:
    """Compatible task buckets under ``root``; ``tasks`` (if given) must all be compatible."""
    if tasks:
        dirs = [root / str(t) for t in tasks]
        bad = [d.name for d in dirs if not is_compatible_bucket(d)]
        if bad:
            raise ValueError(f"G1Dex1: requested tasks are missing or not v3.0 torso-EEF buckets: {bad}")
    else:
        dirs = sorted(d for d in root.iterdir() if d.is_dir() and is_compatible_bucket(d))
    excluded = set(exclude_tasks or ())
    dirs = [d for d in dirs if d.name not in excluded]
    if not dirs:
        raise FileNotFoundError(f"G1Dex1: no compatible buckets under {root}")
    logger.info("G1Dex1: %d task buckets under %s", len(dirs), root)
    return dirs


class MultiG1Dex1Dataset(MultiLeRobotV3Reader):
    """All G1-Dex1 task buckets; every bucket shares one pooled stats file."""

    @property
    def normalization_stats_path(self) -> Optional[str]:
        return self._buckets[0].normalization_stats_path


class G1DSelfDataset(G1Dex1Dataset):
    """Self-collected G1-Dex1 buckets after ``utils/g1d_self_convert.py``.

    Same torso-EEF schema and stats contract as the official data; only the
    camera keys differ, and videos are shared in place with the source (trimmed /
    filtered episodes), so frame offsets come from ``from_timestamp`` instead of
    the cumulative episode lengths.
    """

    DATASET_NAME = "G1DSelf"
    HEAD_CAMERA = "observation.images.cam_left_high"
    LEFT_WRIST_CAMERA = "observation.images.cam_left_wrist"
    RIGHT_WRIST_CAMERA = "observation.images.cam_right_wrist"

    def _add_episode_offsets(self, eps: pd.DataFrame) -> None:
        super()._add_episode_offsets(eps)
        for cam in self._video_cameras():
            col = f"videos/{cam}/from_timestamp"
            if col in eps.columns:
                eps[self._video_offset_col(cam)] = np.round(eps[col].to_numpy() * self._fps).astype(np.int64)

    @classmethod
    def _multibucket_wrapper(cls):
        return MultiG1Dex1Dataset


ROT6D_DIMS = ROT6D_DIMS_EEF20

__all__ = [
    "ACTION_MODE",
    "EEF20_DIM",
    "GRIPPER_DIMS_EEF20",
    "G1DSelfDataset",
    "G1Dex1Dataset",
    "MultiG1Dex1Dataset",
    "ROT6D_DIMS",
    "STATS_CONTRACT",
    "discover_buckets",
    "gripper_to_open_scale",
    "is_compatible_bucket",
    "split_episodes",
    "window_to_eef20",
]
