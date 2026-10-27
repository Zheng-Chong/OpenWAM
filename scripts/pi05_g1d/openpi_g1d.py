"""pi0.5 training config for the RoboDojo G1-D sim pour-beans data, registered into the vendored openpi of XPolicyLab
(`policy/Pi_05/openpi`) without editing it. train.sh uses the PyTorch trainer: the JAX pi05_base on GCS downloads at
~70 KB/s from DSW, while openpi's PyTorch conversion of it is on OSS.

Data: LeRobot v3 from prepare_data.py, 16 dims = [left arm 7, left gripper, right arm 7, right gripper], grippers
0 closed .. 1 open, cameras cam_high / cam_left_wrist / cam_right_wrist, 30 fps. Arm joints are trained as deltas to
the current state, grippers absolute (openpi's ALOHA recipe).

Run an openpi script with this config registered (inside the openpi uv env):
    python openpi_g1d.py scripts/compute_norm_stats.py --config-name pi05_g1d_sz_pour_beans
    torchrun --nproc_per_node=N openpi_g1d.py scripts/train_pytorch.py pi05_g1d_sz_pour_beans --exp-name <run> \
        --pytorch-weight-path <dir with model.safetensors> [--batch-size B ...]
"""

import dataclasses
import pathlib
import runpy
import sys

import einops
import numpy as np
import openpi.models.model as _model
import openpi.models.pi0_config as pi0_config
import openpi.training.config as _config
import openpi.training.optimizer as _optimizer
import openpi.training.weight_loaders as weight_loaders
import openpi.transforms as _transforms

NAME = "pi05_g1d_sz_pour_beans"
REPO_ID = "g1d_sz_pour_beans_cup_or_mug_500"
CAMERAS = {"cam_high": "base_0_rgb", "cam_left_wrist": "left_wrist_0_rgb", "cam_right_wrist": "right_wrist_0_rgb"}
ACTION_DIM = 16
DELTA_MASK = _transforms.make_bool_mask(7, -1, 7, -1)
NUM_TRAIN_STEPS = 10_000


def _hwc_uint8(img) -> np.ndarray:
    img = np.asarray(img)
    if np.issubdtype(img.dtype, np.floating):  # LeRobot frames are float CHW in [0, 1]; the server sends HWC uint8
        img = (255 * img).astype(np.uint8)
    if img.shape[0] == 3:
        img = einops.rearrange(img, "c h w -> h w c")
    return img


@dataclasses.dataclass(frozen=True)
class G1DInputs(_transforms.DataTransformFn):
    def __call__(self, data: dict) -> dict:
        out = {
            "state": np.asarray(data["state"], np.float32),
            "image": {dst: _hwc_uint8(data["images"][src]) for src, dst in CAMERAS.items()},
            "image_mask": {dst: np.True_ for dst in CAMERAS.values()},
        }
        if "actions" in data:
            out["actions"] = np.asarray(data["actions"], np.float32)
        if "prompt" in data:
            out["prompt"] = data["prompt"]
        return out


@dataclasses.dataclass(frozen=True)
class G1DOutputs(_transforms.DataTransformFn):
    def __call__(self, data: dict) -> dict:
        return {"actions": np.asarray(data["actions"][:, :ACTION_DIM])}


@dataclasses.dataclass(frozen=True)
class LeRobotG1DDataConfig(_config.DataConfigFactory):
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> _config.DataConfig:
        repack = _transforms.Group(
            inputs=[
                _transforms.RepackTransform(
                    {
                        "images": {cam: f"observation.images.{cam}" for cam in CAMERAS},
                        "state": "observation.state",
                        "actions": "action",
                        "prompt": "prompt",
                    }
                )
            ]
        )
        data_transforms = _transforms.Group(inputs=[G1DInputs()], outputs=[G1DOutputs()]).push(
            inputs=[_transforms.DeltaActions(DELTA_MASK)], outputs=[_transforms.AbsoluteActions(DELTA_MASK)]
        )
        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            repack_transforms=repack,
            data_transforms=data_transforms,
            model_transforms=_config.ModelTransformFactory()(model_config),
            action_sequence_keys=("action",),
        )


CONFIG = _config.TrainConfig(
    name=NAME,
    model=pi0_config.Pi0Config(pi05=True),
    data=LeRobotG1DDataConfig(repo_id=REPO_ID, base_config=_config.DataConfig(prompt_from_task=True)),
    weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
    lr_schedule=_optimizer.CosineDecaySchedule(
        warmup_steps=500, peak_lr=2.5e-5, decay_steps=NUM_TRAIN_STEPS, decay_lr=2.5e-6
    ),
    batch_size=112,  # global; must divide by the number of visible GPUs
    num_workers=16,
    num_train_steps=NUM_TRAIN_STEPS,
    save_interval=2000,
    wandb_enabled=False,
)
_config._CONFIGS_DICT[NAME] = CONFIG


if __name__ == "__main__":
    import openpi_g1d  # noqa: F401  re-register from the importable module: spawned data workers can't pickle __main__ classes

    script = sys.argv[1]
    sys.argv = sys.argv[1:]
    runpy.run_path(script, run_name="__main__")
