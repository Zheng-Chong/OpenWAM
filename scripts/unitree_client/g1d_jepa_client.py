#!/usr/bin/env python3
"""Robot-side client for the JEPA-WAM G1-Dex1 policy server.

This is ``motus_inference_client.py`` with the MotusV2-specific parts replaced.
Kept verbatim in spirit: the ``unitree_lerobot`` plumbing, the internal Dex1
grippers on ``rt/lowstate`` motors 31/33, the initial-pose ramp, the 30Hz loop,
and the async double-buffer with its staleness trim. Removed, because they are
properties of MotusV2 rather than of this robot:

* **The stitched image.** That model takes one T-shaped canvas; this one takes
  three separate cameras. The canvas would leave the head at roughly 240x640 and
  each wrist at 240x320, against the 480x640 each camera had when these features
  were encoded, so stitching then un-stitching throws away over half the
  resolution for nothing.
* **``_REORDER_FROM_RAW`` / ``_to_arm_interleaved``.** MotusV2 wants
  ``[L7, LG, R7, RG]``. This policy uses ``JOINT16`` -- ``[L7, R7, LG, RG]``,
  both grippers last -- which is *already* the layout the client builds in
  ``raw_state``, so both reorder hops are simply deleted.
* **``SWAP_WRISTS``.** It corrects a key swap for MotusV2, but these features
  were encoded straight off the same swapped dataset keys, so applying it again
  would invert the wrists twice.
* **The YAML config, ``action_interp_factor``, ``PAD_JOINT_VALUES``,
  ``GRIPPER_MAP``.** Chunk geometry now comes from the handshake; the actions are
  already at the robot's 30Hz so there is nothing to interpolate; this checkpoint
  trains all 16 dims so nothing is frozen; and the gripper range comes from the
  checkpoint's own min/max rather than a hand-tuned rescale.
* **``--prompt``.** The instruction embedding is baked into the checkpoint.

Safety numbers come from the server handshake instead of the command line, so
they cannot drift from the checkpoint: ``joint_limits`` is the range the
fine-tune data actually covered and ``max_step_rad`` is the per-step cap. The
rate limiter has to run here, against the pose just measured -- the server cannot
do it.

    python g1d_jepa_client.py \
      --image_host 192.168.123.164 --ee dex1_internal --async_inference
    # prompts to pick a model from MODEL_REGISTRY; or skip the prompt:
    #   --server_host $G1D_GATEWAY_HOST --server_port 10070

**Async needs the server started with ``--no-action-ensemble``.** With ensembling
on, the server is stateful across requests and assumes the client executes
exactly ``action_horizon`` actions; an async loop trims the steps that went stale
while inference ran, which silently averages misaligned chunks. The client checks
the handshake and refuses rather than degrading quietly.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
# The client may live anywhere; unitree_lerobot is a plain checkout, not installed.
LEROBOT_ROOT = os.environ.get("LEROBOT_ROOT", "/home/unitree/unitree_lerobot")
sys.path.append(LEROBOT_ROOT)
from motus_client_adapter import JepaWamClient  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s [g1d-jepa] %(message)s")
logger = logging.getLogger(__name__)

#: Internal Dex1 gripper motors on rt/lowstate. The USB Dex1 hangs on this G1,
#: so the robot-side stack drives these directly. Values copied from the
#: MotusV2 client, which is what the fine-tune data was collected through.
INTERNAL_LEFT_GRIPPER = 31
INTERNAL_RIGHT_GRIPPER = 33
INTERNAL_GRIPPER_KP = 5.0
INTERNAL_GRIPPER_KD = 0.05
INTERNAL_GRIPPER_DELTA = 0.18
#: Grip range is 0 (closed) .. 5.4 (open). The training labels are *measured*
#: positions, so on an object the policy asks for exactly where the fingers
#: stalled and position control then squeezes with ~zero force. Below
#: GRIP_CLOSED_BELOW we aim GRIP_SQUEEZE further closed. Both are CLI knobs.
GRIP_CLOSED_BELOW = 4.5  # grasps stall at 2.4-3.3 in the data; fully open is 5.36
GRIP_SQUEEZE = 0.3

CAMERA_SOURCE = {
    "head_left": "observation.images.cam_left_high",
    "head_right": "observation.images.cam_right_high",
    "wrist_left": "observation.images.cam_left_wrist",
    "wrist_right": "observation.images.cam_right_wrist",
}

#: Known deployed servers, for the interactive picker when ``--server_host`` is
#: omitted. Update this when a server moves or a new fine-tune goes live.
#: dsw-4 ports 8001-8004, published on the relay as 10070-10073. Servers
#: return whole chunks; --exec_steps / --ensemble pick how they are run.
GATEWAY_HOST = os.environ.get("G1D_GATEWAY_HOST", "127.0.0.1")  # public relay to the GPU box; set on the robot
MODEL_REGISTRY = [
    ("OpenWAM G1D 后训练 step20000 - 倒豆子-Plus", GATEWAY_HOST, 10080),
    ("OpenWAM G1D 后训练 step20000 - 卡皮巴拉放箱子", GATEWAY_HOST, 10076),
    ("OpenWAM G1D 后训练 step20000 - 三物体: bottle", GATEWAY_HOST, 10077),
    ("OpenWAM G1D 后训练 step20000 - 三物体: marker", GATEWAY_HOST, 10078),
    ("OpenWAM G1D 后训练 step20000 - 三物体: capybara plush", GATEWAY_HOST, 10079),
    ("ViT-B single ΨG1D chunk120 20k - 三物体: capybara plush", GATEWAY_HOST, 10070),
    ("ACT (wuqingman, 3cam, 60k) - 卡皮巴拉放篮子", GATEWAY_HOST, 10071),
    ("ViT-B single ΨG1D chunk120 20k - 三物体: bottle", GATEWAY_HOST, 10072),
    ("ViT-B single ΨG1D chunk120 20k - 三物体: marker", GATEWAY_HOST, 10073),
]


#: Instructions the OpenWAM post-train saw (first paraphrase of each task). Only offered
#: for OpenWAM servers, which are language-conditioned; older servers bake the task in.
PROMPTS = [
    "Put the capybara plush into the box.",
    "Pick up the bottle and put it into the basket.",
    "Pick up the marker and put it into the basket.",
    "Pick up the capybara plush and put it into the basket.",
    "Pour the beans.",
    "Pick up the cup and kettle and pour water.",
    "Pour the red beans from the square cup into the blue cup, halfway.",
    "Pour the red beans from the square cup into the gray cup, halfway.",
    "Flip the blue cup upright, then pour red beans halfway.",
    "Stand the fallen blue cup up, then pour red beans halfway.",
    "Pour half into the purple cup, then the green cup. Leave other items.",
]


def choose_prompt(server_instruction: str) -> str:
    """Let the operator pick the task instruction sent to a language-conditioned server."""
    print("可选指令 (prompt)：")
    print(f"  [0] 服务端默认: {server_instruction}")
    for i, text in enumerate(PROMPTS, 1):
        print(f"  [{i}] {text}")
    print("  [c] 自定义输入")
    while True:
        choice = input(f"选择指令 [0-{len(PROMPTS)}/c] (回车=0): ").strip().lower()
        if choice in ("", "0"):
            return ""
        if choice == "c":
            text = input("输入指令: ").strip()
            if text:
                return text
        elif choice.isdigit() and 1 <= int(choice) <= len(PROMPTS):
            return PROMPTS[int(choice) - 1]
        print("无效输入，请重新输入。")


def choose_server(registry=MODEL_REGISTRY):
    """Probe each known server's handshake and let the operator pick one."""
    print("可用模型：")
    rows = []
    for i, (label, host, port) in enumerate(registry, 1):
        try:
            probe = JepaWamClient(host, port, timeout_ms=5000)
            instruction = probe.metadata.get("instruction", "?")
            probe.close()
            status = f"instruction={instruction!r}"
        except Exception as exc:  # noqa: BLE001 - report and let the operator retry
            status = f"不可达: {exc}"
        rows.append((label, host, port))
        print(f"  [{i}] {label}  ({host}:{port})  {status}")
    while True:
        choice = input(f"选择模型 [1-{len(rows)}]: ").strip()
        if choice.isdigit() and 1 <= int(choice) <= len(rows):
            _, host, port = rows[int(choice) - 1]
            return host, port
        print("无效输入，请重新输入。")


#: Settled 2026-09-25 on the robot, both ends checked:
#: * dataset (capybara_to_box, pick_3objects_to_basket): cam_left_wrist motion
#:   tracks the LEFT arm (corr +0.53 / +0.28 vs -0.15 / -0.13 for the right);
#: * live image server: covering the physical LEFT wrist camera darkens
#:   cam_right_wrist (scripts/eval/g1_dex1 wrist_check on the robot).
#: The image server's wrist ports are exchanged relative to collection, so the
#: swap is required. Re-run the cover test if the camera wiring is touched.
SWAP_WRISTS_DEFAULT = True



@dataclass
class ClientConfig:
    """Config handed to ``unitree_lerobot``'s ``setup_*`` plus our own knobs.

    The first block is kept **field-for-field identical to the MotusV2 client's
    ClientConfig**, including the entries this client never reads. Those
    functions take the whole dataclass and pick attributes out of it, so a field
    that looks unused here can still be required there -- ``motion`` is exactly
    that, and dropping it produced an AttributeError on the robot. Guessing
    which subset is needed costs a round trip to the robot each time it is
    wrong, so the whole set stays.
    """

    # --- read by unitree_lerobot; keep in step with the MotusV2 client -----
    arm: str = "G1_29"
    ee: str = "dex1_internal"
    frequency: float = 30.0
    image_host: str = "192.168.123.164"
    instruction: str = ""
    motion: bool = False
    sim: bool = False
    base_type: str = "legs"
    root: str = ""

    # --- this client's own ------------------------------------------------
    server_host: str = "127.0.0.1"
    server_port: int = 8000
    jpeg_quality: int = 95
    init_pose_json: str = ""
    init_pose_frame: int = 0
    init_pose_steps: int = 100
    async_inference: bool = False
    smooth_alpha: float = 1.0
    record_log: str = ""
    web_port: int = 8088
    anchor_cap: float = 0.1
    exec_steps: int = 0
    video_dir: str = ""
    ensemble: bool = False
    swap_wrists: bool = SWAP_WRISTS_DEFAULT
    prompt: str | None = None  # None = ask the operator (OpenWAM servers); '' = server default


# ------------------------------------------------------------ observation

def _to_uint8_hwc(img):
    if hasattr(img, "cpu"):
        img = img.cpu().numpy()
    if img is None:
        return None
    if img.dtype != np.uint8:
        img = (img * 255).astype(np.uint8) if img.max() <= 1.0 else img.astype(np.uint8)
    return img


def camera_sources(swap_wrists: bool) -> dict:
    """Policy camera key -> the observation key to read it from."""
    sources = dict(CAMERA_SOURCE)
    if swap_wrists:
        sources["wrist_left"] = CAMERA_SOURCE["wrist_right"]
        sources["wrist_right"] = CAMERA_SOURCE["wrist_left"]
    return sources


def build_request(observation, current_arm_q, left_grip, right_grip,
                  swap_wrists: bool = SWAP_WRISTS_DEFAULT):
    """Native observation -> up to four camera frames plus a joint16 state.

    No stitching: the policy takes cameras at their own resolution. The frames
    are already RGB -- ``process_images_and_observations`` converts them -- so
    nothing here touches the channels. A camera missing from ``observation``
    (e.g. ``head_right``, which only the 4-camera ACT policies need) is simply
    left out here; ``JepaWamClient._encode_images`` is what actually enforces
    "does the selected server's handshake need this camera", with a clearer
    error naming the missing one.
    """
    images = {}
    for camera, source in camera_sources(swap_wrists).items():
        frame = _to_uint8_hwc(observation.get(source))
        if frame is not None:
            images[camera] = frame

    # JOINT16: left arm 7, right arm 7, then both grippers. This is what the
    # MotusV2 client called raw_state, before _to_arm_interleaved.
    state = np.zeros(16, dtype=np.float32)
    state[0:7] = current_arm_q[0:7]
    state[7:14] = current_arm_q[7:14]
    state[14] = left_grip
    state[15] = right_grip
    return images, state


def split_action(action16):
    """One joint16 action -> ``(arm14, left_grip, right_grip)``, no reordering."""
    action16 = np.asarray(action16, dtype=np.float32).reshape(-1)
    if action16.shape != (16,):
        raise ValueError(f"expected 16-D action, got {action16.shape}")
    return action16[0:14].copy(), float(action16[14]), float(action16[15])


# ---------------------------------------------------------------- safety

class SafetyLimiter:
    """Range clamp plus per-step rate limit, seeded from the measured pose.

    Both bounds come from the server handshake, so they track the checkpoint
    rather than a flag someone has to remember to update. This must run here:
    the rate limit is relative to the pose the robot is in right now, which only
    the robot side knows.
    """

    def __init__(self, lower, upper, max_step: float, smooth_alpha: float = 1.0):
        self.lower = np.asarray(lower, dtype=np.float64)
        self.upper = np.asarray(upper, dtype=np.float64)
        self.max_step = float(max_step)
        self.smooth_alpha = float(np.clip(smooth_alpha, 0.0, 1.0))
        self._last = None
        self.n_clamped = 0
        self.n_rate_limited = 0

    @property
    def is_seeded(self) -> bool:
        return self._last is not None

    def reset(self, state16) -> None:
        self._last = np.clip(np.asarray(state16, dtype=np.float64),
                             self.lower, self.upper)

    def filter(self, action16):
        if self._last is None:
            raise RuntimeError("call reset(state) first")
        target = np.asarray(action16, dtype=np.float64)
        if not np.isfinite(target).all():
            raise ValueError("action is not finite")
        clamped = np.clip(target, self.lower, self.upper)
        if np.any(clamped != target):
            self.n_clamped += 1
        if self.smooth_alpha < 1.0:
            clamped = self.smooth_alpha * clamped + (1.0 - self.smooth_alpha) * self._last
        stepped = np.clip(clamped, self._last - self.max_step, self._last + self.max_step)
        if np.any(stepped != clamped):
            self.n_rate_limited += 1
        self._last = stepped
        return stepped.astype(np.float32)

    def hold(self):
        if self._last is None:
            raise RuntimeError("call reset(state) first")
        return self._last.astype(np.float32)


# ------------------------------------------------------------- grippers

def _is_internal_ee(ee: str) -> bool:
    return str(ee or "").lower() in ("dex1_internal", "internal")


def _read_internal_grippers(arm_ctrl):
    q = arm_ctrl.get_current_motor_q()
    return float(q[INTERNAL_LEFT_GRIPPER]), float(q[INTERNAL_RIGHT_GRIPPER])


def _enable_internal_grippers(arm_ctrl):
    from multiprocessing import Value

    left, right = _read_internal_grippers(arm_ctrl)
    for idx, q0 in ((INTERNAL_LEFT_GRIPPER, left), (INTERNAL_RIGHT_GRIPPER, right)):
        cmd = arm_ctrl.msg.motor_cmd[idx]
        cmd.mode, cmd.kp, cmd.kd = 1, INTERNAL_GRIPPER_KP, INTERNAL_GRIPPER_KD
        cmd.dq, cmd.tau, cmd.q = 0.0, 0.0, q0
    logger.info("Internal Dex1 enabled: left=%.3f right=%.3f (motors 31/33)", left, right)
    return {"mode": "internal", "arm_ctrl": arm_ctrl,
            "left": Value("d", left, lock=True), "right": Value("d", right, lock=True)}


def _read_grippers(ee_shared_mem):
    if not ee_shared_mem:
        return 0.0, 0.0
    if ee_shared_mem.get("mode") == "internal":
        return _read_internal_grippers(ee_shared_mem["arm_ctrl"])
    left, right = ee_shared_mem.get("left"), ee_shared_mem.get("right")
    if hasattr(left, "value"):
        return float(left.value), float(right.value)
    return 0.0, 0.0


def _write_gripper(ee_shared_mem, left_grip, right_grip):
    if not ee_shared_mem:
        return
    if ee_shared_mem.get("mode") == "internal":
        arm_ctrl = ee_shared_mem["arm_ctrl"]
        cur_l, cur_r = _read_internal_grippers(arm_ctrl)
        close_step = max(INTERNAL_GRIPPER_DELTA, GRIP_SQUEEZE)

        def squeeze(g, cur):
            if g < GRIP_CLOSED_BELOW:
                g -= GRIP_SQUEEZE
            return float(np.clip(g, cur - close_step, cur + INTERNAL_GRIPPER_DELTA))

        left_grip, right_grip = squeeze(left_grip, cur_l), squeeze(right_grip, cur_r)
        ee_shared_mem["left"].value = left_grip
        ee_shared_mem["right"].value = right_grip
        for idx, q in ((INTERNAL_LEFT_GRIPPER, left_grip), (INTERNAL_RIGHT_GRIPPER, right_grip)):
            cmd = arm_ctrl.msg.motor_cmd[idx]
            cmd.mode, cmd.kp, cmd.kd, cmd.q = 1, INTERNAL_GRIPPER_KP, INTERNAL_GRIPPER_KD, q
        return
    from multiprocessing.sharedctypes import SynchronizedArray

    from unitree_lerobot.eval_robot.utils.utils import to_list

    if isinstance(ee_shared_mem.get("left"), SynchronizedArray):
        ee_shared_mem["left"][:] = to_list(np.array([left_grip]))
        ee_shared_mem["right"][:] = to_list(np.array([right_grip]))
    elif hasattr(ee_shared_mem.get("left"), "value"):
        ee_shared_mem["left"].value = float(left_grip)
        ee_shared_mem["right"].value = float(right_grip)


# ----------------------------------------------------------- initial pose

def init_pose_from_data_json(path, frame=0):
    with open(path, "r", encoding="utf-8") as fh:
        entry = json.load(fh)["data"][frame]["states"]
    arm_q = np.concatenate([
        np.asarray(entry["left_arm"]["qpos"], dtype=np.float32),
        np.asarray(entry["right_arm"]["qpos"], dtype=np.float32),
    ])
    return arm_q, float(entry["left_ee"]["qpos"][0]), float(entry["right_ee"]["qpos"][0])


def capture_ready_pose(cfg: ClientConfig, path: str) -> None:
    """Hand-pose the arms, then save the current pose as an ``--init_pose_json``.

    No image client, no policy server, and no ``ctrl_dual_arm`` call -- this
    only reads, so the arms stay exactly where the operator puts them (backdrive
    them by hand into the elbow-up ready pose, then hit Enter here).
    """
    from dataclasses import replace

    from unitree_lerobot.eval_robot.make_robot import setup_robot_interface

    try:
        robot = setup_robot_interface(
            replace(cfg, ee="") if _is_internal_ee(cfg.ee) else cfg)
    except AttributeError as exc:
        raise _missing_config_field(exc) from exc
    ee_shared_mem = robot["ee_shared_mem"]
    if _is_internal_ee(cfg.ee):
        ee_shared_mem = _enable_internal_grippers(robot["arm_ctrl"])

    input("手动把双臂摆到目标姿态，回车记录当前姿态: ")
    arm_q = robot["arm_ctrl"].get_current_dual_arm_q()
    left_grip, right_grip = _read_grippers(ee_shared_mem)
    payload = {"data": [{"states": {
        "left_arm": {"qpos": np.asarray(arm_q[0:7], dtype=np.float32).tolist()},
        "right_arm": {"qpos": np.asarray(arm_q[7:14], dtype=np.float32).tolist()},
        "left_ee": {"qpos": [left_grip]},
        "right_ee": {"qpos": [right_grip]},
    }}]}
    Path(path).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    logger.info("Saved ready pose to %s -- pass it back as --init_pose_json", path)


def move_to_arm_qpos(arm_ctrl, arm_ik, ee_shared_mem, arm_q, left_grip, right_grip,
                     steps=100, dt=0.02):
    """Ramp from wherever the arms are to arm_q so the robot does not snap."""
    current = arm_ctrl.get_current_dual_arm_q()
    logger.info("Moving to initial pose over %.1fs", steps * dt)
    for j in range(steps):
        alpha = (j + 1) / steps
        interp = current * (1 - alpha) + arm_q * alpha
        arm_ctrl.ctrl_dual_arm(interp, arm_ik.solve_tau(interp))
        _write_gripper(ee_shared_mem, left_grip, right_grip)
        time.sleep(dt)
    arm_ctrl.ctrl_dual_arm(arm_q, arm_ik.solve_tau(arm_q))
    _write_gripper(ee_shared_mem, left_grip, right_grip)
    time.sleep(0.5)
    reached = arm_ctrl.get_current_dual_arm_q()
    logger.info("Reached initial pose (max err %.4f rad)",
                float(np.abs(reached - arm_q).max()))


# ------------------------------------------------------------------ loops

def _missing_config_field(exc: AttributeError) -> SystemExit:
    """Turn a missing ClientConfig field into something self-serviceable.

    ``setup_image_client`` / ``setup_robot_interface`` take the whole dataclass
    and pick attributes out of it, and the set they read is not documented
    anywhere we can see. Rather than another round trip to the robot, say
    exactly what to add.
    """
    return SystemExit(
        f"{exc}\n\n"
        f"unitree_lerobot read a ClientConfig field this client does not define. "
        f"Add it to ClientConfig with the same default the MotusV2 client used "
        f"and re-run -- it only has to exist, this client does not use it."
    )


def _setup(cfg: ClientConfig):
    from unitree_lerobot.eval_robot.make_robot import (
        setup_image_client, setup_robot_interface,
    )

    try:
        image_info = setup_image_client(cfg)
    except AttributeError as exc:
        raise _missing_config_field(exc) from exc
    if isinstance(image_info, tuple):
        image_info = {"image_client": image_info[0], "camera_config": image_info[1]}

    from dataclasses import replace

    # setup_robot_interface would try to open the USB Dex1, which hangs here.
    try:
        robot = setup_robot_interface(
            replace(cfg, ee="") if _is_internal_ee(cfg.ee) else cfg)
    except AttributeError as exc:
        raise _missing_config_field(exc) from exc
    ee_shared_mem = robot["ee_shared_mem"]
    if _is_internal_ee(cfg.ee):
        ee_shared_mem = _enable_internal_grippers(robot["arm_ctrl"])

    client = JepaWamClient(cfg.server_host, cfg.server_port,
                           jpeg_quality=cfg.jpeg_quality)
    if str(client.metadata.get("policy", "")).startswith("OpenWAM"):
        client.prompt = cfg.prompt if cfg.prompt is not None else choose_prompt(client.metadata.get("instruction", ""))
        logger.info("prompt sent to server: %r", client.prompt or "(server default)")
    lower, upper = client.joint_limits
    logger.warning("wrist cameras: %s. Getting this backwards mirrors the "
                   "policy's view and nothing downstream will complain -- confirm "
                   "with the live check in OPERATOR.md.",
                   "SWAPPED (cam_right_wrist -> wrist_left)" if cfg.swap_wrists
                   else "pass-through (cam_left_wrist -> wrist_left)")
    logger.info("server: action_horizon=%d cameras=%s ensemble=%s instruction=%r",
                client.action_horizon, client.cameras,
                client.metadata.get("action_ensemble"),
                client.metadata.get("instruction"))
    if cfg.async_inference and client.metadata.get("action_ensemble", True):
        client.close()
        raise SystemExit(
            "the server has action ensembling on, which assumes every returned "
            "chunk is executed whole; this async loop trims the steps that went "
            "stale during inference, so the two would silently disagree. Restart "
            "the server with --no-action-ensemble, or drop --async_inference."
        )
    limiter = SafetyLimiter(lower, upper, client.max_step_rad, cfg.smooth_alpha)
    image_info["video"] = start_live_view(cfg, image_info, robot, ee_shared_mem, cfg.web_port)
    log = CommandLog(cfg.record_log) if cfg.record_log else None
    if log is not None:
        logger.info("logging commands to %s", log.path)
    return image_info, robot, ee_shared_mem, client, limiter, log


def _observe(cfg, image_info, robot, ee_shared_mem):
    from unitree_lerobot.eval_robot.make_robot import process_images_and_observations

    observation, current_arm_q = process_images_and_observations(
        image_info["image_client"], image_info["camera_config"], robot["arm_ctrl"])
    left_grip, right_grip = _read_grippers(ee_shared_mem)
    images, state = build_request(observation, current_arm_q, left_grip, right_grip,
                                  cfg.swap_wrists)
    return images, state, current_arm_q


VIDEO_FPS = 10


def start_live_view(cfg, image_info, robot, ee_shared_mem, port):
    """Serve the policy's camera inputs as MJPEG on http://<robot>:<port>/.

    Frames go through the same ``build_request`` as an inference request, so the
    page shows exactly what the model is given (keys, wrist swap, RGB order). They
    are RGB; converting to BGR for cv2 makes the page display true colours, so red
    and blue looking exchanged on the page means the model sees them exchanged.
    """
    import cv2
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

    latest = {}
    recorder = {"writer": None, "path": None}  # set by start/stop below

    def tile(frames):
        cells = []
        for cam in CAMERA_SOURCE:
            f = frames.get(cam)
            cell = np.zeros((240, 320, 3), np.uint8) if f is None else cv2.resize(f, (320, 240))
            cv2.putText(cell, cam, (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
            cells.append(cell)
        return np.vstack([np.hstack(cells[:2]), np.hstack(cells[2:])])

    def grab():
        while True:
            t0 = time.perf_counter()
            try:
                images, _, _ = _observe(cfg, image_info, robot, ee_shared_mem)
                bgr = {cam: cv2.cvtColor(np.asarray(rgb), cv2.COLOR_RGB2BGR)
                       for cam, rgb in images.items()}
                if port:
                    for cam, f in bgr.items():
                        ok, buf = cv2.imencode(".jpg", f, [cv2.IMWRITE_JPEG_QUALITY, 70])
                        if ok:
                            latest[cam] = buf.tobytes()
                writer = recorder["writer"]
                if writer is not None:
                    writer.write(tile(bgr))
            except Exception as exc:  # noqa: BLE001 - a viewer must never stop the robot loop
                logger.debug("live view grab failed: %s", exc)
            time.sleep(max(0.0, 1.0 / VIDEO_FPS - (time.perf_counter() - t0)))

    def start_recording(path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        recorder["path"] = str(path)
        recorder["writer"] = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"),
                                             VIDEO_FPS, (640, 480))
        logger.info("recording episode video to %s", path)

    def stop_recording():
        writer, recorder["writer"] = recorder["writer"], None
        if writer is not None:
            time.sleep(2.0 / VIDEO_FPS)  # let an in-flight write finish
            writer.release()
            logger.info("saved %s", recorder["path"])

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_GET(self):
            cam = self.path.strip("/")
            if cam in CAMERA_SOURCE:
                self.send_response(200)
                self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=f")
                self.end_headers()
                try:
                    while True:
                        jpg = latest.get(cam)
                        if jpg:
                            self.wfile.write(b"--f\r\nContent-Type: image/jpeg\r\n\r\n" + jpg + b"\r\n")
                        time.sleep(0.1)
                except (BrokenPipeError, ConnectionResetError):
                    return
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            tiles = "".join(f'<figure><img src="/{c}"><figcaption>{c}</figcaption></figure>'
                            for c in CAMERA_SOURCE)
            self.wfile.write(f"""<!doctype html><title>G1D cameras</title><style>
body{{margin:0;background:#111;color:#ddd;font:14px sans-serif;display:flex;flex-wrap:wrap}}
figure{{margin:4px;flex:1 1 45%}}img{{width:100%}}</style>{tiles}""".encode())

    threading.Thread(target=grab, daemon=True).start()
    if port:
        server = ThreadingHTTPServer(("0.0.0.0", port), Handler)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        logger.info("live camera view on http://%s:%d/", cfg.image_host, port)
    return start_recording, stop_recording


class CommandLog:
    """Per-control-step record of requested, commanded and measured joint16.

    The server logs what the policy asked for; the clamp and the rate limit run
    here, so only this side knows what the arm was actually told, and only this
    side sees what it then measured. Without all three a post-hoc question like
    "did it fail because the policy was wrong or because the limiter held it
    back" cannot be answered.
    """

    def __init__(self, path):
        self.path = Path(path).expanduser()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = self.path.open("w", encoding="utf-8")
        self.step = 0

    def write(self, requested, commanded, measured, source: str) -> None:
        import json

        row = {
            "step": self.step,
            "t": time.time(),
            "source": source,           # "chunk" or "hold"
            "requested": np.asarray(requested, dtype=np.float32).tolist(),
            "commanded": np.asarray(commanded, dtype=np.float32).tolist(),
        }
        if measured is not None:
            row["measured_arm"] = np.asarray(measured, dtype=np.float32).tolist()
        self._fh.write(json.dumps(row) + "\n")
        self.step += 1

    def close(self) -> None:
        self._fh.close()


def _execute(robot, ee_shared_mem, limiter, action16, log=None, source="chunk"):
    commanded = limiter.filter(action16)
    arm, left_grip, right_grip = split_action(commanded)
    robot["arm_ctrl"].ctrl_dual_arm(arm, robot["arm_ik"].solve_tau(arm))
    _write_gripper(ee_shared_mem, left_grip, right_grip)
    if log is not None:
        # Measured after the command goes out, so it is the pose the previous
        # command achieved -- which is the lag the rate limit has to live with.
        measured = robot["arm_ctrl"].get_current_dual_arm_q()
        log.write(action16, commanded, measured, source)


def _hold(robot, ee_shared_mem, limiter, log=None):
    """Re-issue the last command. Never replays a consumed action, which would
    walk the arm backwards through a trajectory it has already run."""
    _execute(robot, ee_shared_mem, limiter, limiter.hold(), log, source="hold")


def _start_pose(cfg, robot, ee_shared_mem):
    arm_ctrl, arm_ik = robot["arm_ctrl"], robot["arm_ik"]
    if cfg.init_pose_json:
        arm_q, lg, rg = init_pose_from_data_json(cfg.init_pose_json, cfg.init_pose_frame)
        move_to_arm_qpos(arm_ctrl, arm_ik, ee_shared_mem, arm_q, lg, rg,
                         steps=cfg.init_pose_steps)
    else:
        current = arm_ctrl.get_current_dual_arm_q()
        arm_ctrl.ctrl_dual_arm(current, arm_ik.solve_tau(current))
        time.sleep(1.0)
        logger.info("Holding the current pose")


def anchor_chunk(actions, last_cmd, measured, cap):
    """Shift a new chunk's arm joints by (last command - measured pose).

    The arm trails its command (gravity, PD tracking error) and the policy
    continues from the *measured* pose, so every new chunk would start by pulling
    the target back onto the lagging arm -- the retreat seen at chunk boundaries.
    Re-anchoring on the last command removes it. Grippers are left alone. The
    offset is capped so contact (large error) is not answered by pushing harder.
    """
    offset = np.zeros(actions.shape[-1], dtype=np.float32)
    offset[:14] = np.clip(np.asarray(last_cmd)[:14] - np.asarray(measured)[:14], -cap, cap)
    return actions + offset


class ChunkEnsembler:
    """Client-side temporal ensembling: average every chunk's prediction for the
    same absolute step. Lets the operator pick exec_steps / ensembling per run
    while the server just returns whole chunks."""

    def __init__(self):
        self.cache: dict[int, list] = {}

    def next_actions(self, chunk, start: int, n: int):
        for i, a in enumerate(chunk):
            self.cache.setdefault(start + i, []).append(np.asarray(a, dtype=np.float32))
        out = [np.mean(self.cache[start + i], axis=0) for i in range(n)]
        for k in [k for k in self.cache if k < start + n]:
            del self.cache[k]
        return out


def run_serial(cfg: ClientConfig):
    """Episodes of: predict a chunk, execute ``exec_steps`` of it, repeat.

    Ctrl-C ends the episode, not the program: the arms ramp back to the pose
    they held before the first episode, the server connection is dropped so the
    next request opens a fresh server-side episode, and the operator picks
    's' (again) or 'q'.
    """
    image_info, robot, ee_shared_mem, client, limiter, log = _setup(cfg)
    _start_pose(cfg, robot, ee_shared_mem)
    arm_ctrl, arm_ik = robot["arm_ctrl"], robot["arm_ik"]
    home_q = arm_ctrl.get_current_dual_arm_q()
    home_grip = _read_grippers(ee_shared_mem)
    server_ensembles = bool(client.metadata.get("action_ensemble"))
    if cfg.ensemble and server_ensembles:
        raise SystemExit("--ensemble averages on the client; restart the server "
                         "with --no-action-ensemble so it is not done twice")

    period = 1.0 / cfg.frequency
    episode = 0
    try:
        while input("Enter 's' to start an episode, 'q' to quit: ").strip().lower() != "q":
            episode += 1
            limiter._last = None
            ensembler = ChunkEnsembler() if cfg.ensemble else None
            queue: list = []
            idx = 0
            logger.info("episode %d started (Ctrl-C ends it and returns home)", episode)
            start_rec, stop_rec = image_info["video"]
            if cfg.video_dir:
                start_rec(Path(cfg.video_dir) / f"{time.strftime('%Y%m%d_%H%M%S')}"
                          f"_port{cfg.server_port}_ep{episode:02d}.mp4")
            try:
                while True:
                    loop_start = time.perf_counter()
                    if not queue:
                        images, state, _ = _observe(cfg, image_info, robot, ee_shared_mem)
                        if not limiter.is_seeded:
                            limiter.reset(state)
                        actions, predict_ms = client.predict(images, state)
                        if cfg.anchor_cap > 0:
                            actions = anchor_chunk(actions, limiter.hold(), state, cfg.anchor_cap)
                        n = min(cfg.exec_steps or len(actions), len(actions))
                        if server_ensembles and n != len(actions):
                            logger.warning("server ensembles and assumes all %d steps run; "
                                           "executing %d desyncs it", len(actions), n)
                        queue = (ensembler.next_actions(actions, idx, n) if ensembler
                                 else list(actions[:n]))
                        logger.info("[%d] server=%.0fms chunk=%d exec=%d ensemble=%s",
                                    idx, predict_ms, len(actions), n, bool(ensembler))
                    _execute(robot, ee_shared_mem, limiter, queue.pop(0), log)
                    idx += 1
                    time.sleep(max(0.0, period - (time.perf_counter() - loop_start)))
            except KeyboardInterrupt:
                logger.info("episode %d stopped after %d steps; returning home", episode, idx)
            finally:
                stop_rec()  # also on a crash, so the mp4 is not left unfinalised
            client.close()  # next predict reconnects = fresh server-side episode
            move_to_arm_qpos(arm_ctrl, arm_ik, ee_shared_mem, home_q, *home_grip,
                             steps=cfg.init_pose_steps)
    except (KeyboardInterrupt, EOFError):
        logger.info("Interrupted")
    finally:
        _shutdown(client, image_info, limiter, log)


def run_async(cfg: ClientConfig):
    """Overlap inference with motion. Needs the server at --no-action-ensemble."""
    image_info, robot, ee_shared_mem, client, limiter, log = _setup(cfg)
    _start_pose(cfg, robot, ee_shared_mem)
    input("Enter 's' to start evaluation: ")

    period = 1.0 / cfg.frequency
    lock = threading.Lock()
    shared = {"busy": False, "req": None, "result": None, "stop": False}

    def worker():
        while True:
            with lock:
                if shared["stop"]:
                    return
                request = shared["req"] if shared["busy"] else None
                shared["req"] = None
            if request is None:
                time.sleep(0.002)
                continue
            try:
                actions, predict_ms = client.predict(*request)
            except Exception as exc:  # noqa: BLE001 - keep the loop alive
                logger.error("predict failed: %s", exc)
                actions, predict_ms = None, 0.0
            with lock:
                shared["result"] = (actions, predict_ms)
                shared["busy"] = False

    thread = threading.Thread(target=worker, daemon=True)
    thread.start()

    images, state, _ = _observe(cfg, image_info, robot, ee_shared_mem)
    limiter.reset(state)
    actions, predict_ms = client.predict(images, state)
    queue = list(actions)
    logger.info("[bootstrap] server=%.0fms chunk=%s", predict_ms, actions.shape)

    pending_mark = None
    idx = 0
    try:
        while True:
            loop_start = time.perf_counter()
            if queue:
                _execute(robot, ee_shared_mem, limiter, queue.pop(0), log)
            else:
                _hold(robot, ee_shared_mem, limiter, log)

            with lock:
                ready = shared["result"]
                shared["result"] = None
            if ready is not None and ready[0] is not None:
                fresh = list(ready[0])
                # The chunk starts at the observation it was built from, so the
                # steps covering the inference latency are already in the past.
                # Commanding them drags the arm backwards and it oscillates.
                stale = int((time.perf_counter() - pending_mark) * cfg.frequency) \
                    if pending_mark else 0
                if stale >= len(fresh):
                    logger.warning("[splice] whole chunk stale (%d late of %d): "
                                   "inference is slower than the chunk is long",
                                   stale, len(fresh))
                    fresh = fresh[-1:]
                elif stale:
                    fresh = fresh[stale:]
                queue = fresh
                pending_mark = None
                logger.info("[splice] server=%.0fms stale=%d queue=%d",
                            ready[1], stale, len(queue))

            with lock:
                idle = not shared["busy"]
            # Re-plan once the queue is down to roughly one inference worth of
            # motion, so the arm is never waiting on the network.
            if idle and pending_mark is None and len(queue) <= max(1, client.action_horizon // 2):
                images, state, _ = _observe(cfg, image_info, robot, ee_shared_mem)
                pending_mark = time.perf_counter()
                with lock:
                    shared["req"] = (images, state)
                    shared["busy"] = True

            idx += 1
            if idx % 30 == 0:
                logger.info("[%d] queue=%d", idx, len(queue))
            time.sleep(max(0.0, period - (time.perf_counter() - loop_start)))
    except KeyboardInterrupt:
        logger.info("Interrupted")
    finally:
        with lock:
            shared["stop"] = True
        thread.join(timeout=2.0)
        _shutdown(client, image_info, limiter, log)


def _shutdown(client, image_info, limiter, log=None):
    if log is not None:
        log.close()
        logger.info("command log: %s", log.path)
    client.close()
    if image_info:
        image_info["image_client"].close()
    logger.info("Done: %d commands range-clamped, %d rate-limited",
                limiter.n_clamped, limiter.n_rate_limited)


def main() -> None:
    global GRIP_SQUEEZE, GRIP_CLOSED_BELOW, INTERNAL_GRIPPER_KP
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--arm", default="G1_29")
    p.add_argument("--ee", default="dex1_internal",
                   help="dex1_internal drives lowstate motors 31/33 (USB Dex1 hangs here)")
    p.add_argument("--frequency", type=float, default=30.0,
                   help="Must match the dataset fps the policy was trained at")
    p.add_argument("--image_host", default="192.168.123.164")
    p.add_argument("--server_host", default=None,
                   help="Omit to pick interactively from MODEL_REGISTRY")
    p.add_argument("--server_port", type=int, default=8000)
    p.add_argument("--jpeg_quality", type=int, default=95,
                   help="0 or less sends raw frames: 2.76 MB per observation, "
                        "55 Mbps to replan at 2.5 Hz, against 3.4 Mbps at q95")
    p.add_argument("--smooth_alpha", type=float, default=1.0,
                   help="1.0 = no EMA; the rate limit already bounds each step")
    p.add_argument("--init_pose_json", default="")
    p.add_argument("--init_pose_frame", type=int, default=0)
    p.add_argument("--init_pose_steps", type=int, default=100)
    p.add_argument("--capture_ready_pose", default="",
                   help="Hand-pose the arms and save the result to this path as "
                        "an --init_pose_json file, instead of running the policy")
    p.add_argument("--async_inference", action="store_true",
                   help="Overlap inference with motion; needs --no-action-ensemble "
                        "on the server")
    # Passed straight through to unitree_lerobot's setup_*; this client does not
    # read them, but those functions do.
    p.add_argument("--no_swap_wrists", action="store_true",
                   help="Read cam_left_wrist as wrist_left. Default swaps them, "
                        "on the operator's reading of the recorded frames; the "
                        "dataset correlation says pass-through. Settle it with "
                        "the live check, not by argument")
    p.add_argument("--record_log", default="",
                   help="JSONL of requested vs commanded vs measured joint16 per "
                        "control step. The server only sees what was requested")
    p.add_argument("--motion", action="store_true")
    p.add_argument("--sim", action="store_true")
    p.add_argument("--base_type", default="legs")
    p.add_argument("--video_dir", default="episode_videos",
                   help="Per-episode 2x2 camera mp4 (what the policy sees); '' disables")
    p.add_argument("--prompt", default=None,
                   help="Instruction for OpenWAM servers; skips the picker (empty string = server default)")
    p.add_argument("--exec_steps", type=int, default=0,
                   help="Actions executed per chunk (serial mode); 0 = all the server returns")
    p.add_argument("--ensemble", action="store_true",
                   help="Average overlapping chunks on the client (needs exec_steps < chunk)")
    p.add_argument("--anchor_cap", type=float, default=0.1,
                   help="Re-anchor each chunk on the last command, offset capped at this (rad); 0 disables")
    p.add_argument("--web_port", type=int, default=8088,
                   help="Live MJPEG view of the policy's cameras; 0 disables")
    p.add_argument("--grip_squeeze", type=float, default=GRIP_SQUEEZE,
                   help="Extra closing (rad) when the grip command is in the closed half; 0 disables")
    p.add_argument("--grip_closed_below", type=float, default=GRIP_CLOSED_BELOW)
    p.add_argument("--grip_kp", type=float, default=INTERNAL_GRIPPER_KP)
    args = p.parse_args()
    GRIP_SQUEEZE, GRIP_CLOSED_BELOW = args.grip_squeeze, args.grip_closed_below
    INTERNAL_GRIPPER_KP = args.grip_kp
    # unitree_lerobot opens its URDF and camera config by cwd-relative paths, so
    # run from its root; resolve our own path arguments first.
    for name in ("init_pose_json", "capture_ready_pose", "record_log", "video_dir"):
        if getattr(args, name):
            setattr(args, name, os.path.abspath(getattr(args, name)))
    os.chdir(LEROBOT_ROOT)

    if args.capture_ready_pose:
        cfg = ClientConfig(arm=args.arm, ee=args.ee, motion=args.motion,
                           sim=args.sim, base_type=args.base_type)
        capture_ready_pose(cfg, args.capture_ready_pose)
        return

    server_host, server_port = (
        (args.server_host, args.server_port) if args.server_host
        else choose_server())

    cfg = ClientConfig(
        arm=args.arm, ee=args.ee, frequency=args.frequency,
        image_host=args.image_host,
        motion=args.motion, sim=args.sim, base_type=args.base_type,
        server_host=server_host, server_port=server_port,
        jpeg_quality=args.jpeg_quality if args.jpeg_quality > 0 else None,
        init_pose_json=args.init_pose_json, init_pose_frame=args.init_pose_frame,
        init_pose_steps=args.init_pose_steps,
        async_inference=args.async_inference, smooth_alpha=args.smooth_alpha,
        record_log=args.record_log, web_port=args.web_port,
        anchor_cap=args.anchor_cap, exec_steps=args.exec_steps, video_dir=args.video_dir, ensemble=args.ensemble,
        swap_wrists=not args.no_swap_wrists, prompt=args.prompt,
    )
    (run_async if cfg.async_inference else run_serial)(cfg)


if __name__ == "__main__":
    main()
