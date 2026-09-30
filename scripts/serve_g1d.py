"""Serve an OpenWAM G1-Dex1 checkpoint to the Unitree robot client (openpi WebSocket protocol).

The robot-side client (``g1d_jepa_client.py``) already speaks: msgpack frames, one metadata frame on
connect, then ``observation/state`` (joint16) + ``observation/images/{head_left,wrist_left,wrist_right}``
in, ``actions`` ``[horizon, 16]`` absolute joint targets out. OpenWAM predicts torso-frame EEF20, so this
server does FK on the incoming joints (proprio) and damped-least-squares IK on the outgoing chunk.

    python scripts/serve_g1d.py --ckpt-dir CKPT --port 8001 \
        --instruction "Put the capybara plush into the box."

    python scripts/serve_g1d.py --self-test        # FK/IK round trip, no GPU
"""

from __future__ import annotations

import argparse
import asyncio
import functools
import http
import sys
import time
import traceback
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from openwam.dataloader.utils.eef import euler_xyz_to_rot6d  # noqa: E402
from openwam.dataloader.utils.g1d_self_convert import (  # noqa: E402
    ARM_JOINTS,
    RAW_OPEN_SELF,
    SIDES,
    G1ArmFK,
)

JOINT16 = [f"{s}_{j}" for s in SIDES for j in ARM_JOINTS] + ["left_gripper", "right_gripper"]
CAMERAS = ["head_left", "wrist_left", "wrist_right"]
CAM_TO_OWAM = {"head_left": "head_camera", "wrist_left": "left_wrist_camera", "wrist_right": "right_wrist_camera"}
STATE_KEY, IMAGE_PREFIX = "observation/state", "observation/images/"
ARM_OFF = {"left": 0, "right": 10}  # EEF20 layout per arm: xyz3, rot6d6, open1


# --------------------------------------------------------------------- kinematics

def rot6d_to_matrix(r6: np.ndarray) -> np.ndarray:
    a1, a2 = r6[..., :3], r6[..., 3:6]
    b1 = a1 / np.linalg.norm(a1, axis=-1, keepdims=True)
    b2 = a2 - (b1 * a2).sum(-1, keepdims=True) * b1
    b2 = b2 / np.linalg.norm(b2, axis=-1, keepdims=True)
    return np.stack([b1, b2, np.cross(b1, b2)], axis=-1)


def joints_to_eef20(fk: G1ArmFK, state: np.ndarray) -> np.ndarray:
    """Client joint16 state → the EEF20 proprio the checkpoint was trained on."""
    out = np.zeros(20, np.float32)
    for side, o in ARM_OFF.items():
        q = state[0:7] if side == "left" else state[7:14]
        pose = fk.pose(side, q[None])
        out[o : o + 3] = pose[0, :3]
        out[o + 3 : o + 9] = euler_xyz_to_rot6d(pose[:, 3:6])[0]
        out[o + 9] = np.clip(state[14 if side == "left" else 15] / RAW_OPEN_SELF, 0, 1) * 2 - 1
    return out


def solve_ik(fk, side, q0, R_t, p_t, lo, hi, iters=30, damping=0.01, null_gain=0.1):
    """Sequential DLS IK, warm-started from ``q0`` and then from the previous step.

    Target rotation error is a rotation vector, so Euler wrap-around never matters.
    The 7th DOF is redundant: it is pulled softly toward the measured pose ``q0``.
    """
    eps = 1e-5
    q, out = q0.copy(), []
    for R_g, p_g in zip(R_t, p_t):
        for _ in range(iters):
            R, p = fk.transform(side, np.vstack([q, q + eps * np.eye(7)]))
            err = np.concatenate([p_g - p[0], Rotation.from_matrix(R_g @ R[0].T).as_rotvec()])
            if np.linalg.norm(err[:3]) < 1e-4 and np.linalg.norm(err[3:]) < 1e-3:
                break
            J = np.zeros((6, 7))
            J[:3] = (p[1:] - p[0]).T / eps
            dR = R[1:] @ R[0].T  # ~ I + skew(eps * column): read the axis off the skew part, no scipy
            J[3:] = np.stack([dR[:, 2, 1] - dR[:, 1, 2], dR[:, 0, 2] - dR[:, 2, 0], dR[:, 1, 0] - dR[:, 0, 1]]) / (2 * eps)
            J_pinv = J.T @ np.linalg.inv(J @ J.T + damping**2 * np.eye(6))
            dq = J_pinv @ err + (np.eye(7) - J_pinv @ J) @ (null_gain * (q0 - q))
            q = np.clip(q + np.clip(dq, -0.3, 0.3), lo, hi)
        out.append(q.copy())
    return np.array(out)


def eef20_to_joints(fk, pred, q_now, limits) -> np.ndarray:
    """``(T, 20)`` EEF20 chunk → ``(T, 16)`` absolute joint targets (raw Dex1 gripper units)."""
    out = np.zeros((len(pred), 16), np.float32)
    for side, o in ARM_OFF.items():
        sl = slice(0, 7) if side == "left" else slice(7, 14)
        R = rot6d_to_matrix(pred[:, o + 3 : o + 9].astype(np.float64))
        out[:, sl] = solve_ik(fk, side, q_now[sl], R, pred[:, o : o + 3].astype(np.float64), *limits[side])
        out[:, 14 if side == "left" else 15] = (np.clip(pred[:, o + 9], -1, 1) + 1) / 2 * RAW_OPEN_SELF
    return out


# --------------------------------------------------------------------- wire format

def _codec():
    import msgpack

    def pack(obj):
        if isinstance(obj, np.ndarray):
            return {b"__ndarray__": True, b"data": obj.tobytes(), b"dtype": obj.dtype.str, b"shape": obj.shape}
        if isinstance(obj, np.generic):
            return {b"__npgeneric__": True, b"data": obj.item(), b"dtype": obj.dtype.str}
        return obj

    def unpack(obj):
        if b"__ndarray__" in obj:
            return np.ndarray(buffer=obj[b"data"], dtype=np.dtype(obj[b"dtype"]), shape=obj[b"shape"])
        if b"__npgeneric__" in obj:
            return np.dtype(obj[b"dtype"]).type(obj[b"data"])
        return obj

    return functools.partial(msgpack.packb, default=pack), functools.partial(msgpack.unpackb, object_hook=unpack)


def decode_frame(value):
    """Raw HWC uint8 RGB array, or JPEG bytes from ``cv2.imencode`` of an RGB array (round-trips as RGB)."""
    if isinstance(value, (bytes, bytearray, memoryview)):
        import cv2

        value = cv2.imdecode(np.frombuffer(value, np.uint8), cv2.IMREAD_COLOR)
        if value is None:
            raise ValueError("camera bytes do not decode as an image")
    return np.asarray(value, np.uint8)


# --------------------------------------------------------------------- policy

class G1DPolicy:
    def __init__(self, ckpt_dir, device, instruction, exec_horizon, denoise_steps, compile_):
        from omegaconf import OmegaConf

        from openwam.deploy import JointInferenceEngine
        from openwam.deploy.model_loader import load_from_checkpoint_dir
        from openwam.deploy.obs_preprocess import ObsPreprocessor
        from openwam.deploy.server import merge_deploy_cfg

        train_cfg, arch = load_from_checkpoint_dir(ckpt_dir, device=device)
        deploy = OmegaConf.load(ROOT / "configs" / "deploy.yaml")
        OmegaConf.update(deploy, "inference.denoise_steps", denoise_steps)
        OmegaConf.update(deploy, "optimization.compile.enabled", compile_)
        self.cfg = merge_deploy_cfg(train_cfg, deploy)
        self.engine = JointInferenceEngine(cfg=self.cfg, architecture=arch)
        self.obs = ObsPreprocessor.from_cfg(self.cfg, self.engine)
        self.chunk = int(self.cfg.inference.num_frames) - 1
        self.horizon = min(exec_horizon, self.chunk)
        self.instruction = instruction
        self.fk = G1ArmFK()
        self.limits = {s: self.fk.limits(s) for s in SIDES}

    def predict(self, obs: dict) -> np.ndarray:
        from PIL import Image

        state = np.asarray(obs[STATE_KEY], np.float64).reshape(-1)
        prompt = obs.get("prompt") or self.instruction
        images = {CAM_TO_OWAM[c]: Image.fromarray(decode_frame(obs[IMAGE_PREFIX + c])) for c in CAMERAS}
        proprio = joints_to_eef20(self.fk, state)
        pre = self.obs.preprocess({"images": images, "prompt": prompt, "state": proprio})
        result = self.engine.generate({"first_frame_image": [pre["image"]], "prompt": pre["prompt"], "proprio": proprio, "seed": 0})
        pred = np.asarray(result["actions"])[: self.horizon, :20]
        return eef20_to_joints(self.fk, pred, state[:14], self.limits)

    def metadata(self, max_step: float) -> dict:
        lo = np.concatenate([self.limits["left"][0], self.limits["right"][0], [0.0, 0.0]])
        hi = np.concatenate([self.limits["left"][1], self.limits["right"][1], [RAW_OPEN_SELF] * 2])
        return {
            "policy": "OpenWAMG1Dex1EefIkPolicy",
            "embodiment_schema": "unitree_g1_dex1_joint16_v1",
            "joint_names": JOINT16,
            "cameras": CAMERAS,
            "observation_keys": {"state": STATE_KEY, "images": {c: IMAGE_PREFIX + c for c in CAMERAS}},
            "frame_encodings": ["raw_hwc_uint8_rgb", "jpeg_bytes_via_cv2_imencode"],
            "action_key": "actions",
            "action_dim": 16,
            "action_space": "absolute joint position, radians, JOINT16 order (gripper: raw Dex1 units)",
            "chunk_size": self.chunk,
            "action_horizon": self.horizon,
            "action_ensemble": False,
            "control_hz": 30.0,
            "n_obs_steps": 1,
            "gripper_dims": [14, 15],
            "binarize_gripper": False,
            "joint_limits": {"lower": lo.tolist(), "upper": hi.tolist()},
            "max_step_rad": max_step,
            "instruction": self.instruction,
            "reset": "stateless; a new connection per episode is fine",
        }


async def serve(policy, host, port, max_step):
    import websockets
    import websockets.asyncio.server as server
    import websockets.frames

    packb, unpackb = _codec()
    meta = policy.metadata(max_step)

    async def handler(ws):
        await ws.send(packb(meta))
        while True:
            try:
                obs = unpackb(await ws.recv())
                t0 = time.monotonic()
                actions = policy.predict(obs)
                ms = (time.monotonic() - t0) * 1000
                print(f"infer {ms:.0f} ms, chunk {actions.shape}", flush=True)
                await ws.send(packb({"actions": np.ascontiguousarray(actions, np.float32), "server_timing": {"infer_ms": ms}}))
            except websockets.ConnectionClosed:
                break
            except Exception:
                await ws.send(traceback.format_exc())  # str frame == error for the openpi client
                await ws.close(code=websockets.frames.CloseCode.INTERNAL_ERROR)
                raise

    def health(conn, request):
        return conn.respond(http.HTTPStatus.OK, "OK\n") if request.path == "/healthz" else None

    # ping_interval=None: inference blocks this event loop, and a keepalive deadline would drop the client
    async with server.serve(handler, host, port, compression=None, max_size=None, ping_interval=None, process_request=health) as s:
        print(f"serving {policy.instruction!r} on ws://{host}:{port} horizon={policy.horizon}", flush=True)
        await s.serve_forever()


# --------------------------------------------------------------------- self test

def self_test() -> None:
    """FK → EEF20 → IK reproduces the pose (mm / deg); rot6d and gripper conventions round-trip."""
    fk = G1ArmFK()
    rng = np.random.default_rng(0)
    limits = {s: fk.limits(s) for s in SIDES}
    q0 = np.array([0.1, 0.2, 0.0, 0.6, 0.0, 0.0, 0.0])
    q0 = np.concatenate([q0, q0 * [1, -1, 1, 1, 1, 1, 1]])
    traj = q0[None] + np.cumsum(rng.normal(0, 0.01, (30, 14)), 0)  # smooth 30-step move, both arms
    eef = np.stack([joints_to_eef20(fk, np.concatenate([q, [5.4, 2.0]])) for q in traj])
    t0 = time.time()
    out = eef20_to_joints(fk, eef, q0, limits)
    print(f"IK 30 steps x 2 arms: {time.time() - t0:.2f}s")
    for side, sl in (("left", slice(0, 7)), ("right", slice(7, 14))):
        o = ARM_OFF[side]
        R, p = fk.transform(side, out[:, sl])
        Rt = rot6d_to_matrix(eef[:, o + 3 : o + 9].astype(np.float64))
        pos_mm = np.linalg.norm(p - eef[:, o : o + 3], axis=1).max() * 1000
        rot_deg = np.degrees(np.linalg.norm(Rotation.from_matrix(Rt @ R.transpose(0, 2, 1)).as_rotvec(), axis=1)).max()
        print(f"{side}: max pos err {pos_mm:.3f} mm, rot err {rot_deg:.3f} deg")
        assert pos_mm < 1.0 and rot_deg < 0.5, (side, pos_mm, rot_deg)
    R, _ = fk.transform("left", traj[:1, 0:7])  # rot6d = first two matrix columns
    assert np.allclose(rot6d_to_matrix(eef[:1, 3:9].astype(np.float64)), R, atol=1e-5)
    assert abs(eef[0, 9] - 1) < 1e-6 and abs(out[0, 15] - 2.0) < 1e-3  # open -> +1; 2.0 raw round-trips
    print("self-test passed")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ckpt-dir")
    p.add_argument("--instruction", default="Put the capybara plush into the box.")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=8001)
    p.add_argument("--exec-horizon", type=int, default=30)
    p.add_argument("--denoise-steps", type=int, default=10)
    p.add_argument("--max-step-rad", type=float, default=0.05)
    p.add_argument("--compile", action="store_true")
    p.add_argument("--self-test", action="store_true")
    args = p.parse_args()
    if args.self_test:
        return self_test()
    policy = G1DPolicy(args.ckpt_dir, args.device, args.instruction, args.exec_horizon, args.denoise_steps, args.compile)
    # torch.compile warm-up (~15 s) must not land on the robot's first request
    warm = {STATE_KEY: np.array([0.1, 0.2, 0, 0.6, 0, 0, 0, 0.1, -0.2, 0, 0.6, 0, 0, 0, 5.4, 5.4], np.float32)}
    warm.update({IMAGE_PREFIX + c: np.zeros((480, 640, 3), np.uint8) for c in CAMERAS})
    for _ in range(2):
        policy.predict(warm)
    asyncio.run(serve(policy, args.host, args.port, args.max_step_rad))


if __name__ == "__main__":
    main()
