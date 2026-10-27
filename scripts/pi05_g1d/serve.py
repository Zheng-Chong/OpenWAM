"""Serve a pi0.5 G1-D checkpoint (config from openpi_g1d.py) over the protocol of scripts/serve_g1d.py, spoken by
RoboDojo's scripts/g1d/remote_policy.py and the robot client: one metadata frame on connect, then
observation/state joint16 = [left arm 7, right arm 7, left gripper, right gripper] (grippers in raw Dex1 units,
0 closed .. 5.4 open), observation/images/{head_left,wrist_left,wrist_right} (HWC uint8 RGB) and prompt in,
actions [exec_horizon, 16] absolute targets in the same layout out.

Run inside the openpi uv env:
    python serve.py --ckpt-dir <checkpoints>/<run>/<step> --port 8301 [--exec-horizon 30]
"""

import argparse
import asyncio
import http
import time
import traceback

import numpy as np
from openpi.policies import policy_config
from openpi_client import msgpack_numpy

import openpi_g1d

RAW_OPEN = 5.4
STATE_KEY, IMAGE_PREFIX = "observation/state", "observation/images/"
CAMERAS = {"cam_high": "head_left", "cam_left_wrist": "wrist_left", "cam_right_wrist": "wrist_right"}
WIRE_TO_MODEL = np.r_[0:7, 14, 7:14, 15]
MODEL_TO_WIRE = np.argsort(WIRE_TO_MODEL)
DEFAULT_PROMPT = "Pick up the tall measuring cup and pour all the red beans into the short measuring cup."


class G1DPi05Policy:
    def __init__(self, ckpt_dir: str, exec_horizon: int):
        self.policy = policy_config.create_trained_policy(openpi_g1d.CONFIG, ckpt_dir)
        self.horizon = min(exec_horizon, openpi_g1d.CONFIG.model.action_horizon)
        self.ckpt_dir = ckpt_dir

    def predict(self, obs: dict) -> np.ndarray:
        state = np.asarray(obs[STATE_KEY], np.float32).reshape(-1)[WIRE_TO_MODEL]
        state[[7, 15]] /= RAW_OPEN
        images = {cam: np.asarray(obs[IMAGE_PREFIX + name], np.uint8) for cam, name in CAMERAS.items()}
        out = self.policy.infer({"state": state, "images": images, "prompt": obs.get("prompt") or DEFAULT_PROMPT})
        actions = np.asarray(out["actions"][: self.horizon], np.float32)[:, MODEL_TO_WIRE]
        actions[:, 14:] = np.clip(actions[:, 14:], 0, 1) * RAW_OPEN
        return actions

    def metadata(self) -> dict:
        return {
            "policy": "Pi05G1DJointPolicy",
            "checkpoint": self.ckpt_dir,
            "embodiment_schema": "unitree_g1_dex1_joint16_v1",
            "observation_keys": {"state": STATE_KEY, "images": {c: IMAGE_PREFIX + c for c in CAMERAS.values()}},
            "frame_encodings": ["raw_hwc_uint8_rgb"],
            "action_key": "actions",
            "action_dim": 16,
            "action_space": "absolute joint position, radians, JOINT16 order (gripper: raw Dex1 units)",
            "chunk_size": openpi_g1d.CONFIG.model.action_horizon,
            "action_horizon": self.horizon,
            "control_hz": 30.0,
            "gripper_dims": [14, 15],
            "reset": "stateless; a new connection per episode is fine",
        }


async def serve(policy: G1DPi05Policy, host: str, port: int) -> None:
    import websockets
    import websockets.asyncio.server as server
    import websockets.frames

    packer = msgpack_numpy.Packer()

    async def handler(ws):
        await ws.send(packer.pack(policy.metadata()))
        while True:
            try:
                obs = msgpack_numpy.unpackb(await ws.recv())
                t0 = time.monotonic()
                actions = policy.predict(obs)
                ms = (time.monotonic() - t0) * 1000
                print(f"infer {ms:.0f} ms, chunk {actions.shape}", flush=True)
                await ws.send(packer.pack({"actions": actions, "server_timing": {"infer_ms": ms}}))
            except websockets.ConnectionClosed:
                break
            except Exception:
                await ws.send(traceback.format_exc())  # str frame == error for the client
                await ws.close(code=websockets.frames.CloseCode.INTERNAL_ERROR)
                raise

    def health(conn, request):
        return conn.respond(http.HTTPStatus.OK, "OK\n") if request.path == "/healthz" else None

    # ping_interval=None: inference blocks this event loop, and a keepalive deadline would drop the client
    async with server.serve(handler, host, port, compression=None, max_size=None, ping_interval=None, process_request=health) as s:
        print(f"serving {policy.ckpt_dir} on ws://{host}:{port} horizon={policy.horizon}", flush=True)
        await s.serve_forever()


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ckpt-dir", required=True)
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8301)
    p.add_argument("--exec-horizon", type=int, default=30)
    args = p.parse_args()
    policy = G1DPi05Policy(args.ckpt_dir, args.exec_horizon)
    # XLA compilation must not land on the first episode's request
    warm = {STATE_KEY: np.array([0.1, 0.2, 0, 0.6, 0, 0, 0, 0.1, -0.2, 0, 0.6, 0, 0, 0, 5.4, 5.4], np.float32)}
    warm.update({IMAGE_PREFIX + c: np.zeros((480, 640, 3), np.uint8) for c in CAMERAS.values()})
    for _ in range(2):
        policy.predict(warm)
    asyncio.run(serve(policy, args.host, args.port))


if __name__ == "__main__":
    main()
