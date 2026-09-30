#!/usr/bin/env python3
"""Drop-in client for the JEPA-WAM G1-Dex1 policy server, for the robot side.

Written to be **copied into the robot-side repo** next to
``motus_inference_client.py``, so it deliberately imports nothing from
JEPA-WAM. Only ``numpy``, ``msgpack`` and ``cv2`` are needed -- no ``openpi``,
and no ``websocket-client`` (the framing below is the same hand-rolled RFC6455
approach that client already uses, extended to binary frames, which the text-only
version cannot carry: msgpack arrives as opcode 0x2 and would raise
``unsupported WebSocket opcode 2``).

Three call-site edits in ``motus_inference_client.py`` are needed, and all three
are deletions:

1. ``observation_to_model_input`` -- return ``raw_state`` and the three images,
   not ``_to_arm_interleaved(raw_state)`` and the stitched canvas::

       # state is already JOINT16: [L7, R7, LG, RG]. Do NOT interleave.
       return {"head_left": head_img,
               "wrist_left": left_wrist,
               "wrist_right": right_wrist}, raw_state.astype(np.float32)

   Keep the cameras at full resolution: ``build_stitched_image`` leaves the head
   at roughly 240x640 and each wrist at 240x320 inside the canvas, while these
   features were encoded from 480x640 per camera.

2. ``SWAP_WRISTS=0``. That flag corrects a key swap for MotusV2, but these
   features were encoded straight off the same swapped dataset keys, so applying
   it again inverts the wrists twice.

3. ``qpos_action_to_g1_action`` -- the returned chunk is already in raw order,
   so skip ``_from_arm_interleaved``::

       arm_action = action16[0:14]
       left_grip, right_grip = float(action16[14]), float(action16[15])

Everything else in that client carries over: the 30Hz loop, the async
double-buffer, the staleness trim on splice, and the smoothing. Two notes:

* ``action_horizon`` comes from the handshake (currently 12 of a 30-action
  chunk). Use it for ``exec_chunk_steps`` -- the server's chunk ensembler
  advances its clock by the length it returned, so replanning early or late
  desynchronises it.
* A reconnect starts a **new episode** server-side: the observation history and
  the chunk ensembler reset. That is correct between rollouts and harmless
  mid-rollout, but the chunk clock restarts, so do not treat a reconnect as
  transparent.
"""
from __future__ import annotations

import base64
import hashlib
import os
import socket
import struct
import time
from urllib.parse import urlparse

import numpy as np

__all__ = ["JepaWamClient"]


# --------------------------------------------------------------- msgpack

def _pack_array(obj):
    """NumPy -> msgpack-encodable, matching openpi's wire layout."""
    if isinstance(obj, (np.ndarray, np.generic)) and obj.dtype.kind in ("V", "O", "c"):
        raise ValueError(f"Unsupported dtype: {obj.dtype}")
    if isinstance(obj, np.ndarray):
        return {b"__ndarray__": True, b"data": obj.tobytes(),
                b"dtype": obj.dtype.str, b"shape": obj.shape}
    if isinstance(obj, np.generic):
        return {b"__npgeneric__": True, b"data": obj.item(), b"dtype": obj.dtype.str}
    return obj


def _unpack_array(obj):
    if b"__ndarray__" in obj:
        return np.ndarray(buffer=obj[b"data"], dtype=np.dtype(obj[b"dtype"]),
                          shape=obj[b"shape"])
    if b"__npgeneric__" in obj:
        return np.dtype(obj[b"dtype"]).type(obj[b"data"])
    return obj


# ------------------------------------------------------------- transport

class _BinaryWebSocket:
    """Minimal RFC6455 client that can carry binary frames.

    Mirrors the robot client's own ``_StdlibWebSocket`` so the robot still needs
    no ``websocket-client``, but sends and receives opcode 0x2 instead of 0x1.
    """

    def __init__(self, url: str, timeout: float):
        parsed = urlparse(url)
        host, port = parsed.hostname, parsed.port or 80
        path = parsed.path or "/"
        if parsed.query:
            path = f"{path}?{parsed.query}"
        key = base64.b64encode(os.urandom(16)).decode("ascii")
        request = (
            f"GET {path} HTTP/1.1\r\nHost: {host}:{port}\r\n"
            "Upgrade: websocket\r\nConnection: Upgrade\r\n"
            f"Sec-WebSocket-Key: {key}\r\nSec-WebSocket-Version: 13\r\n\r\n"
        )
        sock = socket.create_connection((host, port), timeout=timeout)
        sock.settimeout(timeout)
        sock.sendall(request.encode("ascii"))
        buffer = b""
        while b"\r\n\r\n" not in buffer:
            chunk = sock.recv(4096)
            if not chunk:
                sock.close()
                raise RuntimeError(f"WebSocket handshake failed: empty reply from {url}")
            buffer += chunk
        header, _, leftover = buffer.partition(b"\r\n\r\n")
        status = header.split(b"\r\n", 1)[0]
        if b"101" not in status:
            sock.close()
            raise RuntimeError(
                f"WebSocket handshake failed: {status.decode('ascii', 'replace')}")
        expect = base64.b64encode(hashlib.sha1(
            (key + "258EAFA5-E914-47DA-95CA-C5AB0DC85B11").encode("ascii")).digest())
        if expect not in header:
            raise RuntimeError("WebSocket accept key mismatch")
        self._sock = sock
        # Whatever followed the header is already the first frame and must not be
        # dropped. This server sends its metadata immediately on connect, so that
        # frame routinely shares a TCP segment with the 101 response -- discarding
        # it makes the metadata parse from the middle of the frame, intermittently
        # and depending only on segment boundaries.
        self._pending = leftover

    def settimeout(self, timeout):
        self._sock.settimeout(timeout)

    def send_binary(self, payload: bytes) -> None:
        mask = os.urandom(4)
        header = bytearray([0x82])  # FIN + binary
        n = len(payload)
        if n < 126:
            header.append(0x80 | n)
        elif n < 65536:
            header.append(0x80 | 126)
            header.extend(struct.pack("!H", n))
        else:
            header.append(0x80 | 127)
            header.extend(struct.pack("!Q", n))
        header.extend(mask)
        masked = bytes(b ^ mask[i % 4] for i, b in enumerate(payload))
        self._sock.sendall(bytes(header) + masked)

    def recv_frame(self):
        """Return ``(is_text, payload)``; the server sends errors as text."""
        def read_exact(n):
            buf = bytearray()
            if self._pending:
                take = self._pending[:n]
                self._pending = self._pending[len(take):]
                buf.extend(take)
            while len(buf) < n:
                chunk = self._sock.recv(n - len(buf))
                if not chunk:
                    raise RuntimeError("WebSocket closed while reading")
                buf.extend(chunk)
            return bytes(buf)

        b0, b1 = read_exact(2)
        opcode = b0 & 0x0F
        masked = bool(b1 & 0x80)
        n = b1 & 0x7F
        if n == 126:
            n = struct.unpack("!H", read_exact(2))[0]
        elif n == 127:
            n = struct.unpack("!Q", read_exact(8))[0]
        mask = read_exact(4) if masked else b""
        payload = bytearray(read_exact(n))
        if masked:
            payload = bytearray(b ^ mask[i % 4] for i, b in enumerate(payload))
        if opcode == 0x8:
            raise RuntimeError("WebSocket closed by server")
        if opcode == 0x9:  # ping -> pong, then keep waiting
            self._send_pong(bytes(payload)[:125])
            return self.recv_frame()
        if opcode == 0xA:  # stray pong
            return self.recv_frame()
        if opcode not in (0x1, 0x2):
            raise RuntimeError(f"unsupported WebSocket opcode {opcode}")
        return opcode == 0x1, bytes(payload)

    def _send_pong(self, payload: bytes) -> None:
        mask = os.urandom(4)
        header = bytearray([0x8A, 0x80 | len(payload)])
        header.extend(mask)
        masked = bytes(b ^ mask[i % 4] for i, b in enumerate(payload))
        self._sock.sendall(bytes(header) + masked)

    def close(self) -> None:
        try:
            self._sock.sendall(b"\x88\x80\x00\x00\x00\x00")
        except Exception:
            pass
        try:
            self._sock.close()
        except Exception:
            pass


# ---------------------------------------------------------------- client

class JepaWamClient:
    """Talks to the JEPA-WAM G1-Dex1 policy server over the openpi protocol.

    ``jpeg_quality=None`` sends raw frames, which is 2.76 MB per observation for
    three 480x640 cameras and wants 55 Mbps sustained to replan at 2.5 Hz. q95 is
    ~16x smaller. Encoding goes through ``cv2.imencode`` because the server
    decodes with ``cv2.imdecode``: the pair is symmetric, so an RGB array comes
    back as RGB. Encoding with PIL instead would swap R and B, and nothing
    downstream can detect that -- the policy would just do slightly worse.
    """

    def __init__(self, server_host: str, server_port: int, *, timeout_ms: int = 60000,
                 jpeg_quality: int | None = 95):
        self.url = f"ws://{server_host}:{server_port}"
        self.timeout_s = max(1.0, timeout_ms / 1000.0)
        self.jpeg_quality = jpeg_quality
        self._ws = None
        self.metadata: dict = {}
        self.prompt = ""  # sent with every request when set (language-conditioned servers only)
        self._connect()

    # -- contract published by the handshake -----------------------------

    @property
    def action_horizon(self) -> int:
        """Actions to execute per call, before asking again."""
        return int(self.metadata["action_horizon"])

    @property
    def cameras(self) -> list:
        return list(self.metadata["cameras"])

    @property
    def joint_limits(self):
        """``(lower, upper)`` the fine-tune data covered -- clamp against these."""
        limits = self.metadata["joint_limits"]
        return np.asarray(limits["lower"]), np.asarray(limits["upper"])

    @property
    def max_step_rad(self) -> float:
        """Per-control-step cap the rate limiter should use, in radians."""
        return float(self.metadata["max_step_rad"])

    # -- transport -------------------------------------------------------

    def _connect(self) -> None:
        import msgpack

        if self._ws is not None:
            self._ws.close()
        self._ws = _BinaryWebSocket(self.url, timeout=self.timeout_s)
        is_text, payload = self._ws.recv_frame()
        if is_text:
            raise RuntimeError(f"server error on connect: {payload.decode('utf-8', 'replace')}")
        self.metadata = msgpack.unpackb(payload, object_hook=_unpack_array)

    def _encode_images(self, images: dict) -> dict:
        fields = {}
        for camera in self.cameras:
            if camera not in images:
                raise KeyError(f"missing camera {camera!r}; server wants {self.cameras}")
            frame = np.ascontiguousarray(images[camera][:, :, :3], dtype=np.uint8)
            key = f"observation/images/{camera}"
            if self.jpeg_quality is None:
                fields[key] = frame
            else:
                import cv2

                ok, buf = cv2.imencode(
                    ".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), int(self.jpeg_quality)])
                if not ok:
                    raise RuntimeError(f"could not JPEG-encode {camera!r}")
                fields[key] = buf.tobytes()
        return fields

    def predict(self, images: dict, state16) -> tuple:
        """Three RGB frames plus a joint16 state -> ``(actions, predict_ms)``.

        ``state16`` is ``[L7, R7, LG, RG]`` in radians -- the client's own
        ``raw_state``, *before* ``_to_arm_interleaved``. ``actions`` comes back in
        the same order, shaped ``(action_horizon, 16)``, as absolute joint
        positions.
        """
        import msgpack

        state = np.asarray(state16, dtype=np.float32).reshape(-1)
        if state.shape != (16,):
            raise ValueError(f"state must be 16-D joint16, got {state.shape}")
        request = {"observation/state": state, **self._encode_images(images)}
        if self.prompt:
            request["prompt"] = self.prompt
        blob = msgpack.packb(request, default=_pack_array)

        last_error = None
        for _ in range(2):
            try:
                if self._ws is None:
                    self._connect()
                self._ws.settimeout(self.timeout_s)
                started = time.perf_counter()
                self._ws.send_binary(blob)
                is_text, payload = self._ws.recv_frame()
                if is_text:
                    # The server reports errors as one text frame, then closes.
                    raise RuntimeError(
                        f"server error: {payload.decode('utf-8', 'replace')}")
                reply = msgpack.unpackb(payload, object_hook=_unpack_array)
                actions = np.asarray(reply["actions"], dtype=np.float32)
                if actions.ndim != 2 or actions.shape[-1] != 16:
                    raise RuntimeError(f"expected (*, 16) actions, got {actions.shape}")
                predict_ms = float(reply.get("server_timing", {}).get(
                    "infer_ms", (time.perf_counter() - started) * 1e3))
                return actions, predict_ms
            except Exception as exc:  # noqa: BLE001 - reconnect once, then report
                last_error = exc
                # Close before dropping the reference: the server tracks one
                # episode per live connection, and losing the handle here
                # without closing it leaves that episode "connected" from the
                # server's side, so the retry's _connect() gets rejected with
                # "another episode is already connected" instead of a fresh
                # handshake -- self-inflicted, not a real second client.
                if self._ws is not None:
                    self._ws.close()
                self._ws = None
        raise RuntimeError(f"JEPA-WAM inference failed: {last_error}") from last_error

    def close(self) -> None:
        if self._ws is not None:
            self._ws.close()
            self._ws = None


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Smoke-test the adapter")
    parser.add_argument("--server_host", required=True)
    parser.add_argument("--server_port", type=int, default=8000)
    parser.add_argument("--jpeg_quality", type=int, default=95)
    args = parser.parse_args()

    client = JepaWamClient(args.server_host, args.server_port,
                           jpeg_quality=args.jpeg_quality)
    print("action_horizon:", client.action_horizon)
    print("cameras:", client.cameras)
    print("max_step_rad:", client.max_step_rad)
    lower, upper = client.joint_limits
    print("gripper limits: left", (lower[14], upper[14]), "right", (lower[15], upper[15]))

    mid = ((lower + upper) / 2).astype(np.float32)
    frames = {cam: np.full((480, 640, 3), 128, dtype=np.uint8) for cam in client.cameras}
    actions, predict_ms = client.predict(frames, mid)
    print(f"actions {actions.shape} {actions.dtype}  server infer {predict_ms:.0f} ms")
    client.close()
