#!/bin/bash
# JEPA-WAM G1-Dex1 pour-beans (desk) client — robot side.
#
# Server: scripts/eval/g1_dex1/policy_server.py on the GPU box, WebSocket 8000,
# reachable at ws://<gateway>:10050 through the reverse tunnel. Unlike
# MotusV2 there is no ZMQ path and no YAML config: chunk geometry, joint limits
# and the step cap all come from the server handshake.
#
# Prerequisites on the robot:
#   pip install msgpack          # the only new dependency; cv2 and numpy are already there
#   # no websocket-client needed: the framing is hand-rolled, like the MotusV2 client
#
# Files to drop next to this script:
#   motus_client_adapter.py      # protocol (msgpack over a binary WebSocket)
#   g1d_jepa_client.py           # robot loop
#
# Usage on the robot:
#   cd /home/unitree/client/jepa_client && \
#   LEROBOT_ROOT=/home/unitree/unitree_lerobot \
#   UNITREE_DDSINTERFACE=eth0 IMAGE_HOST=192.168.123.164 \
#   bash run_g1d_jepa_client.sh
#
# If the GPU box is not reachable directly, keep a forward up the way the
# MotusV2 client does, then point SERVER_HOST/PORT at the local end:
#   ssh -N -L 127.0.0.1:18000:127.0.0.1:10050 root@<gateway>

set -Eeuo pipefail

SCRIPTS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LEROBOT_ROOT="${LEROBOT_ROOT:-/home/unitree/unitree_lerobot}"

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
cd "$LEROBOT_ROOT"
export PYTHONPATH="$LEROBOT_ROOT":"$SCRIPTS_DIR"
export UNITREE_DDSINTERFACE="${UNITREE_DDSINTERFACE:-eth0}"

# Public endpoint of the policy server (reverse tunnel -> dsw-4:8000).
SERVER_HOST="${SERVER_HOST:-${G1D_GATEWAY_HOST:-127.0.0.1}}"
SERVER_PORT="${SERVER_PORT:-10050}"

# Raw frames are 2.76 MB per observation and want 55 Mbps sustained to replan at
# 2.5 Hz; q95 is ~16x smaller at 3.4 Mbps. Set 0 to send raw.
JPEG_QUALITY="${JPEG_QUALITY:-95}"

# 30Hz is the rate the fine-tune data was recorded at. Do not change it: the
# chunk is 30 absolute joint targets covering exactly one second.
FREQUENCY="${FREQUENCY:-30}"

# Serial by default: predict a chunk, execute all of it, then ask again. The
# arm pauses between chunks, but every command comes from an observation taken
# just before it, which is the behaviour to debug against.
#
# ASYNC_INFERENCE=1 overlaps inference with motion instead, trimming the steps
# that went stale while the request was in flight. It REQUIRES the server on
# --no-action-ensemble; the client reads the handshake and refuses otherwise
# rather than desynchronising quietly.
ASYNC_INFERENCE="${ASYNC_INFERENCE:-0}"

# Internal Dex1 on rt/lowstate motors 31/33 — the USB Dex1 hangs on this G1.
EE="${EE:-dex1_internal}"
ARM="${ARM:-G1_29}"
IMAGE_HOST="${IMAGE_HOST:-192.168.123.164}"

# Optional: ramp both arms to an episode's first frame before the start prompt.
INIT_POSE_JSON="${INIT_POSE_JSON:-}"
INIT_POSE_FRAME="${INIT_POSE_FRAME:-0}"
INIT_POSE_STEPS="${INIT_POSE_STEPS:-100}"

# 1.0 = no EMA. The rate limiter already bounds every step by the server's
# max_step_rad, so extra smoothing mostly adds lag.
SMOOTH_ALPHA="${SMOOTH_ALPHA:-1.0}"

# Per-control-step log of requested vs commanded vs measured joint16. The clamp
# and the rate limit run here, so the server's recording only has "requested" --
# without this one you cannot tell a bad policy from a limiter holding it back.
# Empty disables.
RECORD_LOG="${RECORD_LOG:-logs/commands_$(date +%Y%m%d_%H%M%S).jsonl}"

PYTHON="${PYTHON:-python}"

EXTRA_FLAGS=()
[ "$ASYNC_INFERENCE" = "1" ] && EXTRA_FLAGS+=(--async_inference)
[ -n "$RECORD_LOG" ] && EXTRA_FLAGS+=(--record_log="$RECORD_LOG")
# OpenWAM servers: PROMPT="..." skips the interactive instruction picker (PROMPT="" = server default)
[ -n "${PROMPT+x}" ] && EXTRA_FLAGS+=(--prompt="$PROMPT")
[ -n "$INIT_POSE_JSON" ] && EXTRA_FLAGS+=(
    --init_pose_json="$INIT_POSE_JSON"
    --init_pose_frame="$INIT_POSE_FRAME"
    --init_pose_steps="$INIT_POSE_STEPS"
)

echo "[client] JEPA-WAM G1-Dex1, WebSocket msgpack"
echo "[client] Server:      ws://$SERVER_HOST:$SERVER_PORT"
echo "[client] Image host:  $IMAGE_HOST"
echo "[client] Frames:      $([ "$JPEG_QUALITY" -gt 0 ] && echo "jpeg q$JPEG_QUALITY" || echo raw)"
echo "[client] Async:       $ASYNC_INFERENCE (needs server --no-action-ensemble)"
echo "[client] EE:          $EE (lowstate motors 31/33)"
echo "[client] Command log: ${RECORD_LOG:-off}"
echo "[client] Instruction: baked into the checkpoint, not sent"

exec "$PYTHON" "$SCRIPTS_DIR"/g1d_jepa_client.py \
    --arm="$ARM" \
    --ee="$EE" \
    --frequency="$FREQUENCY" \
    --image_host="$IMAGE_HOST" \
    --server_host="$SERVER_HOST" \
    --server_port="$SERVER_PORT" \
    --jpeg_quality="$JPEG_QUALITY" \
    --smooth_alpha="$SMOOTH_ALPHA" \
    "${EXTRA_FLAGS[@]}"
