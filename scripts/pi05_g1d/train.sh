#!/bin/bash
# pi0.5 on the G1-D sim pour-beans data with the vendored openpi's PyTorch trainer (DDP).
#   bash train.sh <work dir> <run name> <gpu ids, e.g. 1,2,3,4,5,6,7> [extra openpi train args]
# Multi-node: set NNODES, NODE_RANK and MASTER_ADDR (an address on the RDMA network, e.g. the master's eth0) on every
# node; --batch-size stays global. Checkpoints are written by the master node only.
# <work dir> holds openpi/ (uv .venv with models_pytorch/transformers_replace copied into transformers),
# lerobot/<repo id> from prepare_data.py and pi05_base_pytorch/model.safetensors (openpi's JAX->PyTorch pi05_base),
# and receives checkpoints/ and openpi/assets/ (norm stats, computed on first run).
set -euo pipefail
work=$(realpath "$1") run=$2 gpus=$3
shift 3
here=$(cd "$(dirname "$0")" && pwd)
export CUDA_VISIBLE_DEVICES=$gpus
export HF_LEROBOT_HOME=$work/lerobot OPENPI_DATA_HOME=$work/.cache/openpi HF_DATASETS_CACHE=$work/.cache/hf_datasets
export WANDB_MODE=offline TOKENIZERS_PARALLELISM=false
cd "$work/openpi"
py=.venv/bin/python
if [[ ! -f assets/pi05_g1d_sz_pour_beans/g1d_sz_pour_beans_cup_or_mug_500/norm_stats.json ]]; then
  $py "$here/openpi_g1d.py" scripts/compute_norm_stats.py --config-name pi05_g1d_sz_pour_beans
fi
n=$(awk -F, '{print NF}' <<<"$gpus")
if [[ -n ${MASTER_ADDR:-} ]]; then
  export NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME:-eth0} GLOO_SOCKET_IFNAME=${GLOO_SOCKET_IFNAME:-eth0}
  dist=(--nnodes="$NNODES" --node_rank="$NODE_RANK" --master_addr="$MASTER_ADDR" --master_port="${MASTER_PORT:-29500}")
else
  dist=(--standalone --nnodes=1)
fi
exec .venv/bin/torchrun "${dist[@]}" --nproc_per_node="$n" "$here/openpi_g1d.py" scripts/train_pytorch.py \
  pi05_g1d_sz_pour_beans --exp-name "$run" --checkpoint-base-dir "$work/checkpoints" \
  --pytorch-weight-path "$work/pi05_base_pytorch" "$@"
