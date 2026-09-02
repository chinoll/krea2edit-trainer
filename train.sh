#!/usr/bin/env bash
set -euo pipefail

cd "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source /data/ai-toolkit/venv/bin/activate

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

exec accelerate launch \
  --multi_gpu \
  --num_processes 8 \
  --num_machines 1 \
  --mixed_precision bf16 \
  --dynamo_backend no \
  train.py \
  --config configs/train_8gpu_muon.yaml
