#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python -u ./trainduobu1_lowrank_joint_experiment/run_lowrank_joint_multiseed_4gpu.py \
  --project-root "$PROJECT_ROOT" \
  --output-dir "$PROJECT_ROOT/results_joint_encoding_r16_seed16_26_36_bs16" \
  --datasets milan_sms milan_call milan_internet mobile_nj mobile_bj \
  --variants full_v2 lowrank_joint wo_joint \
  --seeds 16 26 36 \
  --gpus 0 1 2 3 \
  --joint-rank 16 \
  --batch-size 16 \
  --epochs 300 \
  --learning-rate 0.001 \
  --loss l1 \
  --branch-aux-weight 0.1 \
  --split-mode chronological \
  --split-gap -1 \
  --milan-loader clean \
  --e-layers 2 \
  --patience 0 \
  --num-workers 0 \
  --log-every 1
