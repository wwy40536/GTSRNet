#!/usr/bin/env bash
set -euo pipefail

# Run this file from the original project root after placing this folder there.
PYTHON_BIN="${PYTHON_BIN:-python}"

"${PYTHON_BIN}" -u SGASeq_baseline/run_sgaseq.py \
  --project-root "$(pwd)" \
  --python "$(command -v "${PYTHON_BIN}")" \
  --gpus 0 1 2 3 \
  --datasets milan_sms milan_call milan_internet mobile_bj mobile_nj \
  --seeds 23 26 29 36 46 56 66 76 103 128 \
  --close-size 24 \
  --predict-size 12 \
  --batch-size 16 \
  --epochs 300 \
  --learning-rate 0.001 \
  --loss l1 \
  --split-mode chronological \
  --split-gap -1 \
  --milan-loader clean \
  --save-predictions

