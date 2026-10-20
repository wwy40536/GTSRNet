#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

python -u ./GTSRNet_lowrank_ablation/train_lowrank_joint.py \
  --project-root "$PROJECT_ROOT" \
  --device cuda \
  --gpu 0 \
  --joint-rank 16 \
  --smoke-test
