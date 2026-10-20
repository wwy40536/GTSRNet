#!/usr/bin/env bash
set -euo pipefail

# Run from any directory. The first argument is the GTSRNet project root.
PROJECT_ROOT="${1:?Usage: bash run_all.sh PROJECT_ROOT [RESULTS_ROOT] [OUTPUT_DIR] [GPU]}"
RESULTS_ROOT="${2:-results_ablation_strict_seed128_bs16}"
OUTPUT_DIR="${3:-results_branch_reliability_seed128}"
GPU="${4:-0}"

python "$(dirname "$0")/run_branch_reliability.py" \
  --project-root "$PROJECT_ROOT" \
  --results-root "$RESULTS_ROOT" \
  --output-dir "$OUTPUT_DIR" \
  --datasets milan_call milan_sms milan_internet \
  --device cuda \
  --gpu "$GPU" \
  --save-predictions
