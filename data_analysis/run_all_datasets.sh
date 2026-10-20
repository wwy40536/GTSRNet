#!/usr/bin/env bash
set -euo pipefail

python -u data_analysis/analyze_dataset_interactions.py \
  --project-root "$(pwd)" \
  --datasets milan_sms milan_call milan_internet mobile_bj mobile_nj \
  --output-dir results_dataset_interactions \
  --top-nodes 24 \
  --max-lag 6 \
  --granger-lags 6 \
  --permutations 500 \
  --alpha 0.05 \
  --seed 2026 \
  --max-network-edges 30 \
  --dpi 300
