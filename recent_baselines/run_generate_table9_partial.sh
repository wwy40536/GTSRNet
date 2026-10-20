#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   cd /path/to/GTSRNet
#   bash recent_baselines/run_generate_table9_partial.sh

PROJECT_ROOT="${1:-$(pwd)}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RAW_DIR="$PROJECT_ROOT/raw_metrics_for_table9"
OUTPUT_DIR="$PROJECT_ROOT/results_table9_partial"

mkdir -p "$RAW_DIR"

echo "[1/4] Searching top-level ZIP files containing metrics.json ..."
ZIP_WITH_METRICS=0
while IFS= read -r -d '' ZIP_FILE; do
  COUNT=$(unzip -Z1 "$ZIP_FILE" | grep -c 'metrics.json$' || true)
  if [ "$COUNT" -gt 0 ]; then
    ZIP_NAME="$(basename "$ZIP_FILE" .zip)"
    DEST="$RAW_DIR/$ZIP_NAME"
    mkdir -p "$DEST"
    unzip -oq "$ZIP_FILE" -d "$DEST"
    ZIP_WITH_METRICS=$((ZIP_WITH_METRICS + 1))
    echo "  extracted $(basename "$ZIP_FILE"): $COUNT metrics files"
  else
    echo "  skipped $(basename "$ZIP_FILE"): no metrics.json"
  fi
done < <(find "$PROJECT_ROOT" -maxdepth 1 -type f -name '*.zip' -print0)

if [ "$ZIP_WITH_METRICS" -gt 0 ]; then
  SEARCH_ROOT="$RAW_DIR"
else
  SEARCH_ROOT="$PROJECT_ROOT"
fi

echo "[2/4] Checking raw metrics under: $SEARCH_ROOT"
METRIC_COUNT=$(find "$SEARCH_ROOT" -type f -name metrics.json | wc -l)
echo "  found $METRIC_COUNT metrics.json files"
if [ "$METRIC_COUNT" -eq 0 ]; then
  echo "ERROR: No raw metrics.json exists. Summary-only ZIP files cannot produce absolute Step 4/8/12 values." >&2
  exit 2
fi

echo "[3/4] Aggregating all currently available runs ..."
python -u "$SCRIPT_DIR/merge_baseline_results.py" \
  --old-results "$SEARCH_ROOT" \
  --new-results "$SEARCH_ROOT" \
  --old-seeds 26 36 66 76 \
  --new-seeds 23 29 103 128 \
  --gtsrnet-results "$SEARCH_ROOT" \
  --output-dir "$OUTPUT_DIR" \
  --allow-partial \
  2>&1 | tee "$PROJECT_ROOT/table9_partial.log"

TABLE="$OUTPUT_DIR/paper_table_steps_4_8_12_overall.csv"
echo "[4/4] Done"
echo "  Table: $TABLE"
echo "  Seed counts: $OUTPUT_DIR/baseline_summary.csv"

