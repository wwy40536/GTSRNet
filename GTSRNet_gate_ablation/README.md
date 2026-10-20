# Dynamic-gate ablation

This package adds two controlled variants to the existing strict ablation code:

- `fixed_avg`: parameter-free `0.5 / 0.5` fusion.
- `horizon_static`: learns one fixed spatial weight `alpha_p` for each forecast
  horizon.  The weight does not depend on the sample or grid cell.  It retains
  the same horizon-wise branch calibration used by the dynamic gate.
- `GTSRNet`: the complete sample/cell/horizon-dependent dynamic gate, trained
  with the configured branch auxiliary loss.
- `GTSRNet_no_aux`: architecture-identical to `GTSRNet`, but its effective branch
  auxiliary-loss weight is forced to zero.

`alpha` denotes the calibrated spatial-branch weight.  `1-alpha` is the
temporal-branch weight.

## 1. Upload

Upload the complete `GTSRNet_gate_ablation` directory into the project root:

```text
/path/to/GTSRNet/
```

## 2. Smoke test

Run from the project root:

```bash
conda activate glsttn
python -u ./GTSRNet_gate_ablation/train_ablation.py \
  --project-root "$PWD" \
  --smoke-test \
  --device cuda \
  --gpu 0
```

The output must include `PASS horizon_static` and `PASS GTSRNet_no_aux`.

## 3. Train the missing controls

The following command deliberately reuses the four existing result roots.
Completed `fixed_avg` and `GTSRNet` tasks are skipped automatically; only the
new `horizon_static` and `GTSRNet_no_aux` directories are trained.  Do not add
`--force`.

```bash
for seed in 23 29 103 128; do
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  python -u ./GTSRNet_gate_ablation/run_ablation.py \
    --project-root "$PWD" \
    --output-dir "$PWD/results_ablation_strict_seed${seed}_bs16" \
    --datasets milan_sms milan_call milan_internet \
    --variants fixed_avg horizon_static GTSRNet_no_aux GTSRNet \
    --gpus 0 1 2 3 \
    --batch-size 16 \
    --epochs 300 \
    --learning-rate 0.001 \
    --loss l1 \
    --branch-aux-weight 0.1 \
    --e-layers 2 \
    --seg-len 6 \
    --seed "$seed" \
    --split-mode chronological \
    --split-gap -1 \
    --milan-loader clean \
    --patience 0 \
    --no-plot || exit 1
done
```

This adds 24 new training tasks: 4 seeds x 3 datasets x 2 new variants.

## 4. Check completion

```bash
python - <<'PY'
from pathlib import Path

seeds = (23, 29, 103, 128)
datasets = ("milan_sms", "milan_call", "milan_internet")
variants = ("fixed_avg", "horizon_static", "GTSRNet_no_aux", "GTSRNet")
missing = []
for seed in seeds:
    root = Path(f"results_ablation_strict_seed{seed}_bs16")
    for dataset in datasets:
        for variant in variants:
            path = root / dataset / variant / "metrics.json"
            if not path.is_file():
                missing.append(str(path))
print(f"complete={len(seeds)*len(datasets)*len(variants)-len(missing)}/48")
for path in missing:
    print("MISSING", path)
PY
```

## 5. Export performance and gate-weight distributions

Run this only after all training processes have stopped:

```bash
python -u ./GTSRNet_gate_ablation/analyze_gate_weights.py \
  --project-root "$PWD" \
  --seeds 23 29 103 128 \
  --datasets milan_sms milan_call milan_internet \
  --variants fixed_avg horizon_static GTSRNet_no_aux GTSRNet \
  --output-dir "$PWD/results_gate_ablation_seed23_29_103_128" \
  --gpu 0 \
  --dpi 300
```

Outputs include:

- `fusion_performance_mean_std.csv`: mean and sample standard deviation of RMSE/MAE.
- `fig_gate_ablation_rmse.png/pdf` and `fig_gate_ablation_mae.png/pdf`.
- `gate_alpha_by_horizon.csv`: exact mean/variance plus sampled quantiles for each horizon.
- `gate_alpha_by_seed.csv`: per-seed alpha statistics.
- `fig_gate_alpha_horizons_<dataset>.png/pdf`: mean alpha and interquartile bands.
- `fig_gate_alpha_distribution_<dataset>.png/pdf`: distributions at horizons 1, 7 and 12.

For a valid dynamic-gate claim, report both prediction metrics and gate
statistics.  A dynamic gate is supported when it improves over
`horizon_static` and its `sample_mean_std`/distribution is non-zero.  Compare
`GTSRNet` with `GTSRNet_no_aux` to assess whether the auxiliary loss reduces
persistent collapse toward one branch.  Do not claim a benefit before these
results are observed.
