"""Diagnose why per-step RMSE/MAE is not monotonic in the forecasting step.

Uses exactly the same loader/split/normalization as train_decoder_experiment.py
and reports, per forecasting step:

  1. persistence     : y_hat(t+h) = x(t)            (the model's own residual base)
  2. seasonal_naive  : y_hat(t+h) = x(t+h-24)       (needs input_len >= 24, h <= 24)
  3. truth_std       : spread of the evaluated targets at that step
  4. PMF / RMF       : trained checkpoints (optional, --results-dir)

each in two evaluation modes:

  * window   : current protocol, step h averaged over all test windows
               (different steps see DIFFERENT target timestamps)
  * aligned  : step h averaged only over target timestamps shared by all steps
               (removes the "which hours are in the test set" confound)

If the naive baselines already show the same non-monotonic shape, the shape is
a property of the data/test window, not a training bug.

Example (run from project root):
  python reviewer_response_experiments_v3/diagnose_horizon_curve.py \
      --project-root "$PWD" --dataset mobile_bj --horizon 24 \
      --results-dir "$PWD/results_decoder_reviewer" --seeds 45 46 47 48 49
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent


# ----------------------------------------------------------------- pure numpy
def per_step(pred: np.ndarray, truth: np.ndarray):
    err = pred.astype(np.float64) - truth.astype(np.float64)
    axes = (0,) + tuple(range(2, err.ndim))
    return np.sqrt(np.mean(err ** 2, axis=axes)), np.mean(np.abs(err), axis=axes)


def aligned_per_step(pred: np.ndarray, truth: np.ndarray):
    """Step h uses window i = k + (H - h), k = 0..n-H, so every step is
    evaluated on the same target timestamps t = test_start + Tin + H - 1 + k."""
    n, horizon = pred.shape[:2]
    common = n - horizon + 1
    if common < 1:
        return None, None, 0
    rmse, mae = np.empty(horizon), np.empty(horizon)
    for h in range(1, horizon + 1):
        offset = horizon - h
        e = pred[offset:offset + common, h - 1].astype(np.float64) - truth[offset:offset + common, h - 1]
        rmse[h - 1] = np.sqrt(np.mean(e ** 2))
        mae[h - 1] = np.mean(np.abs(e))
    return rmse, mae, common


def lag_autocorr(series: np.ndarray, lag: int) -> float:
    if len(series) <= lag + 1:
        return float("nan")
    a, b = series[:-lag] - series.mean(), series[lag:] - series.mean()
    den = np.sqrt(np.sum(a ** 2) * np.sum(b ** 2))
    return float(np.sum(a * b) / den) if den > 1e-12 else float("nan")


def orientation_check(path: Path) -> None:
    raw = np.load(path, allow_pickle=False)
    raw = raw[..., 0] if raw.ndim == 3 and raw.shape[-1] == 1 else raw
    print(f"\n[orientation] {path.name}: shape={raw.shape}")
    for axis in (0, 1):
        s = raw.mean(axis=1 - axis)  # average over the other axis
        print(f"  treat axis {axis} (len={raw.shape[axis]}) as TIME: "
              f"lag-1 ac={lag_autocorr(s, 1):.3f}, lag-24 ac={lag_autocorr(s, 24):.3f}, "
              f"lag-12 ac={lag_autocorr(s, 12):.3f}")
    print("  The real time axis should show high lag-1 and clearly positive lag-24 "
          "(daily cycle) with low/negative lag-12.")


# ----------------------------------------------------------------- main
def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--project-root", type=Path, default=SCRIPT_DIR.parent)
    p.add_argument("--dataset", required=True,
                   choices=("milan_sms", "milan_call", "milan_internet", "mobile_bj", "mobile_nj"))
    p.add_argument("--horizon", type=int, default=24)
    p.add_argument("--input-len", type=int, default=24)
    p.add_argument("--seg-len", type=int, default=6)
    p.add_argument("--d-model", type=int, default=128)
    p.add_argument("--results-dir", type=Path, default=None)
    p.add_argument("--baseline", choices=("last", "seasonal"), default="last",
                   help="Must match how the checkpoints in --results-dir were trained.")
    p.add_argument("--seasonal-period", type=int, default=24)
    p.add_argument("--seeds", nargs="+", type=int, default=(45, 46, 47, 48, 49))
    p.add_argument("--output-prefix", type=Path, default=None)
    a = p.parse_args()

    root = a.project_root.expanduser().resolve()
    for path in (SCRIPT_DIR, root):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))
    import torch
    from train_decoder_experiment import TemporalOnlyModel, build_parser
    from train_embedding_ablation import build_loaders

    args = build_parser().parse_args([
        "--project-root", str(root), "--decoder", "pmf", "--dataset", a.dataset,
        "--input-len", str(a.input_len), "--horizon", str(a.horizon),
        "--seg-len", str(a.seg_len), "--device", "cpu",
    ])
    device = torch.device("cpu")
    _tr, _va, test_loader, norm, height, width = build_loaders(args, device)
    subset = test_loader.dataset
    idx = np.asarray(subset.indices)
    x = norm.inverse_transform(np.asarray(subset.dataset.x[idx], dtype=np.float64))
    y = norm.inverse_transform(np.asarray(subset.dataset.y[idx], dtype=np.float64))
    n, H = y.shape[:2]
    print(f"\ntest windows={n}, horizon={H}, distinct target timestamps={n + H - 1}, "
          f"timestamps shared by all steps={max(0, n - H + 1)}")

    if a.dataset.startswith("mobile_"):
        from train_embedding_ablation import mobile_dataset_defaults
        orientation_check(root / mobile_dataset_defaults(a.dataset)[0])

    curves: Dict[str, np.ndarray] = {}
    preds = {"persistence": np.repeat(x[:, -1:], H, axis=1)}
    if a.input_len >= 24 and H <= 24:
        preds["seasonal_naive"] = x[:, a.input_len - 24:a.input_len - 24 + H]

    if a.results_dir is not None:
        for mode in ("pmf", "rmf"):
            outs = []
            for seed in a.seeds:
                ckpt = a.results_dir / a.dataset / mode / f"H{H}" / f"seed_{seed}" / "best.pt"
                if not ckpt.is_file():
                    print(f"  skip missing {ckpt}")
                    continue
                model = TemporalOnlyModel(a.input_len, H, height, width, a.d_model, a.seg_len, 0.0, mode,
                                          baseline_mode=a.baseline, seasonal_period=a.seasonal_period)
                state = torch.load(ckpt, map_location="cpu")
                model.load_state_dict(state["model_state"]); model.eval()
                chunks = []
                with torch.no_grad():
                    for xb, _c, _t in test_loader:
                        chunks.append(model(xb.float()).numpy())
                outs.append(norm.inverse_transform(np.concatenate(chunks).astype(np.float64)))
            if outs:
                preds[mode] = np.mean(outs, axis=0)  # seed-averaged prediction, for shape only

    axes = (0,) + tuple(range(2, y.ndim))
    curves["truth_std"] = y.std(axis=axes)
    for name, pr in preds.items():
        r, m = per_step(pr, y)
        curves[f"{name}_rmse_window"], curves[f"{name}_mae_window"] = r, m
        ra, ma, common = aligned_per_step(pr, y)
        if ra is not None:
            curves[f"{name}_rmse_aligned"], curves[f"{name}_mae_aligned"] = ra, ma

    default_dir = a.results_dir if a.results_dir is not None else root / "results_decoder_reviewer"
    prefix = a.output_prefix or (default_dir / "diagnostics" / f"{a.dataset}_H{H}")
    prefix.parent.mkdir(parents=True, exist_ok=True)
    with open(str(prefix) + ".csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        keys = list(curves)
        w.writerow(["step"] + keys)
        for h in range(H):
            w.writerow([h + 1] + [f"{curves[k][h]:.6g}" for k in keys])
    print(f"WROTE {prefix}.csv")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.3), constrained_layout=True)
    steps = np.arange(1, H + 1)
    for name in preds:
        ax[0].plot(steps, curves[f"{name}_rmse_window"], marker=".", label=name)
        if f"{name}_rmse_aligned" in curves:
            ax[1].plot(steps, curves[f"{name}_rmse_aligned"], marker=".", label=name)
    ax[2].plot(steps, curves["truth_std"], marker=".", color="k")
    ax[0].set_title("RMSE, current protocol (per-window)")
    ax[1].set_title(f"RMSE, target-time aligned ({max(0, n - H + 1)} timestamps)")
    ax[2].set_title("Std of evaluated targets per step")
    for axis in ax:
        axis.set_xlabel("Forecasting step"); axis.grid(alpha=0.25)
    ax[0].legend(); ax[1].legend()
    fig.suptitle(f"{a.dataset}, H={H}, models trained with {a.baseline} baseline")
    fig.savefig(str(prefix) + ".png", dpi=200)
    print(f"WROTE {prefix}.png")


if __name__ == "__main__":
    main()
