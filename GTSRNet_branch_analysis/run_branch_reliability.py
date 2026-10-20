#!/usr/bin/env python3
"""Evaluate branch reliability from trained GTSRNet ``GTSRNet`` checkpoints.

This script does not train or modify the model.  It rebuilds the original test
split from the checkpoint configuration, extracts the calibrated spatial and
temporal predictions immediately before gated fusion, reconstructs the gate
weights, and writes paper-ready metrics and figures.
"""

from __future__ import annotations

import argparse
import csv
import importlib
import json
import math
import sys
import time
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple


def build_cli() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Extract pre-fusion branch predictions and evaluate whether the "
            "dynamic gate follows the more reliable branch."
        )
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        required=True,
        help="Root of GLSTTN_tars-duobunewdata on the server.",
    )
    parser.add_argument(
        "--results-root",
        type=Path,
        required=True,
        help=(
            "Completed ablation directory, for example "
            "results_ablation_strict_seed128_bs16."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results_branch_reliability"),
        help="Output directory. Relative paths are resolved under project root.",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=("milan_call", "milan_sms", "milan_internet"),
        choices=("milan_call", "milan_sms", "milan_internet", "mobile_bj", "mobile_nj"),
    )
    parser.add_argument("--variant", default="GTSRNet", choices=("GTSRNet",))
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument(
        "--batch-size",
        type=int,
        default=0,
        help="0 keeps the checkpoint batch size; set a smaller value if GPU memory is tight.",
    )
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument(
        "--milan-data",
        type=Path,
        default=None,
        help="Optional override for dataset/data_git_version.h5.",
    )
    parser.add_argument(
        "--feature-path",
        type=Path,
        default=None,
        help="Optional override for dataset/crawled_feature.csv (legacy loader only).",
    )
    parser.add_argument(
        "--save-predictions",
        action="store_true",
        help="Also save truth, both calibrated branches, fused output and alpha to NPZ.",
    )
    return parser


def resolve_from_project(project_root: Path, value: Path) -> Path:
    value = value.expanduser()
    return value.resolve() if value.is_absolute() else (project_root / value).resolve()


def import_training_module(project_root: Path):
    ablation_dir = project_root / "GTSRNet_model"
    train_file = ablation_dir / "train_ablation.py"
    model_file = ablation_dir / "gtsrnet_model.py"
    if not train_file.is_file() or not model_file.is_file():
        raise FileNotFoundError(
            "Expected the original ablation code under "
            f"{ablation_dir}, but the training/model files were not found."
        )
    for path in (project_root, ablation_dir):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))
    module = importlib.import_module("train_ablation")
    module.PROJECT_ROOT = project_root
    return module


def find_checkpoint(results_root: Path, dataset: str, variant: str) -> Path:
    direct = results_root / dataset / variant / "best.pt"
    if direct.is_file():
        return direct
    matches = []
    for path in results_root.rglob("best.pt"):
        parts = path.parts
        if len(parts) >= 3 and parts[-3] == dataset and parts[-2] == variant:
            matches.append(path)
    if not matches:
        raise FileNotFoundError(
            f"No checkpoint matching */{dataset}/{variant}/best.pt under {results_root}"
        )
    if len(matches) > 1:
        listing = "\n".join(f"  - {path}" for path in matches)
        raise RuntimeError(
            f"Multiple checkpoints found for {dataset}. Pass a narrower --results-root:\n{listing}"
        )
    return matches[0]


def torch_load_checkpoint(torch, checkpoint_path: Path, device):
    try:
        return torch.load(checkpoint_path, map_location=device, weights_only=True)
    except TypeError:
        return torch.load(checkpoint_path, map_location=device)


def namespace_from_checkpoint(
    train_module,
    checkpoint: Mapping[str, object],
    cli: argparse.Namespace,
    dataset: str,
) -> argparse.Namespace:
    defaults = vars(train_module.build_parser().parse_args([])).copy()
    saved = checkpoint.get("args", {})
    if not isinstance(saved, Mapping):
        raise TypeError("Checkpoint field 'args' is missing or is not a mapping")
    defaults.update(saved)

    for key in ("project_root", "data", "feature_path", "output_dir"):
        if defaults.get(key) not in (None, "None", ""):
            defaults[key] = Path(defaults[key])

    args = argparse.Namespace(**defaults)
    args.project_root = cli.project_root
    args.dataset = dataset
    args.variant = cli.variant
    args.device = cli.device
    args.gpu = cli.gpu
    args.num_workers = cli.num_workers
    args.eval_only = True
    if cli.batch_size > 0:
        args.batch_size = cli.batch_size

    # Prefer explicit server-side overrides. If a checkpoint contains a stale
    # absolute path, fall back to the normal path relative to project root.
    if dataset.startswith("milan_"):
        if cli.milan_data is not None:
            args.data = resolve_from_project(cli.project_root, cli.milan_data)
        elif args.data is not None:
            candidate = resolve_from_project(cli.project_root, args.data)
            args.data = candidate if candidate.is_file() else None
        if cli.feature_path is not None:
            args.feature_path = resolve_from_project(cli.project_root, cli.feature_path)
        elif args.feature_path is not None:
            candidate = resolve_from_project(cli.project_root, args.feature_path)
            args.feature_path = candidate if candidate.is_file() else Path("dataset/crawled_feature.csv")
    return args


def calibrated_components(torch, model, x, phase):
    """Return fused, calibrated spatial/temporal branches, and spatial alpha.

    ``GTSRNet(return_branches=True)`` exposes the raw branch
    predictions.  For ``GTSRNet`` the horizon calibration is inside the fusion
    module, so it is reproduced here before computing reliability.
    """
    fused, branches = model(x, phase, return_branches=True)
    spatial_raw = branches.get("spatial")
    temporal_raw = branches.get("temporal")
    if spatial_raw is None or temporal_raw is None:
        raise RuntimeError("The selected checkpoint does not contain both branches")

    batch = x.shape[0]
    horizons = model.t_out
    cells = model.n_cells
    spatial_raw = spatial_raw.reshape(batch, horizons, cells)
    temporal_raw = temporal_raw.reshape(batch, horizons, cells)
    fusion = model.fusion

    required = (
        "spatial_scale",
        "spatial_bias",
        "temporal_scale",
        "temporal_bias",
        "horizon_embedding",
        "gate",
    )
    missing = [name for name in required if not hasattr(fusion, name)]
    if missing:
        raise TypeError(
            "This experiment requires the GTSRNet horizon-calibrated gate; "
            f"fusion is missing {missing}"
        )

    spatial = (
        spatial_raw * fusion.spatial_scale.view(1, horizons, 1)
        + fusion.spatial_bias.view(1, horizons, 1)
    )
    temporal = (
        temporal_raw * fusion.temporal_scale.view(1, horizons, 1)
        + fusion.temporal_bias.view(1, horizons, 1)
    )
    horizon_ids = torch.arange(horizons, device=x.device)
    horizon_features = fusion.horizon_embedding(horizon_ids).view(1, horizons, 1, -1)
    horizon_features = horizon_features.expand(batch, -1, cells, -1)
    gate_input = torch.cat(
        (torch.stack((spatial, temporal), dim=-1), horizon_features), dim=-1
    )
    alpha = torch.sigmoid(fusion.gate(gate_input)).squeeze(-1)
    reconstructed = alpha * spatial + (1.0 - alpha) * temporal
    fused_flat = fused.reshape(batch, horizons, cells)
    if not torch.allclose(reconstructed, fused_flat, rtol=1e-5, atol=1e-6):
        maximum_difference = float((reconstructed - fused_flat).abs().max().item())
        raise RuntimeError(
            "Reconstructed fusion does not match model output; "
            f"max difference={maximum_difference:.6g}. Check model-code compatibility."
        )
    shape = (batch, horizons, 1, model.height, model.width)
    return (
        fused,
        spatial.reshape(shape),
        temporal.reshape(shape),
        alpha.reshape(batch, horizons, model.height, model.width),
    )


def safe_corr(x, y, np) -> float:
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    finite = np.isfinite(x) & np.isfinite(y)
    x = x[finite]
    y = y[finite]
    if x.size < 2 or np.std(x) == 0.0 or np.std(y) == 0.0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def scalar_metrics(pred, truth, np) -> Dict[str, float]:
    error = pred.astype(np.float64) - truth.astype(np.float64)
    return {
        "mae": float(np.mean(np.abs(error))),
        "rmse": float(np.sqrt(np.mean(np.square(error)))),
    }


def reliability_row(horizon: int, spatial, temporal, fused, truth, alpha, np):
    spatial_abs = np.abs(spatial.astype(np.float64) - truth.astype(np.float64))
    temporal_abs = np.abs(temporal.astype(np.float64) - truth.astype(np.float64))
    fused_abs = np.abs(fused.astype(np.float64) - truth.astype(np.float64))
    tolerance = 1e-12
    spatial_wins = spatial_abs + tolerance < temporal_abs
    temporal_wins = temporal_abs + tolerance < spatial_abs
    ties = ~(spatial_wins | temporal_wins)
    non_ties = spatial_wins | temporal_wins
    choose_spatial = alpha >= 0.5
    correct = (choose_spatial & spatial_wins) | ((~choose_spatial) & temporal_wins)
    gate_accuracy = float(np.mean(correct[non_ties])) if np.any(non_ties) else float("nan")
    advantage = temporal_abs - spatial_abs  # positive means spatial is more reliable

    spatial_metrics = scalar_metrics(spatial, truth, np)
    temporal_metrics = scalar_metrics(temporal, truth, np)
    fused_metrics = scalar_metrics(fused, truth, np)
    oracle_mae = float(np.mean(np.minimum(spatial_abs, temporal_abs)))
    alpha_spatial_wins = (
        float(np.mean(alpha[spatial_wins])) if np.any(spatial_wins) else float("nan")
    )
    alpha_temporal_wins = (
        float(np.mean(alpha[temporal_wins])) if np.any(temporal_wins) else float("nan")
    )
    return {
        "horizon": horizon,
        "spatial_mae": spatial_metrics["mae"],
        "spatial_rmse": spatial_metrics["rmse"],
        "temporal_mae": temporal_metrics["mae"],
        "temporal_rmse": temporal_metrics["rmse"],
        "fused_mae": fused_metrics["mae"],
        "fused_rmse": fused_metrics["rmse"],
        "oracle_mae": oracle_mae,
        "fused_oracle_mae_gap": fused_metrics["mae"] - oracle_mae,
        "spatial_win_rate": float(np.mean(spatial_wins)),
        "temporal_win_rate": float(np.mean(temporal_wins)),
        "tie_rate": float(np.mean(ties)),
        "mean_alpha_spatial": float(np.mean(alpha)),
        "std_alpha_spatial": float(np.std(alpha)),
        "alpha_q10": float(np.quantile(alpha, 0.10)),
        "alpha_q50": float(np.quantile(alpha, 0.50)),
        "alpha_q90": float(np.quantile(alpha, 0.90)),
        "gate_winner_accuracy": gate_accuracy,
        "alpha_advantage_correlation": safe_corr(alpha, advantage, np),
        "mean_alpha_when_spatial_wins": alpha_spatial_wins,
        "mean_alpha_when_temporal_wins": alpha_temporal_wins,
        "alpha_separation": alpha_spatial_wins - alpha_temporal_wins,
    }


def write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def plot_dataset(dataset: str, rows: Sequence[Mapping[str, float]], output_dir: Path) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("WARNING: matplotlib is unavailable; CSV/JSON outputs were still written.")
        return

    horizons = [int(row["horizon"]) for row in rows]
    colors = {"spatial": "#1f77b4", "temporal": "#ff7f0e", "fused": "#d62728"}
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.0), constrained_layout=True)
    for key, label in (("spatial", "Spatial branch"), ("temporal", "Temporal branch"), ("fused", "Gated fusion")):
        axes[0].plot(horizons, [row[f"{key}_mae"] for row in rows], marker="o", label=label, color=colors[key])
        axes[1].plot(horizons, [row[f"{key}_rmse"] for row in rows], marker="o", label=label, color=colors[key])
    for axis, metric in zip(axes, ("MAE", "RMSE")):
        axis.set_xlabel("Forecast horizon")
        axis.set_ylabel(metric)
        axis.grid(True, alpha=0.25)
        axis.set_xticks(horizons)
    axes[0].legend(frameon=False)
    fig.suptitle(dataset.replace("_", " ").title() + " branch errors")
    fig.savefig(output_dir / "branch_error_by_horizon.png", dpi=300)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.0), constrained_layout=True)
    axes[0].plot(horizons, [row["spatial_win_rate"] for row in rows], marker="o", label="Spatial win rate", color=colors["spatial"])
    axes[0].plot(horizons, [row["temporal_win_rate"] for row in rows], marker="o", label="Temporal win rate", color=colors["temporal"])
    axes[0].plot(horizons, [row["mean_alpha_spatial"] for row in rows], marker="s", linestyle="--", label="Mean spatial gate weight", color=colors["fused"])
    axes[0].axhline(0.5, color="0.5", linestyle=":", linewidth=1)
    axes[0].set_ylabel("Proportion / weight")
    axes[0].set_ylim(0.0, 1.0)
    axes[0].legend(frameon=False)

    axes[1].plot(horizons, [row["gate_winner_accuracy"] for row in rows], marker="o", label="Gate winner accuracy", color="#2ca02c")
    axes[1].plot(horizons, [row["alpha_advantage_correlation"] for row in rows], marker="s", label="Alpha-advantage correlation", color="#9467bd")
    axes[1].axhline(0.5, color="0.5", linestyle=":", linewidth=1)
    axes[1].axhline(0.0, color="0.75", linestyle=":", linewidth=1)
    axes[1].set_ylabel("Accuracy / correlation")
    axes[1].set_ylim(-1.0, 1.0)
    axes[1].legend(frameon=False)
    for axis in axes:
        axis.set_xlabel("Forecast horizon")
        axis.set_xticks(horizons)
        axis.grid(True, alpha=0.25)
    fig.suptitle(dataset.replace("_", " ").title() + " branch reliability and gate alignment")
    fig.savefig(output_dir / "branch_reliability_by_horizon.png", dpi=300)
    plt.close(fig)


def plot_combined(all_rows: Mapping[str, Sequence[Mapping[str, float]]], output_dir: Path) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    datasets = list(all_rows)
    colors = {"spatial": "#1f77b4", "temporal": "#ff7f0e", "fused": "#d62728"}
    fig, axes = plt.subplots(len(datasets), 2, figsize=(10.5, 3.2 * len(datasets)), squeeze=False, constrained_layout=True)
    for row_index, dataset in enumerate(datasets):
        rows = all_rows[dataset]
        horizons = [int(row["horizon"]) for row in rows]
        for col, metric in enumerate(("mae", "rmse")):
            axis = axes[row_index, col]
            for key, label in (("spatial", "Spatial"), ("temporal", "Temporal"), ("fused", "Fusion")):
                axis.plot(horizons, [row[f"{key}_{metric}"] for row in rows], marker="o", markersize=3, label=label, color=colors[key])
            axis.set_title(dataset.replace("_", " ").title() + f" - {metric.upper()}")
            axis.set_xlabel("Forecast horizon")
            axis.set_ylabel(metric.upper())
            axis.set_xticks(horizons)
            axis.grid(True, alpha=0.25)
            if row_index == 0 and col == 0:
                axis.legend(frameon=False)
    fig.savefig(output_dir / "fig_branch_error_all_datasets.png", dpi=300)
    plt.close(fig)

    fig, axes = plt.subplots(len(datasets), 2, figsize=(10.5, 3.2 * len(datasets)), squeeze=False, constrained_layout=True)
    for row_index, dataset in enumerate(datasets):
        rows = all_rows[dataset]
        horizons = [int(row["horizon"]) for row in rows]
        left = axes[row_index, 0]
        left.plot(horizons, [row["spatial_win_rate"] for row in rows], marker="o", label="Spatial win rate", color=colors["spatial"])
        left.plot(horizons, [row["temporal_win_rate"] for row in rows], marker="o", label="Temporal win rate", color=colors["temporal"])
        left.plot(horizons, [row["mean_alpha_spatial"] for row in rows], marker="s", linestyle="--", label="Mean spatial weight", color=colors["fused"])
        left.axhline(0.5, color="0.5", linestyle=":", linewidth=1)
        left.set_ylim(0.0, 1.0)
        left.set_title(dataset.replace("_", " ").title() + " - reliability")
        left.set_ylabel("Proportion / weight")
        if row_index == 0:
            left.legend(frameon=False)

        right = axes[row_index, 1]
        right.plot(horizons, [row["gate_winner_accuracy"] for row in rows], marker="o", label="Winner accuracy", color="#2ca02c")
        right.plot(horizons, [row["alpha_advantage_correlation"] for row in rows], marker="s", label="Alpha-advantage corr.", color="#9467bd")
        right.axhline(0.5, color="0.5", linestyle=":", linewidth=1)
        right.axhline(0.0, color="0.75", linestyle=":", linewidth=1)
        right.set_ylim(-1.0, 1.0)
        right.set_title(dataset.replace("_", " ").title() + " - gate alignment")
        right.set_ylabel("Accuracy / correlation")
        if row_index == 0:
            right.legend(frameon=False)
        for axis in (left, right):
            axis.set_xlabel("Forecast horizon")
            axis.set_xticks(horizons)
            axis.grid(True, alpha=0.25)
    fig.savefig(output_dir / "fig_branch_reliability_all_datasets.png", dpi=300)
    plt.close(fig)


def evaluate_dataset(cli, train_module, torch, np, dataset: str, output_root: Path):
    checkpoint_path = find_checkpoint(cli.results_root, dataset, cli.variant)
    device = train_module.choose_device(cli.device, cli.gpu)
    checkpoint = torch_load_checkpoint(torch, checkpoint_path, device)
    checkpoint_variant = checkpoint.get("variant", cli.variant)
    if checkpoint_variant != cli.variant:
        raise ValueError(
            f"Checkpoint {checkpoint_path} is variant={checkpoint_variant!r}, not {cli.variant!r}"
        )
    args = namespace_from_checkpoint(train_module, checkpoint, cli, dataset)
    train_module.set_seed(int(args.seed))
    _train_loader, _val_loader, test_loader, normalizer, height, width = train_module.build_loaders(args, device)
    model = train_module.make_model(args, cli.variant, height, width).to(device)
    model.load_state_dict(checkpoint["model_state"], strict=True)
    model.eval()

    fused_batches = []
    spatial_batches = []
    temporal_batches = []
    alpha_batches = []
    truth_batches = []
    started = time.perf_counter()
    with torch.no_grad():
        for x, phase, target in test_loader:
            x = x.to(device=device, dtype=torch.float32, non_blocking=True)
            phase = phase.to(device=device, dtype=torch.long, non_blocking=True)
            fused, spatial, temporal, alpha = calibrated_components(torch, model, x, phase)
            fused_batches.append(fused.cpu().numpy())
            spatial_batches.append(spatial.cpu().numpy())
            temporal_batches.append(temporal.cpu().numpy())
            alpha_batches.append(alpha.cpu().numpy())
            truth_batches.append(np.asarray(target))
    elapsed = time.perf_counter() - started

    fused_norm = np.concatenate(fused_batches, axis=0)
    spatial_norm = np.concatenate(spatial_batches, axis=0)
    temporal_norm = np.concatenate(temporal_batches, axis=0)
    alpha = np.concatenate(alpha_batches, axis=0)
    truth_norm = np.concatenate(truth_batches, axis=0)
    # Remove the singleton flow/channel dimension so every prediction and
    # alpha share [samples, horizons, height, width].
    fused = normalizer.inverse_transform(fused_norm)[:, :, 0]
    spatial = normalizer.inverse_transform(spatial_norm)[:, :, 0]
    temporal = normalizer.inverse_transform(temporal_norm)[:, :, 0]
    truth = normalizer.inverse_transform(truth_norm)[:, :, 0]

    rows = []
    for horizon in range(fused.shape[1]):
        rows.append(
            reliability_row(
                horizon + 1,
                spatial[:, horizon],
                temporal[:, horizon],
                fused[:, horizon],
                truth[:, horizon],
                alpha[:, horizon],
                np,
            )
        )

    dataset_dir = output_root / dataset
    dataset_dir.mkdir(parents=True, exist_ok=True)
    write_csv(dataset_dir / "branch_reliability_by_horizon.csv", rows)
    overall = reliability_row(0, spatial, temporal, fused, truth, alpha, np)
    overall.pop("horizon", None)
    report = {
        "dataset": dataset,
        "checkpoint": str(checkpoint_path),
        "checkpoint_epoch": int(checkpoint.get("epoch", -1)),
        "checkpoint_val_loss": float(checkpoint.get("val_loss", float("nan"))),
        "test_samples": int(fused.shape[0]),
        "forecast_horizons": int(fused.shape[1]),
        "grid_shape": [int(height), int(width)],
        "inference_seconds": elapsed,
        "alpha_definition": "weight assigned to calibrated spatial branch",
        "reliability_definition": "branch has smaller absolute error at a sample-horizon-grid position",
        "overall": overall,
    }
    with (dataset_dir / "branch_reliability_summary.json").open("w", encoding="utf-8") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2, allow_nan=True)
    if cli.save_predictions:
        np.savez_compressed(
            dataset_dir / "branch_predictions.npz",
            truth=truth,
            spatial=spatial,
            temporal=temporal,
            fused=fused,
            alpha_spatial=alpha,
        )
    plot_dataset(dataset, rows, dataset_dir)
    print(
        f"[{dataset}] spatial MAE={overall['spatial_mae']:.6f}, "
        f"temporal MAE={overall['temporal_mae']:.6f}, "
        f"fused MAE={overall['fused_mae']:.6f}, "
        f"gate winner accuracy={overall['gate_winner_accuracy']:.4f}",
        flush=True,
    )
    return rows, report


def main() -> None:
    cli = build_cli().parse_args()
    cli.project_root = cli.project_root.expanduser().resolve()
    cli.results_root = resolve_from_project(cli.project_root, cli.results_root)
    cli.output_dir = resolve_from_project(cli.project_root, cli.output_dir)
    if not cli.project_root.is_dir():
        raise FileNotFoundError(f"Project root not found: {cli.project_root}")
    if not cli.results_root.is_dir():
        raise FileNotFoundError(f"Results root not found: {cli.results_root}")

    try:
        import numpy as np
        import torch
    except ImportError as exc:
        raise RuntimeError(
            "Run this script in the same Python environment used to train GTSRNet "
            "(requires torch and numpy)."
        ) from exc

    train_module = import_training_module(cli.project_root)
    cli.output_dir.mkdir(parents=True, exist_ok=True)
    all_rows = {}
    reports = []
    for dataset in cli.datasets:
        rows, report = evaluate_dataset(
            cli, train_module, torch, np, dataset, cli.output_dir
        )
        all_rows[dataset] = rows
        reports.append(report)

    plot_combined(all_rows, cli.output_dir)
    overall_rows = []
    for report in reports:
        overall_rows.append({"dataset": report["dataset"], **report["overall"]})
    write_csv(cli.output_dir / "branch_reliability_overall.csv", overall_rows)
    with (cli.output_dir / "run_manifest.json").open("w", encoding="utf-8") as stream:
        json.dump(
            {
                "project_root": str(cli.project_root),
                "results_root": str(cli.results_root),
                "variant": cli.variant,
                "datasets": list(cli.datasets),
                "reports": reports,
            },
            stream,
            ensure_ascii=False,
            indent=2,
            allow_nan=True,
        )
    print(f"\nAll outputs written to: {cli.output_dir}", flush=True)


if __name__ == "__main__":
    main()
