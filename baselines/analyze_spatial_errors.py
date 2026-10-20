"""Create per-seed spatial error heatmaps and representative-grid curves."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, Mapping, Sequence, Tuple

import numpy as np


DEFAULT_DATASETS = ("milan_sms", "milan_call", "milan_internet", "mobile_bj", "mobile_nj")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=Path("results_GTSRNet_baselines"))
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--datasets", nargs="+", choices=DEFAULT_DATASETS, default=DEFAULT_DATASETS)
    parser.add_argument("--seeds", nargs="+", type=int, default=(16, 26, 36))
    parser.add_argument("--metric", choices=("mae", "rmse"), default="mae")
    parser.add_argument("--curve-horizon", type=int, default=12)
    parser.add_argument("--curve-points", type=int, default=168)
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--formats", nargs="+", choices=("png", "pdf"), default=("png",))
    parser.add_argument("--no-plots", action="store_true")
    return parser


def load_config(run_dir: Path) -> Mapping[str, object]:
    path = run_dir / "metrics.json"
    if not path.is_file():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    config = payload.get("config", {})
    return config if isinstance(config, dict) else {}


def original_offsets(dataset: str, config: Mapping[str, object], height: int, width: int) -> Tuple[int, int]:
    if not dataset.startswith("milan_") or not bool(config.get("crop", True)):
        return 0, 0
    rows = config.get("rows", (40, 60))
    cols = config.get("cols", (40, 60))
    try:
        row_start, row_end = int(rows[0]), int(rows[1])  # type: ignore[index]
        col_start, col_end = int(cols[0]), int(cols[1])  # type: ignore[index]
    except (TypeError, ValueError, IndexError):
        return 0, 0
    if row_end - row_start == height and col_end - col_start == width:
        return row_start, col_start
    return 0, 0


def grid_metrics(pred: np.ndarray, truth: np.ndarray) -> Dict[str, np.ndarray]:
    if pred.shape != truth.shape:
        raise ValueError(f"pred/truth shape mismatch: {pred.shape} vs {truth.shape}")
    if pred.ndim != 5:
        raise ValueError(f"Expected [sample,horizon,flow,row,col], got shape {pred.shape}")
    error = pred.astype(np.float64) - truth.astype(np.float64)
    axes = (0, 1, 2)
    mae = np.mean(np.abs(error), axis=axes)
    rmse = np.sqrt(np.mean(np.square(error), axis=axes))
    mean_traffic = np.mean(truth.astype(np.float64), axis=axes)
    relative_mae = mae / np.maximum(np.abs(mean_traffic), 1e-8)
    return {"mae": mae, "rmse": rmse, "mean_traffic": mean_traffic, "relative_mae": relative_mae}


def select_grids(metrics: Mapping[str, np.ndarray], metric: str) -> Dict[str, Tuple[int, int]]:
    score = metrics[metric]
    traffic = metrics["mean_traffic"]
    finite = np.isfinite(score) & np.isfinite(traffic)
    positive = finite & (np.abs(traffic) > 1e-8)
    active = positive if np.any(positive) else finite
    if not np.any(active):
        raise ValueError("No finite grid metrics are available")
    best_score = np.where(active, score, np.inf)
    worst_score = np.where(active, score, -np.inf)
    traffic_score = np.where(active, traffic, -np.inf)
    return {
        "best": tuple(int(v) for v in np.unravel_index(np.argmin(best_score), score.shape)),
        "worst": tuple(int(v) for v in np.unravel_index(np.argmax(worst_score), score.shape)),
        "high_traffic": tuple(int(v) for v in np.unravel_index(np.argmax(traffic_score), score.shape)),
    }


def write_grid_csv(
    path: Path,
    metrics: Mapping[str, np.ndarray],
    selected: Mapping[str, Tuple[int, int]],
    offsets: Tuple[int, int],
) -> None:
    labels: Dict[Tuple[int, int], list[str]] = {}
    for label, coordinate in selected.items():
        labels.setdefault(coordinate, []).append(label)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        fields = ("local_row", "local_col", "original_row", "original_col", "mae", "rmse", "mean_traffic", "relative_mae", "selection")
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        height, width = metrics["mae"].shape
        for row in range(height):
            for col in range(width):
                writer.writerow(
                    {
                        "local_row": row,
                        "local_col": col,
                        "original_row": row + offsets[0],
                        "original_col": col + offsets[1],
                        "mae": float(metrics["mae"][row, col]),
                        "rmse": float(metrics["rmse"][row, col]),
                        "mean_traffic": float(metrics["mean_traffic"][row, col]),
                        "relative_mae": float(metrics["relative_mae"][row, col]),
                        "selection": "+".join(labels.get((row, col), [])),
                    }
                )


def plot_diagnostics(
    dataset: str,
    seed: int,
    pred: np.ndarray,
    truth: np.ndarray,
    metrics: Mapping[str, np.ndarray],
    selected: Mapping[str, Tuple[int, int]],
    offsets: Tuple[int, int],
    metric: str,
    horizon: int,
    curve_points: int,
    output_stem: Path,
    dpi: int,
    formats: Sequence[str],
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    horizon_index = horizon - 1
    if horizon_index < 0 or horizon_index >= pred.shape[1]:
        raise ValueError(f"--curve-horizon must be in [1, {pred.shape[1]}]")
    count = min(curve_points, pred.shape[0]) if curve_points > 0 else pred.shape[0]
    start = max(0, pred.shape[0] - count)
    x = np.arange(start, pred.shape[0])
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), constrained_layout=True)

    heat_axis = axes[0, 0]
    heat = heat_axis.imshow(metrics[metric], origin="upper", cmap="magma", aspect="equal")
    fig.colorbar(heat, ax=heat_axis, fraction=0.046, pad=0.04, label=metric.upper())
    marker_style = {"best": ("o", "#3CB44B"), "worst": ("X", "#00FFFF"), "high_traffic": ("s", "#FFFFFF")}
    for label, (row, col) in selected.items():
        marker, color = marker_style[label]
        heat_axis.scatter(col, row, marker=marker, s=95, facecolors="none", edgecolors=color, linewidths=1.8, label=label.replace("_", " "))
    heat_axis.legend(loc="upper right", fontsize=8, framealpha=0.85)
    heat_axis.set_title(f"Per-grid {metric.upper()} over all test samples and horizons")
    heat_axis.set_xlabel("Grid column")
    heat_axis.set_ylabel("Grid row")

    curve_axes = (axes[0, 1], axes[1, 0], axes[1, 1])
    for axis, label in zip(curve_axes, ("best", "worst", "high_traffic")):
        row, col = selected[label]
        actual = truth[start:, horizon_index, :, row, col].mean(axis=1)
        forecast = pred[start:, horizon_index, :, row, col].mean(axis=1)
        axis.plot(x, actual, color="#222222", linewidth=1.4, label="Truth")
        axis.plot(x, forecast, color="#C35A4A", linewidth=1.25, label="Prediction")
        original = (row + offsets[0], col + offsets[1])
        axis.set_title(
            f"{label.replace('_', ' ').title()} grid {original} | "
            f"MAE={metrics['mae'][row, col]:.4g}, mean flow={metrics['mean_traffic'][row, col]:.4g}"
        )
        axis.set_xlabel("Test forecast origin")
        axis.set_ylabel("Traffic")
        axis.grid(linestyle="--", alpha=0.25)
        axis.legend(loc="best")

    fig.suptitle(f"GTSRNet spatial diagnostics — {dataset.replace('_', ' ').title()} — seed {seed} — horizon {horizon}")
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    for file_format in formats:
        output = output_stem.with_suffix(f".{file_format}")
        fig.savefig(output, dpi=dpi, bbox_inches="tight", facecolor="white")
        print(f"WROTE {output}", flush=True)
    plt.close(fig)


def analyze_run(args: argparse.Namespace, dataset: str, seed: int, output_root: Path) -> None:
    run_dir = args.results_dir / dataset / "GTSRNet" / f"seed_{seed}"
    prediction_path = run_dir / "predictions.npz"
    if not prediction_path.is_file():
        raise FileNotFoundError(f"Missing {prediction_path}; run prediction export first")
    with np.load(prediction_path) as arrays:
        pred = np.asarray(arrays["pred"])
        truth = np.asarray(arrays["truth"])
    metrics = grid_metrics(pred, truth)
    selected = select_grids(metrics, args.metric)
    config = load_config(run_dir)
    offsets = original_offsets(dataset, config, pred.shape[-2], pred.shape[-1])
    run_output = output_root / dataset / f"seed_{seed}"
    write_grid_csv(run_output / "per_grid_metrics.csv", metrics, selected, offsets)
    selection_payload = {
        "dataset": dataset,
        "model": "GTSRNet",
        "seed": seed,
        "selection_metric": args.metric,
        "curve_horizon": args.curve_horizon,
        "prediction_shape": list(pred.shape),
        "selected_grids": {
            label: {
                "local_row": row,
                "local_col": col,
                "original_row": row + offsets[0],
                "original_col": col + offsets[1],
                "mae": float(metrics["mae"][row, col]),
                "rmse": float(metrics["rmse"][row, col]),
                "mean_traffic": float(metrics["mean_traffic"][row, col]),
                "relative_mae": float(metrics["relative_mae"][row, col]),
            }
            for label, (row, col) in selected.items()
        },
    }
    (run_output / "selected_grids.json").write_text(
        json.dumps(selection_payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"WROTE {run_output / 'per_grid_metrics.csv'}", flush=True)
    print(f"WROTE {run_output / 'selected_grids.json'}", flush=True)
    if not args.no_plots:
        plot_diagnostics(
            dataset, seed, pred, truth, metrics, selected, offsets,
            args.metric, args.curve_horizon, args.curve_points,
            run_output / "spatial_diagnostics", args.dpi, args.formats,
        )


def main() -> None:
    args = build_parser().parse_args()
    args.results_dir = args.results_dir.expanduser().resolve()
    output_root = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else args.results_dir / "spatial_analysis"
    )
    completed = 0
    for dataset in args.datasets:
        for seed in args.seeds:
            analyze_run(args, dataset, seed, output_root)
            completed += 1
    print(f"DONE: analyzed {completed} GTSRNet dataset/seed runs under {output_root}", flush=True)


if __name__ == "__main__":
    main()
