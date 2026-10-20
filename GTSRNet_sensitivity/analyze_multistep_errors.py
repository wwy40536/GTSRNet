"""Create detailed multi-step error diagnostics from exported predictions."""


from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Analyze step-wise error distributions.")
    parser.add_argument("--predictions-dir", type=Path, required=True)
    parser.add_argument("--datasets", nargs="+", default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--cdf-max-points", type=int, default=200000)
    parser.add_argument("--cdf-zoom-quantile", type=float, default=0.995)
    parser.add_argument("--no-plots", action="store_true", help="Write tables only.")
    return parser


def write_csv(path: Path, rows: Sequence[Mapping[str, object]], fields: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    print(f"WROTE {path}", flush=True)


def discover(root: Path, selected: Optional[Sequence[str]]) -> List[Tuple[str, Path]]:
    found = []
    for path in sorted(root.glob("*/predictions.npz")):
        dataset = path.parent.name
        if not selected or dataset in selected:
            found.append((dataset, path))
    if not found:
        raise FileNotFoundError(f"No <dataset>/predictions.npz found below {root}")
    return found


def metrics(error: np.ndarray) -> Dict[str, float]:
    signed = np.asarray(error, dtype=np.float64).reshape(-1)
    absolute = np.abs(signed)
    return {
        "mae": float(absolute.mean()),
        "rmse": float(np.sqrt(np.mean(signed * signed))),
        "signed_error_mean": float(signed.mean()),
        "signed_error_std": float(signed.std(ddof=1)) if signed.size > 1 else 0.0,
        "absolute_error_std": float(absolute.std(ddof=1)) if absolute.size > 1 else 0.0,
        "p95_absolute_error": float(np.quantile(absolute, 0.95)),
        "p99_absolute_error": float(np.quantile(absolute, 0.99)),
        "max_absolute_error": float(absolute.max()),
    }


def original_coords(config: Mapping[str, object], row: int, col: int) -> Tuple[int, int]:
    if bool(config.get("crop", False)):
        rows, cols = config.get("rows", [0, 0]), config.get("cols", [0, 0])
        if isinstance(rows, Sequence) and not isinstance(rows, str) and len(rows) == 2:
            row += int(rows[0])
        if isinstance(cols, Sequence) and not isinstance(cols, str) and len(cols) == 2:
            col += int(cols[0])
    return row, col


def plotting():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def plot_curves(dataset: str, rows: Sequence[Mapping[str, object]], path: Path, dpi: int) -> None:
    plt = plotting()
    steps = [int(row["forecast_step"]) for row in rows]
    fig, axes = plt.subplots(2, 1, figsize=(8.6, 7.2), sharex=True, constrained_layout=True)
    axes[0].plot(steps, [row["mae"] for row in rows], marker="o", label="MAE")
    axes[0].plot(steps, [row["rmse"] for row in rows], marker="s", label="RMSE")
    axes[0].set_ylabel("Error")
    axes[0].legend(frameon=False)
    for key, label, marker in (
        ("absolute_error_std", "Std(|error|)", "o"),
        ("p95_absolute_error", "P95", "^"),
        ("p99_absolute_error", "P99", "s"),
    ):
        axes[1].plot(steps, [row[key] for row in rows], marker=marker, label=label)
    axes[1].set_ylabel("Distribution statistic")
    axes[1].set_xlabel("Forecast step (hour)")
    axes[1].legend(frameon=False, ncol=3)
    for axis in axes:
        axis.set_xticks(steps)
        axis.grid(alpha=0.25, linestyle="--")
    fig.suptitle(f"Multi-step errors — {dataset.replace('_', ' ').title()}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_boxplot(dataset: str, error: np.ndarray, path: Path, dpi: int) -> None:
    plt = plotting()
    sample_mae = np.mean(np.abs(error), axis=tuple(range(2, error.ndim)))
    fig, axis = plt.subplots(figsize=(9.0, 4.8), constrained_layout=True)
    axis.boxplot([sample_mae[:, step] for step in range(sample_mae.shape[1])], showfliers=True)
    axis.set_xlabel("Forecast step (hour)")
    axis.set_ylabel("Per-sample spatial MAE")
    axis.set_title(f"Step-wise error distribution — {dataset.replace('_', ' ').title()}")
    axis.grid(axis="y", alpha=0.25, linestyle="--")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_cdf(
    dataset: str,
    error: np.ndarray,
    path: Path,
    dpi: int,
    max_points: int,
    zoom_quantile: float,
) -> None:
    plt = plotting()
    rng = np.random.default_rng(0)
    absolute = np.abs(error)
    pooled = absolute.reshape(-1)
    zoom_max = float(np.quantile(pooled, zoom_quantile))
    p95, p99 = (float(np.quantile(pooled, value)) for value in (0.95, 0.99))
    fig, axes = plt.subplots(1, 2, figsize=(12.4, 5.2), sharey=True, constrained_layout=True)
    for step in range(error.shape[1]):
        values = absolute[:, step].reshape(-1)
        if values.size > max_points:
            values = values[rng.choice(values.size, max_points, replace=False)]
        values = np.sort(values)
        probability = np.arange(1, len(values) + 1) / len(values)
        for axis in axes:
            axis.plot(values, probability, linewidth=1.0, alpha=0.85, label=f"t+{step + 1}")
    axes[0].set_xlim(0, zoom_max)
    axes[0].axvline(p95, color="#555555", linestyle="--", linewidth=1.0, label="P95")
    axes[0].axvline(p99, color="#222222", linestyle="--", linewidth=1.0, label="P99")
    axes[0].set_title(f"Central distribution (0–P{zoom_quantile * 100:g})")
    axes[1].set_xscale("symlog", linthresh=1.0)
    axes[1].set_xlim(0, float(pooled.max()) * 1.03)
    axes[1].set_title("Full tail (log-scaled x-axis)")
    for axis in axes:
        axis.set_xlabel("Absolute error")
        axis.set_ylim(0, 1.005)
        axis.grid(alpha=0.25, linestyle="--")
    axes[0].set_ylabel("Empirical CDF")
    axes[0].legend(frameon=False, ncol=3, fontsize=8)
    fig.suptitle(f"Absolute-error empirical CDF — {dataset.replace('_', ' ').title()}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> None:
    args = build_parser().parse_args()
    if args.cdf_max_points <= 0:
        raise ValueError("--cdf-max-points must be positive")
    if not 0.9 < args.cdf_zoom_quantile < 1.0:
        raise ValueError("--cdf-zoom-quantile must be between 0.9 and 1")
    predictions_dir = args.predictions_dir.expanduser().resolve()
    output_dir = (args.output_dir or (predictions_dir / "analysis")).expanduser().resolve()
    all_steps: List[Dict[str, object]] = []
    all_grids: List[Dict[str, object]] = []
    all_overall: List[Dict[str, object]] = []
    worst_cases: Dict[str, object] = {}

    for dataset, path in discover(predictions_dir, args.datasets):
        archive = np.load(path, allow_pickle=False)
        pred = np.asarray(archive["pred"], dtype=np.float64)
        truth = np.asarray(archive["truth"], dtype=np.float64)
        if pred.shape != truth.shape or pred.ndim != 5 or pred.shape[2] != 1:
            raise ValueError(f"{path}: expected matching [N,H,1,R,C], got {pred.shape}/{truth.shape}")
        error = pred - truth
        manifest_path = path.with_name("evaluation_manifest.json")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.is_file() else {}
        config = manifest.get("config", {})
        target_time = archive["target_time"] if "target_time" in archive.files else None

        overall = {"dataset": dataset, "test_samples": pred.shape[0], **metrics(error)}
        all_overall.append(overall)
        step_rows = []
        for step in range(error.shape[1]):
            row = {"dataset": dataset, "forecast_step": step + 1, **metrics(error[:, step])}
            step_rows.append(row)
            all_steps.append(row)

        grid_rows = []
        for row_idx in range(error.shape[-2]):
            for col_idx in range(error.shape[-1]):
                source_row, source_col = original_coords(config, row_idx, col_idx)
                row = {
                    "dataset": dataset,
                    "grid_index": row_idx * error.shape[-1] + col_idx,
                    "crop_row": row_idx,
                    "crop_col": col_idx,
                    "source_row": source_row,
                    "source_col": source_col,
                    **metrics(error[:, :, 0, row_idx, col_idx]),
                }
                grid_rows.append(row)
                all_grids.append(row)

        absolute = np.abs(error)
        flat_index = int(np.argmax(absolute))
        sample, step, _channel, row_idx, col_idx = np.unravel_index(flat_index, absolute.shape)
        source_row, source_col = original_coords(config, row_idx, col_idx)
        worst_step_mae = max(step_rows, key=lambda row: float(row["mae"]))
        worst_step_rmse = max(step_rows, key=lambda row: float(row["rmse"]))
        worst_grid = max(grid_rows, key=lambda row: float(row["mae"]))
        timestamp = str(target_time[sample, step]) if target_time is not None else None
        worst_cases[dataset] = {
            "worst_forecast_step_by_mae": int(worst_step_mae["forecast_step"]),
            "worst_forecast_step_by_rmse": int(worst_step_rmse["forecast_step"]),
            "worst_grid_by_mae": worst_grid,
            "maximum_absolute_error_event": {
                "test_sample_index": int(sample),
                "forecast_step": int(step + 1),
                "target_time": timestamp,
                "crop_row": int(row_idx),
                "crop_col": int(col_idx),
                "source_row": int(source_row),
                "source_col": int(source_col),
                "prediction": float(pred[sample, step, 0, row_idx, col_idx]),
                "truth": float(truth[sample, step, 0, row_idx, col_idx]),
                "signed_error": float(error[sample, step, 0, row_idx, col_idx]),
                "absolute_error": float(absolute[sample, step, 0, row_idx, col_idx]),
            },
        }
        if not args.no_plots:
            figure_dir = output_dir / "figures" / dataset
            plot_curves(dataset, step_rows, figure_dir / "fig_multistep_curve.png", args.dpi)
            plot_boxplot(dataset, error, figure_dir / "fig_multistep_boxplot.png", args.dpi)
            plot_cdf(
                dataset,
                error,
                figure_dir / "fig_absolute_error_cdf.png",
                args.dpi,
                args.cdf_max_points,
                args.cdf_zoom_quantile,
            )

    fields = (
        "dataset", "mae", "rmse", "signed_error_mean", "signed_error_std",
        "absolute_error_std", "p95_absolute_error", "p99_absolute_error", "max_absolute_error",
    )
    write_csv(output_dir / "overall_error_distribution.csv", all_overall, ("dataset", "test_samples", *fields[1:]))
    write_csv(output_dir / "per_step_error_distribution.csv", all_steps, ("dataset", "forecast_step", *fields[1:]))
    write_csv(
        output_dir / "per_grid_error_distribution.csv",
        all_grids,
        ("dataset", "grid_index", "crop_row", "crop_col", "source_row", "source_col", *fields[1:]),
    )
    (output_dir / "worst_cases.json").write_text(
        json.dumps(worst_cases, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"WROTE {output_dir / 'worst_cases.json'}", flush=True)


if __name__ == "__main__":
    main()
