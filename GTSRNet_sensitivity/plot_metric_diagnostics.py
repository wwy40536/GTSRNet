"""Plot per-horizon curves and per-grid heatmaps for augmented metrics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


METRICS = (
    ("rmse", "RMSE"),
    ("mae", "MAE"),
    ("r2", "$R^2$"),
    ("smape_percent", "sMAPE (%)"),
    ("mase", "MASE"),
    ("psnr_db", "PSNR (dB)"),
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Plot augmented metric diagnostics.")
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--variants", nargs="+", default=("GTSRNet",))
    parser.add_argument("--datasets", nargs="+", default=None)
    parser.add_argument("--dpi", type=int, default=300)
    return parser


def plot_one(path: Path, output_dir: Path, dpi: int) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    payload = json.loads(path.read_text(encoding="utf-8"))
    metrics = payload["metrics"]
    horizons = metrics.get("horizons", [])
    grids = metrics.get("grids", [])
    if not horizons or not grids or "r2" not in horizons[0]:
        raise ValueError(f"Metrics have not been augmented: {path}")

    fig, axes = plt.subplots(2, 3, figsize=(12.0, 7.0), constrained_layout=True)
    x = [int(row["horizon"]) for row in horizons]
    for axis, (key, label) in zip(axes.flat, METRICS):
        y = [float(row[key]) for row in horizons]
        axis.plot(x, y, marker="o", linewidth=1.7, color="#3B6FB6")
        axis.set_xlabel("Forecast step (hour)")
        axis.set_ylabel(label)
        axis.set_xticks(x)
        axis.grid(alpha=0.25, linestyle="--")
    fig.suptitle("Metrics by forecast step")
    output_dir.mkdir(parents=True, exist_ok=True)
    horizon_path = output_dir / "fig_metrics_by_horizon.png"
    fig.savefig(horizon_path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    height = max(int(row["row"]) for row in grids) + 1
    width = max(int(row["col"]) for row in grids) + 1
    fig, axes = plt.subplots(2, 3, figsize=(12.0, 7.2), constrained_layout=True)
    for axis, (key, label) in zip(axes.flat, METRICS):
        image = np.full((height, width), np.nan, dtype=np.float64)
        for row in grids:
            value = row.get(key)
            if value is not None:
                image[int(row["row"]), int(row["col"])] = float(value)
        shown = axis.imshow(image, cmap="viridis", aspect="auto")
        axis.set_title(label)
        axis.set_xlabel("Grid column")
        axis.set_ylabel("Grid row")
        fig.colorbar(shown, ax=axis, fraction=0.046, pad=0.04)
    fig.suptitle("Metrics by spatial grid")
    grid_path = output_dir / "fig_metrics_by_grid.png"
    fig.savefig(grid_path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"WROTE {horizon_path}", flush=True)
    print(f"WROTE {grid_path}", flush=True)


def main() -> None:
    args = build_parser().parse_args()
    results_dir = args.results_dir.expanduser().resolve()
    matched = 0
    for path in sorted(results_dir.glob("*/*/metrics.json")):
        dataset, variant = path.parents[1].name, path.parent.name
        if variant not in args.variants:
            continue
        if args.datasets and dataset not in args.datasets:
            continue
        plot_one(
            path,
            results_dir / "extra_metrics_summary" / "figures" / dataset / variant,
            args.dpi,
        )
        matched += 1
    if not matched:
        raise FileNotFoundError("No matching augmented metrics found")


if __name__ == "__main__":
    main()
