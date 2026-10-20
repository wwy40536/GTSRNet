#!/usr/bin/env python3
"""Aggregate windowed GSTRNet accuracy runs into all-run and mean/std CSVs."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, required=True)
    return parser.parse_args()


def write_csv(path: Path, rows: list[dict[str, object]], fields: tuple[str, ...]) -> None:
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    print(f"WROTE {path}")


def main() -> None:
    args = parse_args()
    root = args.results_dir.expanduser().resolve()
    rows: list[dict[str, object]] = []
    for path in sorted(root.glob("milan_*/seed_*/metrics.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        metric = payload["metrics"]
        rows.append(
            {
                "dataset": metric["dataset"],
                "seed": int(metric["seed"]),
                "grid_size": int(metric["grid_size"]),
                "window_size": int(metric["window_size"]),
                "tiles_per_grid": int(metric["tiles_per_grid"]),
                "parameters": int(metric["parameters"]),
                "best_epoch": int(metric["best_epoch"]),
                "mae": float(metric["mae"]),
                "rmse": float(metric["rmse"]),
                "training_peak_memory_mb": float(metric["training_peak_memory_mb"]),
                "inference_ms_per_full_grid": float(metric["inference_ms_per_full_grid"]),
            }
        )
    if not rows:
        raise FileNotFoundError(f"No metrics.json files found below {root}")
    all_fields = tuple(rows[0].keys())
    write_csv(root / "scalability_accuracy_all_runs.csv", rows, all_fields)

    grouped: dict[tuple[str, int, int], list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["dataset"]), int(row["grid_size"]), int(row["window_size"]))].append(row)
    summary: list[dict[str, object]] = []
    for (dataset, grid_size, window_size), group in sorted(grouped.items()):
        mae = np.asarray([float(row["mae"]) for row in group])
        rmse = np.asarray([float(row["rmse"]) for row in group])
        memory = np.asarray([float(row["training_peak_memory_mb"]) for row in group])
        latency = np.asarray([float(row["inference_ms_per_full_grid"]) for row in group])
        summary.append(
            {
                "dataset": dataset,
                "grid_size": grid_size,
                "window_size": window_size,
                "n_seeds": len(group),
                "seeds": "/".join(str(row["seed"]) for row in sorted(group, key=lambda x: x["seed"])),
                "mae_mean": float(mae.mean()),
                "mae_std": float(mae.std(ddof=1)) if len(mae) > 1 else 0.0,
                "rmse_mean": float(rmse.mean()),
                "rmse_std": float(rmse.std(ddof=1)) if len(rmse) > 1 else 0.0,
                "peak_memory_mb_mean": float(memory.mean()),
                "inference_ms_mean": float(latency.mean()),
            }
        )
    summary_fields = tuple(summary[0].keys())
    write_csv(root / "scalability_accuracy_mean_std.csv", summary, summary_fields)


if __name__ == "__main__":
    main()
