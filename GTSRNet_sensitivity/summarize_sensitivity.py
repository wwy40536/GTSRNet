"""Aggregate multi-seed sensitivity metrics and draw reviewer-ready figures."""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np


CONFIG_RE = re.compile(r"^hist_(\d+)_pred_(\d+)$")
SEED_RE = re.compile(r"^seed_(-?\d+)$")
METRICS = ("rmse", "mae", "r2", "smape_percent", "mase", "psnr_db")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Summarize GTSRNet sensitivity runs.")
    parser.add_argument("--results-dir", type=Path, default=Path("results_GTSRNet_sensitivity"))
    parser.add_argument("--histories", nargs="+", type=int, default=(12, 24, 48, 72))
    parser.add_argument("--horizons", nargs="+", type=int, default=(3, 6, 12, 24))
    parser.add_argument("--base-history", type=int, default=24)
    parser.add_argument("--base-horizon", type=int, default=12)
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--allow-partial", action="store_true")
    parser.add_argument("--no-plot", action="store_true", help="Write CSV files only.")
    return parser


def mean_std(values: Sequence[float]) -> Tuple[float, float]:
    array = np.asarray(values, dtype=np.float64)
    return float(array.mean()), float(array.std(ddof=1)) if len(array) > 1 else 0.0


def write_csv(path: Path, rows: Sequence[Mapping[str, object]], fields: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    print(f"WROTE {path}", flush=True)


def collect(results_dir: Path) -> Tuple[List[Dict[str, object]], List[Dict[str, object]]]:
    runs: List[Dict[str, object]] = []
    steps: List[Dict[str, object]] = []
    pattern = "configs/hist_*_pred_*/seed_*/*/GTSRNet/metrics.json"
    for path in sorted(results_dir.glob(pattern)):
        try:
            config_name = path.parents[3].name
            seed_name = path.parents[2].name
            dataset = path.parents[1].name
            match_config = CONFIG_RE.match(config_name)
            match_seed = SEED_RE.match(seed_name)
            if not match_config or not match_seed:
                continue
            history, horizon = map(int, match_config.groups())
            seed = int(match_seed.group(1))
            payload = json.loads(path.read_text(encoding="utf-8"))
            metrics = payload["metrics"]
            config = payload.get("config", {})
            if int(config.get("close_size", history)) != history:
                raise ValueError(f"history mismatch in {path}")
            if int(config.get("predict_size", horizon)) != horizon:
                raise ValueError(f"horizon mismatch in {path}")
            row: Dict[str, object] = {
                "dataset": dataset,
                "history_hours": history,
                "forecast_hours": horizon,
                "seed": seed,
                "rmse": float(metrics["rmse"]),
                "mae": float(metrics["mae"]),
                "r2": float(metrics["r2"]),
                "smape_percent": float(metrics["smape_percent"]),
                "mase": float(metrics["mase"]),
                "psnr_db": float(metrics["psnr_db"]),
                "best_epoch": int(metrics["best_epoch"]),
                "parameters": int(metrics["parameters"]),
                "training_seconds": float(metrics.get("training_seconds", float("nan"))),
                "inference_ms_per_sample": float(metrics.get("inference_ms_per_sample", float("nan"))),
                "gpu_name": str(metrics.get("gpu_name", "unknown")),
                "metrics_path": str(path),
            }
            runs.append(row)
            for item in metrics.get("horizons", []):
                steps.append(
                    {
                        "dataset": dataset,
                        "history_hours": history,
                        "forecast_hours": horizon,
                        "seed": seed,
                        "forecast_step": int(item["horizon"]),
                        **{metric: float(item[metric]) for metric in METRICS},
                    }
                )
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise ValueError(f"Cannot parse {path}: {exc}") from exc
    if not runs:
        raise FileNotFoundError(f"No completed metrics found below {results_dir / 'configs'}")
    return runs, steps


def aggregate_axis(
    runs: Sequence[Mapping[str, object]],
    histories: Sequence[int],
    horizons: Sequence[int],
    base_history: int,
    base_horizon: int,
) -> List[Dict[str, object]]:
    selected: List[Tuple[str, int, int, int]] = []
    selected.extend(("input_window", value, value, base_horizon) for value in histories)
    selected.extend(("forecast_horizon", value, base_history, value) for value in horizons)
    grouped: Dict[Tuple[str, int, int, str], List[float]] = defaultdict(list)
    for row in runs:
        key = (str(row["dataset"]), int(row["history_hours"]), int(row["forecast_hours"]))
        for metric in METRICS:
            grouped[key + (metric,)].append(float(row[metric]))

    datasets = sorted({str(row["dataset"]) for row in runs})
    summary: List[Dict[str, object]] = []
    for dataset in datasets:
        for axis, length, history, horizon in selected:
            for metric in METRICS:
                values = grouped.get((dataset, history, horizon, metric), [])
                if not values:
                    continue
                mean, std = mean_std(values)
                summary.append(
                    {
                        "dataset": dataset,
                        "axis": axis,
                        "length_hours": length,
                        "history_hours": history,
                        "forecast_hours": horizon,
                        "metric": metric,
                        "mean": mean,
                        "std": std,
                        "n_seeds": len(values),
                        "mean_pm_std": f"{mean:.4f} ± {std:.4f}",
                    }
                )
    return summary


def reviewer_tables(summary: Sequence[Mapping[str, object]]) -> Tuple[List[Dict[str, object]], List[Dict[str, object]]]:
    grouped: Dict[Tuple[str, str, int], Dict[str, Mapping[str, object]]] = defaultdict(dict)
    for row in summary:
        grouped[(str(row["dataset"]), str(row["axis"]), int(row["length_hours"]))][str(row["metric"])] = row
    tables: Dict[str, List[Dict[str, object]]] = {"input_window": [], "forecast_horizon": []}
    for (dataset, axis, length), metrics in sorted(grouped.items()):
        if any(metric not in metrics for metric in METRICS):
            continue
        row: Dict[str, object] = {
            "dataset": dataset,
            "length_hours": length,
            "history_hours": metrics["rmse"]["history_hours"],
            "forecast_hours": metrics["rmse"]["forecast_hours"],
            "n_seeds": metrics["rmse"]["n_seeds"],
        }
        for metric in METRICS:
            row[f"{metric}_mean"] = metrics[metric]["mean"]
            row[f"{metric}_std"] = metrics[metric]["std"]
            row[f"{metric} (mean ± std)"] = metrics[metric]["mean_pm_std"]
        tables[axis].append(row)
    return tables["input_window"], tables["forecast_horizon"]


def aggregate_efficiency(runs: Sequence[Mapping[str, object]]) -> List[Dict[str, object]]:
    grouped: Dict[Tuple[str, int, int], List[Mapping[str, object]]] = defaultdict(list)
    for row in runs:
        grouped[(str(row["dataset"]), int(row["history_hours"]), int(row["forecast_hours"]))].append(row)
    output: List[Dict[str, object]] = []
    for (dataset, history, horizon), rows in sorted(grouped.items()):
        inference = [float(row["inference_ms_per_sample"]) for row in rows]
        training = [float(row["training_seconds"]) for row in rows]
        inference_mean, inference_std = mean_std(inference)
        training_mean, training_std = mean_std(training)
        output.append(
            {
                "dataset": dataset,
                "history_hours": history,
                "forecast_hours": horizon,
                "parameters": int(rows[0]["parameters"]),
                "inference_ms_per_sample_mean": inference_mean,
                "inference_ms_per_sample_std": inference_std,
                "training_seconds_mean": training_mean,
                "training_seconds_std": training_std,
                "n_seeds": len(rows),
                "gpu_names": " | ".join(sorted({str(row["gpu_name"]) for row in rows})),
            }
        )
    return output


def plot_dataset(
    dataset: str,
    summary: Sequence[Mapping[str, object]],
    output: Path,
    dpi: int,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    subset = [row for row in summary if row["dataset"] == dataset]
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 7.2), constrained_layout=True)
    specifications = (
        ("input_window", "rmse", "Input window (hours)", "RMSE"),
        ("forecast_horizon", "rmse", "Forecast horizon (hours)", "RMSE"),
        ("input_window", "mae", "Input window (hours)", "MAE"),
        ("forecast_horizon", "mae", "Forecast horizon (hours)", "MAE"),
    )
    for axis, (kind, metric, xlabel, ylabel) in zip(axes.flat, specifications):
        rows = sorted(
            (row for row in subset if row["axis"] == kind and row["metric"] == metric),
            key=lambda row: int(row["length_hours"]),
        )
        x = [int(row["length_hours"]) for row in rows]
        y = [float(row["mean"]) for row in rows]
        error = [float(row["std"]) for row in rows]
        axis.errorbar(x, y, yerr=error, marker="o", linewidth=1.8, capsize=4, color="#3B6FB6")
        axis.set_xlabel(xlabel)
        axis.set_ylabel(ylabel)
        axis.set_xticks(x)
        axis.grid(alpha=0.25, linestyle="--")
    fig.suptitle(f"GTSRNet sensitivity — {dataset.replace('_', ' ').title()}", fontsize=14)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"WROTE {output}", flush=True)


def check_completeness(
    runs: Sequence[Mapping[str, object]],
    histories: Sequence[int],
    horizons: Sequence[int],
    base_history: int,
    base_horizon: int,
) -> List[str]:
    datasets = sorted({str(row["dataset"]) for row in runs})
    seeds = sorted({int(row["seed"]) for row in runs})
    settings = {(value, base_horizon) for value in histories}
    settings.update((base_history, value) for value in horizons)
    present = {
        (str(row["dataset"]), int(row["history_hours"]), int(row["forecast_hours"]), int(row["seed"]))
        for row in runs
    }
    return [
        f"{dataset}/hist_{history:03d}_pred_{horizon:03d}/seed_{seed}"
        for dataset in datasets
        for history, horizon in sorted(settings)
        for seed in seeds
        if (dataset, history, horizon, seed) not in present
    ]


def main() -> None:
    args = build_parser().parse_args()
    results_dir = args.results_dir.expanduser().resolve()
    runs, steps = collect(results_dir)
    missing = check_completeness(
        runs, args.histories, args.horizons, args.base_history, args.base_horizon
    )
    if missing and not args.allow_partial:
        preview = "\n  ".join(missing[:20])
        suffix = f"\n  ... and {len(missing) - 20} more" if len(missing) > 20 else ""
        raise RuntimeError(f"Missing {len(missing)} expected runs:\n  {preview}{suffix}")
    if missing:
        print(f"WARNING: summarizing partial results; {len(missing)} runs are missing", flush=True)

    summary = aggregate_axis(
        runs, args.histories, args.horizons, args.base_history, args.base_horizon
    )
    efficiency = aggregate_efficiency(runs)
    input_table, horizon_table = reviewer_tables(summary)
    summary_dir = results_dir / "summary"
    write_csv(
        summary_dir / "sensitivity_runs.csv",
        runs,
        ("dataset", "history_hours", "forecast_hours", "seed", *METRICS, "best_epoch", "parameters", "training_seconds", "inference_ms_per_sample", "gpu_name", "metrics_path"),
    )
    write_csv(
        summary_dir / "per_step_runs.csv",
        steps,
        ("dataset", "history_hours", "forecast_hours", "seed", "forecast_step", *METRICS),
    )
    write_csv(
        summary_dir / "sensitivity_summary.csv",
        summary,
        ("dataset", "axis", "length_hours", "history_hours", "forecast_hours", "metric", "mean", "std", "n_seeds", "mean_pm_std"),
    )
    write_csv(
        summary_dir / "efficiency_summary.csv",
        efficiency,
        ("dataset", "history_hours", "forecast_hours", "parameters", "inference_ms_per_sample_mean", "inference_ms_per_sample_std", "training_seconds_mean", "training_seconds_std", "n_seeds", "gpu_names"),
    )
    table_fields = ["dataset", "length_hours", "history_hours", "forecast_hours", "n_seeds"]
    for metric in METRICS:
        table_fields.extend((f"{metric}_mean", f"{metric}_std", f"{metric} (mean ± std)"))
    write_csv(summary_dir / "table_input_window.csv", input_table, table_fields)
    write_csv(summary_dir / "table_forecast_horizon.csv", horizon_table, table_fields)
    if not args.no_plot:
        for dataset in sorted({str(row["dataset"]) for row in runs}):
            plot_dataset(dataset, summary, summary_dir / "figures" / f"fig_sensitivity_{dataset}.png", args.dpi)


if __name__ == "__main__":
    main()
