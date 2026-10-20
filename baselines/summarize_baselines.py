"""Aggregate baseline runs into paper-ready CSV tables and figures."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Mapping, Sequence, Tuple

import numpy as np


DEFAULT_MODELS = ("seasonal_ha", "gru_seq2seq", "mtgnn", "patchtst", "dlinear", "timemixer", "GTSRNet")
DEFAULT_DATASETS = ("milan_sms", "milan_call", "milan_internet", "mobile_bj", "mobile_nj")
DISPLAY = {
    "seasonal_ha": "S-HA",
    "gru_seq2seq": "GRU-Seq2Seq",
    "mtgnn": "MTGNN",
    "patchtst": "PatchTST",
    "dlinear": "DLinear",
    "timemixer": "TimeMixer",
    "GTSRNet": "GTSRNet",
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=Path("results_GTSRNet_baselines"))
    parser.add_argument("--models", nargs="+", choices=DEFAULT_MODELS, default=DEFAULT_MODELS)
    parser.add_argument("--datasets", nargs="+", choices=DEFAULT_DATASETS, default=DEFAULT_DATASETS)
    parser.add_argument("--seeds", nargs="+", type=int, default=(15, 25, 35))
    parser.add_argument("--allow-partial", action="store_true")
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--no-plots", action="store_true")
    return parser


def write_csv(path: Path, rows: Sequence[Mapping[str, object]], fields: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    print(f"WROTE {path}", flush=True)


def mean_std(values: Sequence[float]) -> Tuple[float, float]:
    array = np.asarray(values, dtype=np.float64)
    return float(array.mean()), float(array.std(ddof=1)) if len(array) > 1 else 0.0


def collect(results_dir: Path) -> Tuple[List[Dict[str, object]], List[Dict[str, object]]]:
    runs: List[Dict[str, object]] = []
    steps: List[Dict[str, object]] = []
    for path in sorted(results_dir.glob("*/*/seed_*/metrics.json")):
        dataset, model, seed_name = path.parents[2].name, path.parents[1].name, path.parent.name
        if model not in DEFAULT_MODELS or not seed_name.startswith("seed_"):
            continue
        seed = int(seed_name[5:])
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            metrics = payload["metrics"]
            row: Dict[str, object] = {
                "dataset": dataset,
                "model": model,
                "display_name": DISPLAY[model],
                "seed": seed,
                "rmse": float(metrics["rmse"]),
                "mae": float(metrics["mae"]),
                "parameters": int(metrics["parameters"]),
                "best_epoch": int(metrics["best_epoch"]),
                "training_seconds": float(metrics["training_seconds"]),
                "inference_seconds": float(metrics["inference_seconds"]),
                "metrics_path": str(path),
            }
            runs.append(row)
            for item in metrics.get("horizons", []):
                steps.append(
                    {
                        "dataset": dataset,
                        "model": model,
                        "display_name": DISPLAY[model],
                        "seed": seed,
                        "forecast_step": int(item["horizon"]),
                        "rmse": float(item["rmse"]),
                        "mae": float(item["mae"]),
                    }
                )
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise ValueError(f"Cannot parse {path}: {exc}") from exc
    if not runs:
        raise FileNotFoundError(f"No metrics.json files found under {results_dir}")
    return runs, steps


def missing_runs(
    runs: Sequence[Mapping[str, object]],
    models: Sequence[str],
    datasets: Sequence[str],
    seeds: Sequence[int],
) -> List[str]:
    present = {(str(row["dataset"]), str(row["model"]), int(row["seed"])) for row in runs}
    missing: List[str] = []
    for model in models:
        expected_seeds = (seeds[0],) if model == "seasonal_ha" else seeds
        for dataset in datasets:
            for seed in expected_seeds:
                if (dataset, model, seed) not in present:
                    missing.append(f"{dataset}/{model}/seed_{seed}")
    return missing


def aggregate(runs: Sequence[Mapping[str, object]]) -> List[Dict[str, object]]:
    grouped: Dict[Tuple[str, str], List[Mapping[str, object]]] = defaultdict(list)
    for row in runs:
        grouped[(str(row["dataset"]), str(row["model"]))].append(row)
    summary: List[Dict[str, object]] = []
    for (dataset, model), rows in sorted(grouped.items()):
        rmse_mean, rmse_std = mean_std([float(row["rmse"]) for row in rows])
        mae_mean, mae_std = mean_std([float(row["mae"]) for row in rows])
        train_mean, train_std = mean_std([float(row["training_seconds"]) for row in rows])
        infer_mean, infer_std = mean_std([float(row["inference_seconds"]) for row in rows])
        summary.append(
            {
                "dataset": dataset,
                "model": model,
                "display_name": DISPLAY[model],
                "n_seeds": len(rows),
                "rmse_mean": rmse_mean,
                "rmse_std": rmse_std,
                "mae_mean": mae_mean,
                "mae_std": mae_std,
                "RMSE (mean ± std)": f"{rmse_mean:.4f} ± {rmse_std:.4f}",
                "MAE (mean ± std)": f"{mae_mean:.4f} ± {mae_std:.4f}",
                "parameters": int(round(np.mean([int(row["parameters"]) for row in rows]))),
                "training_seconds_mean": train_mean,
                "training_seconds_std": train_std,
                "inference_seconds_mean": infer_mean,
                "inference_seconds_std": infer_std,
            }
        )
    return summary


def aggregate_steps(steps: Sequence[Mapping[str, object]]) -> List[Dict[str, object]]:
    grouped: Dict[Tuple[str, str, int], Dict[str, List[float]]] = defaultdict(lambda: {"rmse": [], "mae": []})
    for row in steps:
        key = (str(row["dataset"]), str(row["model"]), int(row["forecast_step"]))
        grouped[key]["rmse"].append(float(row["rmse"]))
        grouped[key]["mae"].append(float(row["mae"]))
    output: List[Dict[str, object]] = []
    for (dataset, model, step), values in sorted(grouped.items()):
        rmse_mean, rmse_std = mean_std(values["rmse"])
        mae_mean, mae_std = mean_std(values["mae"])
        output.append(
            {
                "dataset": dataset,
                "model": model,
                "display_name": DISPLAY[model],
                "forecast_step": step,
                "rmse_mean": rmse_mean,
                "rmse_std": rmse_std,
                "mae_mean": mae_mean,
                "mae_std": mae_std,
                "n_seeds": len(values["rmse"]),
            }
        )
    return output


def wide_paper_table(
    summary: Sequence[Mapping[str, object]], models: Sequence[str], datasets: Sequence[str]
) -> List[Dict[str, object]]:
    lookup = {(str(row["dataset"]), str(row["model"])): row for row in summary}
    rows: List[Dict[str, object]] = []
    for model in models:
        row: Dict[str, object] = {"model": DISPLAY[model]}
        for dataset in datasets:
            values = lookup.get((dataset, model))
            row[f"{dataset}_RMSE"] = values["RMSE (mean ± std)"] if values else "MISSING"
            row[f"{dataset}_MAE"] = values["MAE (mean ± std)"] if values else "MISSING"
        rows.append(row)
    return rows


def plot_dataset(dataset: str, summary: Sequence[Mapping[str, object]], models: Sequence[str], output: Path, dpi: int) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    lookup = {(str(row["dataset"]), str(row["model"])): row for row in summary}
    available = [model for model in models if (dataset, model) in lookup]
    labels = [DISPLAY[model] for model in available]
    x = np.arange(len(available))
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), constrained_layout=True)
    for axis, metric, title in ((axes[0], "rmse", "RMSE"), (axes[1], "mae", "MAE")):
        means = [float(lookup[(dataset, model)][f"{metric}_mean"]) for model in available]
        stds = [float(lookup[(dataset, model)][f"{metric}_std"]) for model in available]
        colors = ["#C35A4A" if model == "GTSRNet" else "#4F77AA" for model in available]
        axis.bar(x, means, yerr=stds, capsize=3, color=colors, edgecolor="white")
        axis.set_xticks(x, labels, rotation=30, ha="right")
        axis.set_ylabel(title)
        axis.grid(axis="y", linestyle="--", alpha=0.25)
    fig.suptitle(f"Baseline comparison — {dataset.replace('_', ' ').title()}")
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"WROTE {output}", flush=True)


def main() -> None:
    args = build_parser().parse_args()
    results_dir = args.results_dir.expanduser().resolve()
    runs, steps = collect(results_dir)
    missing = missing_runs(runs, args.models, args.datasets, args.seeds)
    if missing and not args.allow_partial:
        preview = "\n  ".join(missing[:20])
        suffix = f"\n  ... and {len(missing) - 20} more" if len(missing) > 20 else ""
        raise RuntimeError(f"Missing {len(missing)} expected runs:\n  {preview}{suffix}")
    if missing:
        print(f"WARNING: partial summary; {len(missing)} expected runs are missing", flush=True)
    summary = aggregate(runs)
    step_summary = aggregate_steps(steps)
    paper_rows = wide_paper_table(summary, args.models, args.datasets)
    output = results_dir / "summary"
    write_csv(
        output / "all_runs.csv", runs,
        ("dataset", "model", "display_name", "seed", "rmse", "mae", "parameters", "best_epoch", "training_seconds", "inference_seconds", "metrics_path"),
    )
    write_csv(
        output / "baseline_summary.csv", summary,
        ("dataset", "model", "display_name", "n_seeds", "rmse_mean", "rmse_std", "mae_mean", "mae_std", "RMSE (mean ± std)", "MAE (mean ± std)", "parameters", "training_seconds_mean", "training_seconds_std", "inference_seconds_mean", "inference_seconds_std"),
    )
    write_csv(
        output / "per_horizon_summary.csv", step_summary,
        ("dataset", "model", "display_name", "forecast_step", "rmse_mean", "rmse_std", "mae_mean", "mae_std", "n_seeds"),
    )
    wide_fields = ["model"] + [f"{dataset}_{metric}" for dataset in args.datasets for metric in ("RMSE", "MAE")]
    write_csv(output / "paper_table.csv", paper_rows, wide_fields)
    if not args.no_plots:
        for dataset in args.datasets:
            plot_dataset(dataset, summary, args.models, output / "figures" / f"fig_baselines_{dataset}.png", args.dpi)


if __name__ == "__main__":
    main()
