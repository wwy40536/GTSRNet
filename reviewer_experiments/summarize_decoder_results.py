"""Summarize decoder accuracy, convergence, stability, and noise propagation."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Mapping, Sequence, Tuple

import numpy as np


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--results-dir", type=Path, default=Path("results_decoder_reviewer"))
    value.add_argument("--seeds", nargs="+", type=int, default=(45, 46, 47, 48, 49))
    value.add_argument("--allow-partial", action="store_true")
    value.add_argument("--dpi", type=int, default=300)
    return value


def mean_sd(values: Sequence[float]) -> Tuple[float, float]:
    array = np.asarray(values, dtype=np.float64)
    return float(array.mean()), float(array.std(ddof=1)) if len(array) > 1 else 0.0


def write_csv(path: Path, rows: Sequence[Mapping[str, object]], fields: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    print(f"WROTE {path}", flush=True)


def collect(root: Path):
    runs: List[Dict[str, object]] = []
    histories: Dict[Tuple[str, int, str, int], List[Dict[str, float]]] = {}
    noises: List[Dict[str, object]] = []
    for path in sorted(root.glob("*/?mf/H*/seed_*/summary.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        dataset = path.parents[3].name
        decoder = path.parents[2].name
        horizon = int(path.parents[1].name[1:])
        seed = int(path.parent.name.split("_")[-1])
        metrics = payload["metrics"]
        runs.append(
            {
                "dataset": dataset,
                "decoder": decoder,
                "horizon": horizon,
                "seed": seed,
                "rmse": float(metrics["rmse"]),
                "mae": float(metrics["mae"]),
                "first_rmse": float(metrics["first_rmse"]),
                "last_rmse": float(metrics["last_rmse"]),
                "growth_ratio": float(metrics["rmse_growth_ratio"]),
                "parameters": int(payload["parameters"]),
                "latency_ms": float(payload["latency_ms_per_batch"]),
                "peak_cuda_memory_mb": float(payload["peak_cuda_memory_mb"]),
                "best_epoch": int(payload["best_epoch"]),
                "baseline": payload.get("config", {}).get("baseline", "last"),
                "rmse_h": metrics["rmse_h"],
                "mae_h": metrics["mae_h"],
            }
        )
        history_path = path.parent / "training_history.json"
        if history_path.is_file():
            histories[(dataset, horizon, decoder, seed)] = json.loads(
                history_path.read_text(encoding="utf-8")
            )
        noise_path = path.parent / "noise_sweep.json"
        if noise_path.is_file():
            for item in json.loads(noise_path.read_text(encoding="utf-8")):
                noises.append(
                    {
                        "dataset": dataset,
                        "decoder": decoder,
                        "horizon": horizon,
                        "seed": seed,
                        **item,
                    }
                )
    if not runs:
        raise FileNotFoundError(f"No decoder summary.json found under {root}")
    return runs, histories, noises


def aggregate(runs: Sequence[Mapping[str, object]]) -> List[Dict[str, object]]:
    groups = defaultdict(list)
    for row in runs:
        groups[(row["dataset"], row["horizon"], row["decoder"])].append(row)
    output = []
    for (dataset, horizon, decoder), rows in sorted(groups.items()):
        result: Dict[str, object] = {
            "dataset": dataset, "horizon": horizon, "decoder": decoder, "n_seeds": len(rows)
        }
        for metric in ("rmse", "mae", "first_rmse", "last_rmse", "growth_ratio", "latency_ms", "peak_cuda_memory_mb"):
            mean, sd = mean_sd([float(row[metric]) for row in rows])
            result[f"{metric}_mean"] = mean
            result[f"{metric}_sd"] = sd
        output.append(result)
    return output


def plot_horizon(dataset: str, horizon: int, rows, output: Path, dpi: int) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.3), constrained_layout=True)
    for decoder, color in (("pmf", "#2878B5"), ("rmf", "#F07C2B")):
        selected = [row for row in rows if row["dataset"] == dataset and row["horizon"] == horizon and row["decoder"] == decoder]
        if not selected:
            continue
        for axis, metric, label in ((axes[0], "rmse_h", "RMSE"), (axes[1], "mae_h", "MAE")):
            matrix = np.asarray([row[metric] for row in selected], dtype=np.float64)
            steps = np.arange(1, matrix.shape[1] + 1)
            for values in matrix:
                axis.plot(steps, values, color=color, alpha=0.16, linewidth=0.8)
            mean = matrix.mean(axis=0)
            sd = matrix.std(axis=0, ddof=1) if len(matrix) > 1 else np.zeros_like(mean)
            axis.plot(steps, mean, color=color, linewidth=2.2, label=decoder.upper())
            axis.fill_between(steps, mean - sd, mean + sd, color=color, alpha=0.15)
            axis.set_xlabel("Forecasting step")
            axis.set_ylabel(label)
            axis.grid(alpha=0.25)
            axis.legend()
    for boundary in range(6, horizon, 6):
        for axis in axes:
            axis.axvline(boundary + 0.5, color="gray", linestyle="--", alpha=0.35)
    baseline = next((str(row["baseline"]) for row in rows if row["dataset"] == dataset), "last")
    fig.suptitle(f"{dataset}: PMF vs RMF, H={horizon}, {baseline} baseline (individual seeds + mean±SD)")
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def plot_convergence(dataset: str, horizon: int, histories, output: Path, dpi: int) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.3), constrained_layout=True)
    for decoder, color in (("pmf", "#2878B5"), ("rmf", "#F07C2B")):
        selected = [history for (data, h, mode, _seed), history in histories.items() if data == dataset and h == horizon and mode == decoder]
        for axis, metric, title in ((axes[0], "train_loss", "Training loss"), (axes[1], "val_loss", "Validation loss")):
            for history in selected:
                epochs = [item["epoch"] for item in history]
                values = [item[metric] for item in history]
                axis.plot(epochs, values, color=color, alpha=0.35, linewidth=1.0)
            axis.set_title(title)
            axis.set_xlabel("Epoch")
            axis.set_ylabel("Normalized loss")
            axis.grid(alpha=0.25)
            handles = [
                plt.Line2D([0], [0], color="#2878B5", label="PMF seeds"),
                plt.Line2D([0], [0], color="#F07C2B", label="RMF seeds"),
            ]
            axis.legend(handles=handles)
    fig.suptitle(f"{dataset}: multi-seed convergence, H={horizon}")
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def plot_noise(dataset: str, horizon: int, noises, output: Path, dpi: int) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.3), constrained_layout=True)
    for decoder, color in (("pmf", "#2878B5"), ("rmf", "#F07C2B")):
        selected = [row for row in noises if row["dataset"] == dataset and row["horizon"] == horizon and row["decoder"] == decoder]
        levels = sorted({float(row["noise_std"]) for row in selected})
        for axis, metric, title in ((axes[0], "delta_rmse", "Output change RMSE"), (axes[1], "rmse", "Prediction RMSE")):
            means, sds = [], []
            for level in levels:
                values = [float(row[metric]) for row in selected if float(row["noise_std"]) == level]
                mean, sd = mean_sd(values)
                means.append(mean)
                sds.append(sd)
            axis.errorbar(levels, means, yerr=sds, marker="o", capsize=3, color=color, label=decoder.upper())
            axis.set_xlabel("Feedback noise std (normalized scale)")
            axis.set_ylabel(title)
            axis.grid(alpha=0.25)
            axis.legend()
    fig.suptitle(f"{dataset}: feedback perturbation, H={horizon}")
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parser().parse_args()
    root = args.results_dir.expanduser().resolve()
    runs, histories, noises = collect(root)
    present = {(row["dataset"], row["horizon"], row["decoder"], row["seed"]) for row in runs}
    datasets = sorted({row["dataset"] for row in runs})
    horizons = sorted({int(row["horizon"]) for row in runs})
    expected = {(d, h, mode, seed) for d in datasets for h in horizons for mode in ("pmf", "rmf") for seed in args.seeds}
    missing = sorted(expected - present)
    if missing and not args.allow_partial:
        raise RuntimeError(f"Missing {len(missing)} expected runs; first entries: {missing[:10]}")
    baselines = sorted({str(row["baseline"]) for row in runs})
    if len(baselines) > 1:
        raise RuntimeError(f"Mixed output baselines in {root}: {baselines}")
    summary = aggregate(runs)
    output = root / "summary"
    flat_runs = [{key: value for key, value in row.items() if key not in ("rmse_h", "mae_h")} for row in runs]
    write_csv(output / "all_runs.csv", flat_runs, tuple(flat_runs[0].keys()))
    write_csv(output / "decoder_summary.csv", summary, tuple(summary[0].keys()))
    if noises:
        flat_noise = [{key: value for key, value in row.items() if not key.endswith("_h")} for row in noises]
        write_csv(output / "noise_all_runs.csv", flat_noise, tuple(flat_noise[0].keys()))
    for dataset in datasets:
        for horizon in horizons:
            plot_horizon(dataset, horizon, runs, output / "figures" / f"horizon_{dataset}_H{horizon}.png", args.dpi)
            plot_convergence(dataset, horizon, histories, output / "figures" / f"loss_{dataset}_H{horizon}.png", args.dpi)
            if noises:
                plot_noise(dataset, horizon, noises, output / "figures" / f"noise_{dataset}_H{horizon}.png", args.dpi)
    print(f"DONE: {output}", flush=True)


if __name__ == "__main__":
    main()
