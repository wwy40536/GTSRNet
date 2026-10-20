"""Create paper-ready embedding ablation tables and multi-seed figures."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Mapping, Sequence, Tuple

import numpy as np


VARIANTS = ("none", "channel", "phase", "channel_phase", "joint")
DISPLAY = {"none": "No embedding", "channel": "Channel", "phase": "Phase", "channel_phase": "Channel+Phase", "joint": "Joint"}


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=Path("results_embedding_reviewer"))
    parser.add_argument("--seeds", nargs="+", type=int, default=(16, 26, 36))
    parser.add_argument("--allow-partial", action="store_true")
    parser.add_argument("--dpi", type=int, default=300)
    return parser


def stats(values: Sequence[float]) -> Tuple[float, float]:
    array = np.asarray(values, dtype=np.float64)
    return float(array.mean()), float(array.std(ddof=1)) if len(array) > 1 else 0.0


def write_csv(path: Path, rows: Sequence[Mapping[str, object]], fields: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader(); writer.writerows(rows)
    print(f"WROTE {path}", flush=True)


def collect(root: Path):
    rows: List[Dict[str, object]] = []
    histories = {}
    for path in sorted(root.glob("*/*/seed_*/metrics.json")):
        dataset, variant = path.parents[2].name, path.parents[1].name
        if variant not in VARIANTS:
            continue
        seed = int(path.parent.name.split("_")[-1])
        payload = json.loads(path.read_text(encoding="utf-8"))
        metrics = payload["metrics"]
        rows.append(
            {
                "dataset": dataset, "embedding": variant, "display": DISPLAY[variant], "seed": seed,
                "rmse": float(metrics["rmse"]), "mae": float(metrics["mae"]),
                "parameters": int(metrics["parameters"]),
                "embedding_parameters": int(metrics["embedding_parameters"]),
                "best_epoch": int(metrics["best_epoch"]),
                "best_val_loss": float(metrics["best_val_loss"]),
                "inference_seconds": float(metrics["inference_seconds"]),
            }
        )
        histories[(dataset, variant, seed)] = payload.get("history", [])
    if not rows:
        raise FileNotFoundError(f"No embedding metrics.json found under {root}")
    return rows, histories


def aggregate(rows):
    groups = defaultdict(list)
    for row in rows: groups[(row["dataset"], row["embedding"])].append(row)
    output = []
    for (dataset, variant), group in sorted(groups.items()):
        rm, rs = stats([row["rmse"] for row in group]); mm, ms = stats([row["mae"] for row in group])
        tm, ts = stats([row["inference_seconds"] for row in group])
        output.append(
            {
                "dataset": dataset, "embedding": variant, "display": DISPLAY[variant], "n_seeds": len(group),
                "rmse_mean": rm, "rmse_sd": rs, "mae_mean": mm, "mae_sd": ms,
                "RMSE (mean ± SD)": f"{rm:.4f} ± {rs:.4f}", "MAE (mean ± SD)": f"{mm:.4f} ± {ms:.4f}",
                "parameters": group[0]["parameters"], "embedding_parameters": group[0]["embedding_parameters"],
                "inference_seconds_mean": tm, "inference_seconds_sd": ts,
            }
        )
    return output


def plot_bars(dataset: str, summary, output: Path, dpi: int) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    lookup = {row["embedding"]: row for row in summary if row["dataset"] == dataset}
    variants = [variant for variant in VARIANTS if variant in lookup]
    x = np.arange(len(variants))
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), constrained_layout=True)
    for axis, metric, title in ((axes[0], "rmse", "RMSE"), (axes[1], "mae", "MAE")):
        means = [lookup[v][f"{metric}_mean"] for v in variants]
        sds = [lookup[v][f"{metric}_sd"] for v in variants]
        bars = axis.bar(x, means, yerr=sds, capsize=4, color="#4F77AA")
        axis.set_xticks(x, [DISPLAY[v] for v in variants], rotation=25, ha="right")
        axis.set_ylabel(title); axis.grid(axis="y", alpha=0.25)
        for bar, value in zip(bars, means):
            axis.text(bar.get_x() + bar.get_width()/2, bar.get_height(), f"{value:.4g}", ha="center", va="bottom", fontsize=8)
    fig.suptitle(f"full_v2 embedding ablation — {dataset} (mean±SD)")
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=dpi, bbox_inches="tight"); plt.close(fig)


def plot_losses(dataset: str, histories, output: Path, dpi: int) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    colors = dict(zip(VARIANTS, ("#777777", "#2878B5", "#55A868", "#C44E52", "#8172B3")))
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), constrained_layout=True)
    for variant in VARIANTS:
        selected = [history for (data, mode, _seed), history in histories.items() if data == dataset and mode == variant]
        for axis, metric, title in ((axes[0], "train_loss", "Training loss"), (axes[1], "val_loss", "Validation loss")):
            for history in selected:
                if history:
                    axis.plot([x["epoch"] for x in history], [x[metric] for x in history], color=colors[variant], alpha=0.28, linewidth=0.8)
            if selected:
                shortest = min(len(history) for history in selected)
                if shortest:
                    matrix = np.asarray([[x[metric] for x in history[:shortest]] for history in selected])
                    axis.plot(np.arange(1, shortest + 1), matrix.mean(axis=0), color=colors[variant], linewidth=1.8, label=DISPLAY[variant])
            axis.set_title(title); axis.set_xlabel("Epoch"); axis.set_ylabel("Normalized loss"); axis.grid(alpha=0.25)
    for axis in axes: axis.legend(fontsize=8)
    fig.suptitle(f"full_v2 embedding convergence — {dataset}")
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=dpi, bbox_inches="tight"); plt.close(fig)


def main() -> None:
    args = build_parser().parse_args(); root = args.results_dir.expanduser().resolve()
    rows, histories = collect(root)
    datasets = sorted({row["dataset"] for row in rows})
    present = {(row["dataset"], row["embedding"], row["seed"]) for row in rows}
    expected = {(dataset, variant, seed) for dataset in datasets for variant in VARIANTS for seed in args.seeds}
    missing = sorted(expected - present)
    if missing and not args.allow_partial:
        raise RuntimeError(f"Missing {len(missing)} expected runs; first entries: {missing[:10]}")
    summary = aggregate(rows); output = root / "summary"
    write_csv(output / "all_runs.csv", rows, tuple(rows[0].keys()))
    write_csv(output / "embedding_summary.csv", summary, tuple(summary[0].keys()))
    for dataset in datasets:
        plot_bars(dataset, summary, output / "figures" / f"embedding_{dataset}.png", args.dpi)
        plot_losses(dataset, histories, output / "figures" / f"loss_{dataset}.png", args.dpi)
    print(f"DONE: {output}", flush=True)


if __name__ == "__main__":
    main()
