"""Aggregate dense, low-rank, and no-joint results over random seeds."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Mapping, Sequence, Tuple

import numpy as np


DATASET_LABELS: Mapping[str, str] = {
    "milan_sms": "Milan-SMS",
    "milan_call": "Milan-Call",
    "milan_internet": "Milan-Internet",
    "mobile_nj": "MobileNJ",
    "mobile_bj": "MobileBJ",
}
VARIANT_ORDER = ("GTSRNet", "lowrank_joint", "wo_joint")
VARIANT_LABELS: Mapping[str, str] = {
    "GTSRNet": "Dense Joint",
    "lowrank_joint": "Low-rank Joint",
    "wo_joint": "No Joint",
}
COLORS: Mapping[str, str] = {
    "GTSRNet": "#76558E",
    "lowrank_joint": "#2A9D8F",
    "wo_joint": "#E9C46A",
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Aggregate multiseed joint-encoding comparisons.")
    parser.add_argument("--results-dir", type=Path, default=Path("results_lowrank_joint_r16"))
    parser.add_argument("--seeds", nargs="+", type=int, default=(16, 26, 36))
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--allow-partial", action="store_true")
    return parser


def aggregate(values: Sequence[float]) -> Tuple[float, float]:
    array = np.asarray(values, dtype=np.float64)
    return float(np.mean(array)), float(np.std(array, ddof=1)) if len(array) > 1 else 0.0


def read_results(
    results_dir: Path, seeds: Sequence[int], allow_partial: bool
) -> Dict[str, Dict[str, Dict[str, object]]]:
    raw: Dict[str, Dict[str, List[dict]]] = {
        dataset: {variant: [] for variant in VARIANT_ORDER}
        for dataset in DATASET_LABELS
    }
    missing: List[str] = []
    for seed in seeds:
        for dataset in DATASET_LABELS:
            for variant in VARIANT_ORDER:
                path = results_dir / f"seed_{seed}" / dataset / variant / "metrics.json"
                if not path.is_file():
                    missing.append(f"seed_{seed}/{dataset}/{variant}")
                    continue
                raw[dataset][variant].append(
                    json.loads(path.read_text(encoding="utf-8"))["metrics"]
                )
    if missing and not allow_partial:
        raise FileNotFoundError("Missing metrics for: " + ", ".join(missing))

    results: Dict[str, Dict[str, Dict[str, object]]] = {dataset: {} for dataset in DATASET_LABELS}
    for dataset, variants in raw.items():
        for variant, records in variants.items():
            if not records:
                continue
            rmse_mean, rmse_std = aggregate([float(record["rmse"]) for record in records])
            mae_mean, mae_std = aggregate([float(record["mae"]) for record in records])
            epoch_mean, epoch_std = aggregate([float(record["best_epoch"]) for record in records])
            results[dataset][variant] = {
                "rmse_mean": rmse_mean,
                "rmse_std": rmse_std,
                "mae_mean": mae_mean,
                "mae_std": mae_std,
                "best_epoch_mean": epoch_mean,
                "best_epoch_std": epoch_std,
                "parameters": int(records[0]["parameters"]),
                "joint_rank": records[0].get("joint_rank"),
                "seed_count": len(records),
            }
    return results


def draw_group(
    results: Mapping[str, Mapping[str, Mapping[str, object]]],
    datasets: Sequence[str],
    metric: str,
    output_path: Path,
    dpi: int,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plotted_variants = [
        variant
        for variant in VARIANT_ORDER
        if any(variant in results.get(dataset, {}) for dataset in datasets)
    ]
    available = [
        dataset
        for dataset in datasets
        if all(variant in results.get(dataset, {}) for variant in plotted_variants)
    ]
    if not available or not plotted_variants:
        return
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "DejaVu Serif"],
            "axes.linewidth": 1.0,
            "font.size": 11,
        }
    )
    bar_height = 0.15
    group_gap = 0.42
    centers = np.arange(len(available)) * (len(plotted_variants) * bar_height + group_gap)
    offsets = (
        np.arange(len(plotted_variants)) - (len(plotted_variants) - 1) / 2
    ) * bar_height
    fig, ax = plt.subplots(figsize=(10.8, max(4.8, 1.7 * len(available) + 1.7)), dpi=dpi)
    values_all: List[float] = []
    for index, variant in enumerate(plotted_variants):
        values = [float(results[dataset][variant][f"{metric}_mean"]) for dataset in available]
        errors = [float(results[dataset][variant][f"{metric}_std"]) for dataset in available]
        values_all.extend(value + error for value, error in zip(values, errors))
        bars = ax.barh(
            centers + offsets[index], values, xerr=errors, capsize=2.5,
            height=bar_height * 0.9, color=COLORS[variant], edgecolor="white",
            linewidth=0.55, error_kw={"ecolor": "#333333", "elinewidth": 0.65},
            label=VARIANT_LABELS[variant],
        )
        for bar, value, error in zip(bars, values, errors):
            ax.annotate(
                f"{value:.3f}±{error:.3f}",
                xy=(value + error, bar.get_y() + bar.get_height() / 2), xytext=(5, 0),
                textcoords="offset points", ha="left", va="center", fontsize=8.5,
            )
    maximum = max(values_all)
    ax.set_xlim(0, maximum * 1.32 if maximum > 0 else 1.0)
    ax.set_yticks(centers)
    ax.set_yticklabels([DATASET_LABELS[dataset] for dataset in available], fontsize=13)
    ax.invert_yaxis()
    ax.set_xlabel(f"Overall {metric.upper()} (mean ± std over seeds)", fontsize=13)
    ax.xaxis.grid(True, color="#D8D8D8", linestyle="--", linewidth=0.7, alpha=0.65)
    ax.set_axisbelow(True)
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(
        handles, labels, title="Joint Encoding", ncol=2, loc="upper right",
        frameon=True, fancybox=False, edgecolor="#777777", fontsize=10, title_fontsize=11,
    )
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def write_summary(results_dir: Path, results: Mapping[str, Mapping[str, Mapping[str, object]]]) -> Path:
    path = results_dir / "summary_multiseed.csv"
    fields = (
        "dataset", "variant", "seed_count", "rmse_mean", "rmse_std", "mae_mean", "mae_std",
        "best_epoch_mean", "best_epoch_std", "parameters", "joint_rank",
    )
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for dataset in DATASET_LABELS:
            for variant in VARIANT_ORDER:
                if variant not in results.get(dataset, {}):
                    continue
                writer.writerow({"dataset": dataset, "variant": variant, **results[dataset][variant]})
    return path


def main() -> None:
    args = build_parser().parse_args()
    results_dir = args.results_dir.expanduser().resolve()
    results = read_results(results_dir, args.seeds, args.allow_partial)
    groups: Tuple[Tuple[str, Sequence[str]], ...] = (
        ("milan", ("milan_sms", "milan_call", "milan_internet")),
        ("mobile_bj_nj", ("mobile_bj", "mobile_nj")),
    )
    for group_name, datasets in groups:
        for metric in ("rmse", "mae"):
            output = results_dir / "figures" / f"fig_joint_encoding_{group_name}_{metric}_mean.png"
            draw_group(results, datasets, metric, output, args.dpi)
            if output.is_file():
                print(f"WROTE {output}", flush=True)
    print(f"WROTE {write_summary(results_dir, results)}", flush=True)


if __name__ == "__main__":
    main()
