"""Collect ablation metrics and draw paper-style horizontal grouped bars."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np


DATASET_LABELS: Mapping[str, str] = {
    "milan_sms": "Milan-SMS",
    "milan_call": "Milan-Call",
    "milan_internet": "Milan-Internet",
    "mobile_bj": "MobileBJ",
    "mobile_nj": "MobileNJ",
}
VARIANT_ORDER = ("wo_s", "wo_t", "wo_sthe", "wo_seg", "wo_g", "GTSRNet")
VARIANT_LABELS: Mapping[str, str] = {
    "wo_s": "w/o S",
    "wo_t": "w/o T",
    "wo_sthe": "w/o STHE",
    "wo_seg": "w/o Seg",
    "wo_g": "w/o G",
    "full": "Uncalibrated gate",
    "GTSRNet": "GTSRNet",
}
COLORS: Mapping[str, str] = {
    "wo_s": "#8FAF80",
    "wo_t": "#8CA5AE",
    "wo_sthe": "#D9BE86",
    "wo_seg": "#CD9675",
    "wo_g": "#8FBCC3",
    "full": "#365C82",
    "GTSRNet": "#76558E",
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Plot completed ablation metrics.")
    parser.add_argument("--results-dir", type=Path, default=Path("results_ablation"))
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--allow-partial", action="store_true")
    return parser


def read_results(results_dir: Path, allow_partial: bool) -> Dict[str, Dict[str, dict]]:
    collected: Dict[str, Dict[str, dict]] = {}
    missing: List[str] = []
    for dataset in DATASET_LABELS:
        collected[dataset] = {}
        for variant in VARIANT_ORDER:
            path = results_dir / dataset / variant / "metrics.json"
            if not path.is_file():
                missing.append(f"{dataset}/{variant}")
                continue
            payload = json.loads(path.read_text(encoding="utf-8"))
            collected[dataset][variant] = payload["metrics"]
    if missing and not allow_partial:
        raise FileNotFoundError(
            "Cannot draw complete figures; missing metrics for: " + ", ".join(missing)
        )
    return collected


def value_text(value: float, group: Sequence[str]) -> str:
    return f"{value:.3f}" if all(name.startswith("mobile_") for name in group) else f"{value:.3f}"


def draw_group(
    results: Mapping[str, Mapping[str, dict]],
    datasets: Sequence[str],
    metric: str,
    output_path: Path,
    dpi: int,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    available = [
        dataset
        for dataset in datasets
        if results.get(dataset)
        and all(variant in results[dataset] for variant in VARIANT_ORDER)
    ]
    if not available:
        return

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "DejaVu Serif"],
            "axes.linewidth": 1.0,
            "font.size": 11,
        }
    )
    figure_height = max(4.8, 1.75 * len(available) + 1.7)
    fig, ax = plt.subplots(figsize=(10.8, figure_height), dpi=dpi)

    bar_height = 0.12
    group_gap = 0.34
    group_centers = np.arange(len(available)) * (len(VARIANT_ORDER) * bar_height + group_gap)
    offsets = (np.arange(len(VARIANT_ORDER)) - (len(VARIANT_ORDER) - 1) / 2) * bar_height
    all_values: List[float] = []
    for variant_index, variant in enumerate(VARIANT_ORDER):
        values = [float(results[dataset][variant][metric]) for dataset in available]
        all_values.extend(values)
        bars = ax.barh(
            group_centers + offsets[variant_index],
            values,
            height=bar_height * 0.92,
            color=COLORS[variant],
            edgecolor="white",
            linewidth=0.55,
            label=VARIANT_LABELS[variant],
        )
        for bar, value in zip(bars, values):
            ax.annotate(
                value_text(value, available),
                xy=(value, bar.get_y() + bar.get_height() / 2),
                xytext=(5, 0),
                textcoords="offset points",
                ha="left",
                va="center",
                fontsize=9,
            )

    maximum = max(all_values)
    ax.set_xlim(0, maximum * 1.30 if maximum > 0 else 1.0)
    ax.set_yticks(group_centers)
    ax.set_yticklabels([DATASET_LABELS[name] for name in available], fontsize=13)
    ax.invert_yaxis()
    ax.set_xlabel(f"Overall {metric.upper()} Values", fontsize=13)
    ax.xaxis.grid(True, color="#D8D8D8", linestyle="--", linewidth=0.7, alpha=0.65)
    ax.set_axisbelow(True)
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(
        handles,
        labels,
        title="Model Variants",
        ncol=2,
        loc="upper right",
        frameon=True,
        fancybox=False,
        edgecolor="#777777",
        fontsize=10,
        title_fontsize=11,
    )
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def write_combined_csv(results_dir: Path, results: Mapping[str, Mapping[str, dict]]) -> Path:
    output = results_dir / "summary_all.csv"
    fields = (
        "dataset",
        "variant",
        "rmse",
        "mae",
        "parameters",
        "best_epoch",
        "best_val_loss",
        "fusion_alpha",
        "inference_seconds",
    )
    with output.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for dataset in DATASET_LABELS:
            for variant in VARIANT_ORDER:
                if variant not in results.get(dataset, {}):
                    continue
                row = dict(results[dataset][variant])
                row["dataset"] = dataset
                row["variant"] = variant
                writer.writerow(row)
    return output


def main() -> None:
    args = build_parser().parse_args()
    results_dir = args.results_dir.expanduser().resolve()
    results = read_results(results_dir, args.allow_partial)
    figures_dir = results_dir / "figures"
    groups: Tuple[Tuple[str, Sequence[str]], ...] = (
        ("milan", ("milan_sms", "milan_call", "milan_internet")),
        ("mobile_bj_nj", ("mobile_bj", "mobile_nj")),
    )
    for group_name, datasets in groups:
        for metric in ("rmse", "mae"):
            output = figures_dir / f"fig_ablation_{group_name}_{metric}.png"
            draw_group(results, datasets, metric, output, args.dpi)
            print(f"WROTE {output}", flush=True)
    summary = write_combined_csv(results_dir, results)
    print(f"WROTE {summary}", flush=True)


if __name__ == "__main__":
    main()
