"""Aggregate strict ablation results from multiple seed-specific directories."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Mapping, Sequence, Tuple

import numpy as np


DATASETS = ("milan_sms", "milan_call", "milan_internet")
VARIANTS = ("wo_s", "wo_t", "wo_sthe", "wo_seg", "fixed_avg", "wo_g", "GTSRNet")
LABELS = {
    "wo_s": "w/o S",
    "wo_t": "w/o T",
    "wo_sthe": "w/o STHE",
    "wo_seg": "w/o Seg",
    "fixed_avg": "Fixed Average",
    "wo_g": "Concat Fusion",
    "GTSRNet": "GTSRNet",
}


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--result-dirs", nargs="+", type=Path, required=True)
    result.add_argument("--output-dir", type=Path, required=True)
    result.add_argument("--datasets", nargs="+", choices=DATASETS, default=DATASETS)
    result.add_argument("--variants", nargs="+", choices=VARIANTS, default=VARIANTS)
    result.add_argument("--dpi", type=int, default=300)
    return result


def write_csv(path: Path, rows: Sequence[Mapping[str, object]], fields: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    print(f"WROTE {path}", flush=True)


def collect(
    result_dirs: Sequence[Path], datasets: Sequence[str], variants: Sequence[str]
) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    missing: List[str] = []
    for root_arg in result_dirs:
        root = root_arg.expanduser().resolve()
        for dataset in datasets:
            for variant in variants:
                path = root / dataset / variant / "metrics.json"
                if not path.is_file():
                    missing.append(str(path))
                    continue
                payload = json.loads(path.read_text(encoding="utf-8"))
                metrics, config = payload["metrics"], payload["config"]
                rows.append(
                    {
                        "seed": int(config["seed"]),
                        "dataset": dataset,
                        "variant": variant,
                        "label": LABELS[variant],
                        "rmse": float(metrics["rmse"]),
                        "mae": float(metrics["mae"]),
                        "parameters": int(metrics["parameters"]),
                        "best_epoch": int(metrics["best_epoch"]),
                        "metrics_path": str(path),
                    }
                )
    if missing:
        raise FileNotFoundError("Missing expected runs:\n  " + "\n  ".join(missing))
    expected = len(result_dirs) * len(datasets) * len(variants)
    if len(rows) != expected:
        raise RuntimeError(f"Expected {expected} runs, collected {len(rows)}")
    return rows


def aggregate(rows: Sequence[Mapping[str, object]]) -> List[Dict[str, object]]:
    grouped: Dict[Tuple[str, str], List[Mapping[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["dataset"]), str(row["variant"]))].append(row)
    output: List[Dict[str, object]] = []
    for (dataset, variant), group in sorted(grouped.items()):
        rmse = np.asarray([float(row["rmse"]) for row in group], dtype=np.float64)
        mae = np.asarray([float(row["mae"]) for row in group], dtype=np.float64)
        output.append(
            {
                "dataset": dataset,
                "variant": variant,
                "label": LABELS[variant],
                "n_seeds": len(group),
                "seeds": "/".join(str(row["seed"]) for row in sorted(group, key=lambda item: int(item["seed"]))),
                "rmse_mean": float(rmse.mean()),
                "rmse_std": float(rmse.std(ddof=1)) if len(rmse) > 1 else 0.0,
                "mae_mean": float(mae.mean()),
                "mae_std": float(mae.std(ddof=1)) if len(mae) > 1 else 0.0,
                "RMSE (mean ± std)": f"{rmse.mean():.4f} ± {(rmse.std(ddof=1) if len(rmse) > 1 else 0.0):.4f}",
                "MAE (mean ± std)": f"{mae.mean():.4f} ± {(mae.std(ddof=1) if len(mae) > 1 else 0.0):.4f}",
                "parameters": int(round(np.mean([int(row["parameters"]) for row in group]))),
            }
        )
    return output


def draw(summary: Sequence[Mapping[str, object]], datasets: Sequence[str], variants: Sequence[str], output: Path, dpi: int) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    lookup = {(str(row["dataset"]), str(row["variant"])): row for row in summary}
    figure, axes = plt.subplots(len(datasets), 2, figsize=(13, 4.2 * len(datasets)), constrained_layout=True)
    if len(datasets) == 1:
        axes = np.asarray([axes])
    x = np.arange(len(variants))
    colors = ["#C35A4A" if variant == "GTSRNet" else "#4F77AA" for variant in variants]
    for row_index, dataset in enumerate(datasets):
        for column, metric in enumerate(("rmse", "mae")):
            axis = axes[row_index, column]
            means = [float(lookup[(dataset, variant)][f"{metric}_mean"]) for variant in variants]
            stds = [float(lookup[(dataset, variant)][f"{metric}_std"]) for variant in variants]
            axis.bar(x, means, yerr=stds, capsize=3, color=colors, edgecolor="white")
            axis.set_xticks(x, [LABELS[variant] for variant in variants], rotation=28, ha="right")
            axis.set_ylabel(metric.upper())
            axis.set_title(dataset.replace("_", " ").title())
            axis.grid(axis="y", linestyle="--", alpha=0.25)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(figure)
    print(f"WROTE {output}", flush=True)


def main() -> None:
    args = parser().parse_args()
    rows = collect(args.result_dirs, args.datasets, args.variants)
    summary = aggregate(rows)
    output = args.output_dir.expanduser().resolve()
    write_csv(
        output / "all_runs.csv",
        rows,
        ("seed", "dataset", "variant", "label", "rmse", "mae", "parameters", "best_epoch", "metrics_path"),
    )
    write_csv(
        output / "summary_mean_std.csv",
        summary,
        ("dataset", "variant", "label", "n_seeds", "seeds", "rmse_mean", "rmse_std", "mae_mean", "mae_std", "RMSE (mean ± std)", "MAE (mean ± std)", "parameters"),
    )
    draw(summary, args.datasets, args.variants, output / "fig_ablation_multiseed.png", args.dpi)


if __name__ == "__main__":
    main()
