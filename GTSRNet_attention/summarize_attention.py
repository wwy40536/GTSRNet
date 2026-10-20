"""Aggregate attention diagnostics and perturbations across datasets and seeds."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Mapping, Sequence, Tuple

import numpy as np


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--no-plot", action="store_true")
    return parser.parse_args()


def read_csvs(root: Path, filename: str):
    rows = []
    for path in sorted(root.glob(f"runs/*/*/{filename}")):
        with path.open("r", encoding="utf-8-sig", newline="") as stream:
            rows.extend(csv.DictReader(stream))
    if not rows:
        raise FileNotFoundError(f"No {filename} files below {root / 'runs'}")
    return rows


def write_csv(path: Path, rows: Sequence[Mapping[str, object]], fields: Sequence[str]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    print(f"WROTE {path}", flush=True)


def mean_std(values):
    array = np.asarray(values, dtype=np.float64)
    return float(array.mean()), float(array.std(ddof=1)) if len(array) > 1 else 0.0


def summarize_attention(rows):
    metrics = (
        "normalized_entropy", "top_mass", "self_mass", "local_3x3_mass",
        "mean_grid_distance", "mean_pairwise_head_js",
    )
    per_seed = defaultdict(lambda: defaultdict(list))
    for row in rows:
        event = row["event"] if row["event"] != "sampled_test" else "all_sampled"
        key = (row["dataset"], int(row["training_seed"]), int(row["layer"]), event)
        for metric in metrics:
            per_seed[key][metric].append(float(row[metric]))
    seed_rows = []
    for key, values in sorted(per_seed.items()):
        dataset, seed, layer, event = key
        for metric in metrics:
            seed_rows.append(
                {"dataset": dataset, "training_seed": seed, "layer": layer, "event": event,
                 "metric": metric, "mean": float(np.mean(values[metric])), "n_samples": len(values[metric])}
            )
    final = []
    grouped = defaultdict(list)
    for row in seed_rows:
        grouped[(row["dataset"], row["layer"], row["event"], row["metric"])].append(row["mean"])
    for key, values in sorted(grouped.items()):
        mean, std = mean_std(values)
        final.append(
            {"dataset": key[0], "layer": key[1], "event": key[2], "metric": key[3],
             "mean": mean, "std_across_seeds": std, "n_seeds": len(values),
             "mean_pm_std": f"{mean:.4f} ± {std:.4f}"}
        )
    return seed_rows, final


def summarize_perturbations(rows):
    # First average random/shuffle repeats within each test window.
    per_sample = defaultdict(lambda: defaultdict(list))
    for row in rows:
        if row["intervention"] == "baseline":
            continue
        key = (
            row["dataset"], int(row["training_seed"]), int(row["sample_index"]),
            row["branch"], row["intervention"], float(row["fraction"]),
        )
        per_sample[key]["delta_rmse"].append(float(row["delta_rmse"]))
        per_sample[key]["delta_mae"].append(float(row["delta_mae"]))
    sample_rows = []
    for key, values in sorted(per_sample.items()):
        sample_rows.append(
            {"dataset": key[0], "training_seed": key[1], "sample_index": key[2],
             "branch": key[3], "intervention": key[4], "fraction": key[5],
             "delta_rmse": float(np.mean(values["delta_rmse"])),
             "delta_mae": float(np.mean(values["delta_mae"]))}
        )
    per_seed = defaultdict(lambda: defaultdict(list))
    for row in sample_rows:
        key = (row["dataset"], row["training_seed"], row["branch"], row["intervention"], row["fraction"])
        per_seed[key]["delta_rmse"].append(row["delta_rmse"])
        per_seed[key]["delta_mae"].append(row["delta_mae"])
    seed_rows = []
    for key, values in sorted(per_seed.items()):
        seed_rows.append(
            {"dataset": key[0], "training_seed": key[1], "branch": key[2],
             "intervention": key[3], "fraction": key[4],
             "delta_rmse": float(np.mean(values["delta_rmse"])),
             "delta_mae": float(np.mean(values["delta_mae"])),
             "n_samples": len(values["delta_rmse"])}
        )
    final = []
    grouped = defaultdict(list)
    for row in seed_rows:
        for metric in ("delta_rmse", "delta_mae"):
            grouped[(row["dataset"], row["branch"], row["intervention"], row["fraction"], metric)].append(row[metric])
    for key, values in sorted(grouped.items()):
        mean, std = mean_std(values)
        final.append(
            {"dataset": key[0], "branch": key[1], "intervention": key[2],
             "fraction": key[3], "metric": key[4], "mean": mean,
             "std_across_seeds": std, "n_seeds": len(values),
             "mean_pm_std": f"{mean:.4f} ± {std:.4f}"}
        )
    return sample_rows, seed_rows, final


def paired_high_random(sample_rows):
    lookup = {(row["dataset"], row["training_seed"], row["sample_index"], row["branch"], row["fraction"], row["intervention"]): row for row in sample_rows}
    per_seed_groups = defaultdict(list)
    for key, high in lookup.items():
        if key[-1] != "mask_high_attention":
            continue
        random_key = key[:-1] + ("mask_random",)
        if random_key not in lookup:
            continue
        random_row = lookup[random_key]
        for metric in ("delta_rmse", "delta_mae"):
            per_seed_groups[(key[0], key[1], key[3], key[4], metric)].append(
                float(high[metric]) - float(random_row[metric])
            )

    seed_rows = []
    for key, values in sorted(per_seed_groups.items()):
        array = np.asarray(values, dtype=np.float64)
        seed_rows.append(
            {
                "dataset": key[0], "training_seed": key[1], "branch": key[2],
                "fraction": key[3], "metric": key[4],
                "high_minus_random": float(array.mean()),
                "win_rate_high_gt_random": float(np.mean(array > 0)),
                "n_samples": len(array),
            }
        )

    # Use a hierarchical paired bootstrap: resample training seeds first and
    # test windows within each selected seed. This preserves the paired design
    # and avoids treating temporally related windows/repeats as independent
    # training runs, which would make the confidence intervals too optimistic.
    groups = defaultdict(lambda: defaultdict(list))
    for key, values in per_seed_groups.items():
        groups[(key[0], key[2], key[3], key[4])][key[1]].extend(values)
    rows = []
    rng = np.random.default_rng(2026)
    for key, values_by_seed in sorted(groups.items()):
        seed_ids = np.asarray(sorted(values_by_seed), dtype=np.int64)
        seed_means = np.asarray(
            [np.mean(values_by_seed[int(seed)]) for seed in seed_ids], dtype=np.float64
        )
        bootstrap_means = []
        for _ in range(5000):
            sampled_seeds = rng.choice(seed_ids, size=len(seed_ids), replace=True)
            sampled_seed_means = []
            for sampled_seed in sampled_seeds:
                sample_values = np.asarray(values_by_seed[int(sampled_seed)], dtype=np.float64)
                sampled_seed_means.append(
                    float(rng.choice(sample_values, size=len(sample_values), replace=True).mean())
                )
            bootstrap_means.append(float(np.mean(sampled_seed_means)))
        boot = np.asarray(bootstrap_means, dtype=np.float64)
        all_differences = np.concatenate(
            [np.asarray(values_by_seed[int(seed)], dtype=np.float64) for seed in seed_ids]
        )
        rows.append(
            {"dataset": key[0], "branch": key[1], "fraction": key[2], "metric": key[3],
             "high_minus_random": float(seed_means.mean()),
             "std_across_seed_means": float(seed_means.std(ddof=1)) if len(seed_means) > 1 else 0.0,
             "ci95_low": float(np.quantile(boot, 0.025)),
             "ci95_high": float(np.quantile(boot, 0.975)),
             "win_rate_high_gt_random": float(np.mean(all_differences > 0)),
             "n_seeds": len(seed_ids), "n_samples_total": len(all_differences),
             "bootstrap": "hierarchical paired: seeds, then windows (5000 resamples)"}
        )
    return seed_rows, rows


def plot_results(summary, output: Path, dpi: int):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = {
        "mask_high_attention": "High-attention mask", "mask_low_attention": "Low-attention mask",
        "mask_random": "Random mask", "mask_local_around_top": "Local-region mask",
        "shuffle_high_positions": "High-attention position shuffle",
    }
    for dataset in sorted({row["dataset"] for row in summary}):
        for branch in ("GTSRNet", "spatial_branch"):
            fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
            for axis, metric in zip(axes, ("delta_rmse", "delta_mae")):
                for intervention, label in labels.items():
                    rows = sorted(
                        (row for row in summary if row["dataset"] == dataset and row["branch"] == branch
                         and row["metric"] == metric and row["intervention"] == intervention),
                        key=lambda row: row["fraction"],
                    )
                    if not rows:
                        continue
                    x = [100 * float(row["fraction"]) for row in rows]
                    y = [float(row["mean"]) for row in rows]
                    e = [float(row["std_across_seeds"]) for row in rows]
                    axis.errorbar(x, y, yerr=e, marker="o", capsize=3, label=label)
                axis.axhline(0, color="black", linewidth=0.7)
                axis.set_xlabel("Perturbed cells (%)")
                axis.set_ylabel(metric.replace("_", " ").upper())
                axis.grid(alpha=0.25, linestyle="--")
            axes[0].legend(frameon=False, fontsize=8)
            fig.suptitle(f"{dataset} — {branch}")
            path = output / "figures" / f"fig_perturbation_{dataset}_{branch}.png"
            path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor="white")
            plt.close(fig)
            print(f"WROTE {path}", flush=True)


def main():
    args = parse_args()
    root = args.results_dir.expanduser().resolve()
    attention = read_csvs(root, "attention_statistics.csv")
    perturbation = read_csvs(root, "perturbation_results.csv")
    attention_seed, attention_summary = summarize_attention(attention)
    sample_rows, perturb_seed, perturb_summary = summarize_perturbations(perturbation)
    comparison_seed, comparisons = paired_high_random(sample_rows)
    summary_dir = root / "summary"
    write_csv(summary_dir / "attention_seed_summary.csv", attention_seed,
              ("dataset", "training_seed", "layer", "event", "metric", "mean", "n_samples"))
    write_csv(summary_dir / "attention_summary.csv", attention_summary,
              ("dataset", "layer", "event", "metric", "mean", "std_across_seeds", "n_seeds", "mean_pm_std"))
    write_csv(summary_dir / "perturbation_seed_summary.csv", perturb_seed,
              ("dataset", "training_seed", "branch", "intervention", "fraction", "delta_rmse", "delta_mae", "n_samples"))
    write_csv(summary_dir / "perturbation_summary.csv", perturb_summary,
              ("dataset", "branch", "intervention", "fraction", "metric", "mean", "std_across_seeds", "n_seeds", "mean_pm_std"))
    write_csv(summary_dir / "high_vs_random_seed.csv", comparison_seed,
              ("dataset", "training_seed", "branch", "fraction", "metric",
               "high_minus_random", "win_rate_high_gt_random", "n_samples"))
    write_csv(summary_dir / "high_vs_random_paired.csv", comparisons,
              ("dataset", "branch", "fraction", "metric", "high_minus_random",
               "std_across_seed_means", "ci95_low", "ci95_high",
               "win_rate_high_gt_random", "n_seeds", "n_samples_total", "bootstrap"))
    if not args.no_plot:
        plot_results(perturb_summary, summary_dir, args.dpi)


if __name__ == "__main__":
    main()
