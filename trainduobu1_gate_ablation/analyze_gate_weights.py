"""Summarize fusion performance and export horizon-wise gate distributions.

Alpha is the weight assigned to the calibrated spatial branch; ``1-alpha`` is
the temporal-branch weight.  Statistics are computed on the chronological test
split using the best validation checkpoint saved for each run.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Mapping, MutableMapping, Sequence, Tuple

import numpy as np
import torch


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from train_trainduobu1_v2_ablation import build_loaders, make_model  # noqa: E402


DATASETS: Tuple[Tuple[str, str], ...] = (
    ("milan_sms", "Milan-SMS"),
    ("milan_call", "Milan-Call"),
    ("milan_internet", "Milan-Internet"),
    ("mobile_bj", "MobileBJ"),
    ("mobile_nj", "MobileNJ"),
)
VARIANTS: Tuple[Tuple[str, str], ...] = (
    ("fixed_avg", "Fixed Average"),
    ("horizon_static", "Horizon-static"),
    ("full_v2_aux0", "Dynamic, λ=0"),
    ("full_v2", "Dynamic, λ=0.1"),
)
COLORS: Mapping[str, str] = {
    "fixed_avg": "#9E9E9E",
    "horizon_static": "#D9A441",
    "full_v2_aux0": "#D97962",
    "full_v2": "#2F5597",
}
PATH_FIELDS = ("project_root", "data", "feature_path", "output_dir")
PROTOCOL_FIELDS = (
    "batch_size", "epochs", "learning_rate", "loss", "branch_aux_weight",
    "e_layers", "seg_len", "split_mode", "resolved_split_gap", "close_size",
    "predict_size", "patience", "milan_loader",
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=SCRIPT_DIR.parent)
    parser.add_argument("--seeds", nargs="+", type=int, required=True)
    parser.add_argument(
        "--result-pattern",
        default="results_ablation_strict_seed{seed}_bs16",
        help="Result directory pattern under project root; must include {seed}.",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=tuple(name for name, _ in DATASETS),
        default=("milan_sms", "milan_call", "milan_internet"),
    )
    parser.add_argument(
        "--variants",
        nargs="+",
        choices=tuple(name for name, _ in VARIANTS),
        default=tuple(name for name, _ in VARIANTS),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument(
        "--sample-per-batch-horizon",
        type=int,
        default=2000,
        help="Random alpha values retained per batch/horizon for quantiles and histograms.",
    )
    parser.add_argument(
        "--max-samples-per-horizon",
        type=int,
        default=100000,
        help="Maximum retained alpha values per dataset/variant/horizon.",
    )
    return parser


def resolve_under(root: Path, value: Path) -> Path:
    return value.expanduser().resolve() if value.is_absolute() else (root / value).resolve()


def read_payload(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def namespace_from_config(config: Mapping[str, object], project_root: Path):
    namespace = argparse.Namespace(**dict(config))
    namespace.project_root = project_root
    for field in PATH_FIELDS:
        value = getattr(namespace, field, None)
        if value in (None, "", "None"):
            setattr(namespace, field, None)
        elif field != "project_root":
            setattr(namespace, field, Path(str(value)))
    return namespace


class DistributionAccumulator:
    def __init__(self) -> None:
        self.count = 0
        self.total = 0.0
        self.total_square = 0.0
        self.minimum = float("inf")
        self.maximum = float("-inf")
        self.sample_chunks: List[np.ndarray] = []
        self.sample_means: List[np.ndarray] = []
        self.within_sample_stds: List[np.ndarray] = []

    def update(
        self,
        values: np.ndarray,
        rng: np.random.Generator,
        sample_per_batch: int,
    ) -> None:
        """Update from alpha values shaped [batch, cells] for one horizon."""
        values = np.asarray(values, dtype=np.float64)
        flat = values.reshape(-1)
        self.count += int(flat.size)
        self.total += float(flat.sum(dtype=np.float64))
        self.total_square += float(np.square(flat, dtype=np.float64).sum(dtype=np.float64))
        self.minimum = min(self.minimum, float(flat.min()))
        self.maximum = max(self.maximum, float(flat.max()))
        self.sample_means.append(values.mean(axis=1))
        self.within_sample_stds.append(values.std(axis=1, ddof=0))
        take = min(sample_per_batch, flat.size)
        if take == flat.size:
            sampled = flat.copy()
        else:
            sampled = flat[rng.choice(flat.size, size=take, replace=False)]
        self.sample_chunks.append(sampled.astype(np.float32, copy=False))

    def retained(self, rng: np.random.Generator, maximum: int) -> np.ndarray:
        values = np.concatenate(self.sample_chunks) if self.sample_chunks else np.empty(0)
        if values.size > maximum:
            values = values[rng.choice(values.size, size=maximum, replace=False)]
        return values.astype(np.float64, copy=False)

    def summary(self, retained: np.ndarray) -> Dict[str, float]:
        mean = self.total / self.count
        variance = max(self.total_square / self.count - mean * mean, 0.0)
        sample_means = np.concatenate(self.sample_means)
        within_stds = np.concatenate(self.within_sample_stds)
        quantiles = np.quantile(retained, (0.05, 0.25, 0.50, 0.75, 0.95))
        return {
            "count": self.count,
            "mean": mean,
            "std_all": math.sqrt(variance),
            "sample_mean_std": float(sample_means.std(ddof=1)) if sample_means.size > 1 else 0.0,
            "mean_within_sample_std": float(within_stds.mean()),
            "min": self.minimum,
            "q05": float(quantiles[0]),
            "q25": float(quantiles[1]),
            "median": float(quantiles[2]),
            "q75": float(quantiles[3]),
            "q95": float(quantiles[4]),
            "max": self.maximum,
        }


def write_csv(path: Path, rows: Sequence[Mapping[str, object]], fields: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    print(f"WROTE {path}", flush=True)


def collect_alpha(
    model: torch.nn.Module,
    test_loader,
    device: torch.device,
    accumulators: Sequence[DistributionAccumulator],
    rng: np.random.Generator,
    sample_per_batch: int,
) -> None:
    if not hasattr(model, "fusion") or not hasattr(model.fusion, "alpha"):
        raise TypeError(f"{model.variant} does not expose a convex fusion alpha")
    model.eval()
    with torch.no_grad():
        for x, phase, _target in test_loader:
            x = x.to(device=device, dtype=torch.float32, non_blocking=True)
            phase = phase.to(device=device, dtype=torch.long, non_blocking=True)
            spatial = model.spatial_branch(x, phase).reshape(x.shape[0], model.t_out, -1)
            temporal = model.temporal_branch(x)
            alpha = model.fusion.alpha(spatial, temporal).detach().cpu().numpy()
            for horizon, accumulator in enumerate(accumulators):
                accumulator.update(alpha[:, horizon, :], rng, sample_per_batch)


def aggregate_performance(rows: Sequence[Mapping[str, object]]) -> List[Dict[str, object]]:
    grouped: MutableMapping[Tuple[str, str], List[Mapping[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["dataset"]), str(row["variant"]))].append(row)
    output: List[Dict[str, object]] = []
    for dataset, dataset_label in DATASETS:
        for variant, variant_label in VARIANTS:
            group = grouped.get((dataset, variant))
            if not group:
                continue
            rmse = np.asarray([float(row["rmse"]) for row in group])
            mae = np.asarray([float(row["mae"]) for row in group])
            output.append(
                {
                    "dataset": dataset,
                    "dataset_label": dataset_label,
                    "variant": variant,
                    "variant_label": variant_label,
                    "n_seeds": len(group),
                    "seeds": "/".join(str(row["seed"]) for row in sorted(group, key=lambda x: int(x["seed"]))),
                    "rmse_mean": float(rmse.mean()),
                    "rmse_std": float(rmse.std(ddof=1)) if rmse.size > 1 else 0.0,
                    "mae_mean": float(mae.mean()),
                    "mae_std": float(mae.std(ddof=1)) if mae.size > 1 else 0.0,
                }
            )
    return output


def plot_performance(
    summary: Sequence[Mapping[str, object]],
    datasets: Sequence[str],
    variants: Sequence[str],
    output_dir: Path,
    dpi: int,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    dataset_labels = dict(DATASETS)
    variant_labels = dict(VARIANTS)
    lookup = {(str(r["dataset"]), str(r["variant"])): r for r in summary}
    plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"]})
    for metric in ("rmse", "mae"):
        fig, ax = plt.subplots(figsize=(10.8, max(4.8, 1.65 * len(datasets) + 1.8)), dpi=dpi)
        height = 0.16
        gap = 0.42
        centers = np.arange(len(datasets)) * (len(variants) * height + gap)
        offsets = (np.arange(len(variants)) - (len(variants) - 1) / 2) * height
        maximum = 0.0
        for index, variant in enumerate(variants):
            means = np.asarray([float(lookup[(d, variant)][f"{metric}_mean"]) for d in datasets])
            stds = np.asarray([float(lookup[(d, variant)][f"{metric}_std"]) for d in datasets])
            maximum = max(maximum, float(means.max()))
            bars = ax.barh(
                centers + offsets[index], means, xerr=stds, height=height * 0.9,
                color=COLORS[variant], edgecolor="white", linewidth=0.5,
                error_kw={"ecolor": "#444444", "elinewidth": 0.8, "capsize": 2},
                label=variant_labels[variant],
            )
            for bar, value in zip(bars, means):
                ax.annotate(f"{value:.3f}", (value, bar.get_y() + bar.get_height() / 2),
                            xytext=(4, 0), textcoords="offset points", va="center", fontsize=8)
        ax.set_xlim(0, maximum * 1.28)
        ax.set_yticks(centers)
        ax.set_yticklabels([dataset_labels[d] for d in datasets], fontsize=12)
        ax.invert_yaxis()
        ax.set_xlabel(metric.upper(), fontsize=12)
        ax.xaxis.grid(True, linestyle="--", alpha=0.35)
        ax.set_axisbelow(True)
        ax.legend(title="Fusion strategy", ncol=2, loc="upper right", frameon=True)
        fig.tight_layout()
        path = output_dir / f"fig_gate_ablation_{metric}.png"
        fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor="white")
        fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
        plt.close(fig)
        print(f"WROTE {path}", flush=True)


def plot_alpha(
    alpha_rows: Sequence[Mapping[str, object]],
    samples: Mapping[Tuple[str, str, int], np.ndarray],
    datasets: Sequence[str],
    variants: Sequence[str],
    output_dir: Path,
    dpi: int,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    dataset_labels = dict(DATASETS)
    variant_labels = dict(VARIANTS)
    lookup = {(str(r["dataset"]), str(r["variant"]), int(r["horizon"])): r for r in alpha_rows}
    plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"]})
    for dataset in datasets:
        horizons = sorted({int(r["horizon"]) for r in alpha_rows if r["dataset"] == dataset})
        x = np.asarray(horizons)
        fig, ax = plt.subplots(figsize=(9.4, 5.4), dpi=dpi)
        for variant in variants:
            means = np.asarray([float(lookup[(dataset, variant, h)]["mean"]) for h in horizons])
            q25 = np.asarray([float(lookup[(dataset, variant, h)]["q25"]) for h in horizons])
            q75 = np.asarray([float(lookup[(dataset, variant, h)]["q75"]) for h in horizons])
            style = "--" if variant == "fixed_avg" else "-"
            ax.plot(x, means, style, marker="o", markersize=3.5, linewidth=1.8,
                    color=COLORS[variant], label=variant_labels[variant])
            if variant in ("full_v2", "full_v2_aux0"):
                ax.fill_between(x, q25, q75, color=COLORS[variant], alpha=0.16)
        ax.set_ylim(0.0, 1.0)
        ax.set_xticks(x)
        ax.set_xlabel("Forecast horizon")
        ax.set_ylabel("Spatial-branch weight α")
        ax.set_title(dataset_labels[dataset])
        ax.grid(True, linestyle="--", alpha=0.3)
        ax.legend(ncol=2, frameon=True)
        fig.tight_layout()
        path = output_dir / f"fig_gate_alpha_horizons_{dataset}.png"
        fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor="white")
        fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
        plt.close(fig)

        chosen = sorted(set((horizons[0], horizons[len(horizons) // 2], horizons[-1])))
        fig, axes = plt.subplots(1, len(chosen), figsize=(4.6 * len(chosen), 4.1), dpi=dpi, sharey=True)
        axes = np.atleast_1d(axes)
        for axis, horizon in zip(axes, chosen):
            for variant in ("full_v2_aux0", "full_v2"):
                if variant not in variants:
                    continue
                values = samples[(dataset, variant, horizon)]
                axis.hist(values, bins=40, range=(0, 1), density=True, histtype="step",
                          linewidth=1.8, color=COLORS[variant], label=variant_labels[variant])
            if "horizon_static" in variants:
                static_mean = float(lookup[(dataset, "horizon_static", horizon)]["mean"])
                axis.axvline(static_mean, color=COLORS["horizon_static"], linewidth=1.8,
                             label=variant_labels["horizon_static"])
            axis.axvline(0.5, color=COLORS["fixed_avg"], linestyle="--", linewidth=1.4,
                         label=variant_labels["fixed_avg"])
            axis.set_xlim(0, 1)
            axis.set_xlabel("Spatial-branch weight α")
            axis.set_title(f"Horizon {horizon}")
            axis.grid(axis="y", linestyle="--", alpha=0.25)
        axes[0].set_ylabel("Density")
        handles, labels = axes[-1].get_legend_handles_labels()
        fig.legend(handles, labels, loc="upper center", ncol=2, frameon=True)
        fig.suptitle(dataset_labels[dataset], y=1.04)
        fig.tight_layout(rect=(0, 0, 1, 0.91))
        path = output_dir / f"fig_gate_alpha_distribution_{dataset}.png"
        fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor="white")
        fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
        plt.close(fig)
        print(f"WROTE alpha figures for {dataset}", flush=True)


def main() -> None:
    args = build_parser().parse_args()
    project_root = args.project_root.expanduser().resolve()
    output_dir = resolve_under(project_root, args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if "{seed}" not in args.result_pattern:
        raise ValueError("--result-pattern must contain {seed}")
    if len(set(args.seeds)) != len(args.seeds):
        raise ValueError("--seeds contains duplicates")
    if args.sample_per_batch_horizon <= 0 or args.max_samples_per_horizon <= 0:
        raise ValueError("sampling limits must be positive")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    device = torch.device(f"cuda:{args.gpu}")

    label_by_dataset = dict(DATASETS)
    label_by_variant = dict(VARIANTS)
    global_acc: Dict[Tuple[str, str, int], DistributionAccumulator] = {}
    seed_acc_rows: List[Dict[str, object]] = []
    performance_rows: List[Dict[str, object]] = []
    rng = np.random.default_rng(20260927)
    reference_protocol: Dict[str, object] | None = None

    for seed in args.seeds:
        result_root = project_root / args.result_pattern.format(seed=seed)
        for dataset in args.datasets:
            reference_path = result_root / dataset / args.variants[0] / "metrics.json"
            reference_payload = read_payload(reference_path)
            data_args = namespace_from_config(reference_payload["config"], project_root)
            data_args.dataset = dataset
            data_args.seed = seed
            train_loader, val_loader, test_loader, normalizer, height, width = build_loaders(
                data_args, device
            )
            del train_loader, val_loader, normalizer

            for variant in args.variants:
                variant_dir = result_root / dataset / variant
                payload = read_payload(variant_dir / "metrics.json")
                config = payload["config"]
                metrics = payload["metrics"]
                if int(config["seed"]) != seed:
                    raise ValueError(f"Seed mismatch in {variant_dir}")
                protocol = {key: config.get(key) for key in PROTOCOL_FIELDS}
                if reference_protocol is None:
                    reference_protocol = protocol
                elif protocol != reference_protocol:
                    raise ValueError(
                        f"Training protocol mismatch in {variant_dir}: "
                        f"expected {reference_protocol}, found {protocol}"
                    )
                performance_rows.append(
                    {
                        "seed": seed,
                        "dataset": dataset,
                        "dataset_label": label_by_dataset[dataset],
                        "variant": variant,
                        "variant_label": label_by_variant[variant],
                        "rmse": float(metrics["rmse"]),
                        "mae": float(metrics["mae"]),
                        "effective_branch_aux_weight": float(
                            metrics.get(
                                "effective_branch_aux_weight",
                                config.get("effective_branch_aux_weight", config.get("branch_aux_weight", 0.0)),
                            )
                        ),
                    }
                )
                model_args = namespace_from_config(config, project_root)
                model_args.dataset = dataset
                model = make_model(model_args, variant, height, width).to(device)
                checkpoint_path = variant_dir / "best.pt"
                try:
                    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
                except TypeError:
                    checkpoint = torch.load(checkpoint_path, map_location=device)
                model.load_state_dict(checkpoint["model_state"])
                local_acc = [DistributionAccumulator() for _ in range(model.t_out)]
                collect_alpha(
                    model, test_loader, device, local_acc, rng, args.sample_per_batch_horizon
                )
                for horizon_index, local in enumerate(local_acc, start=1):
                    local_retained = local.retained(rng, args.max_samples_per_horizon)
                    local_summary = local.summary(local_retained)
                    seed_acc_rows.append(
                        {
                            "seed": seed,
                            "dataset": dataset,
                            "variant": variant,
                            "horizon": horizon_index,
                            **local_summary,
                        }
                    )
                    key = (dataset, variant, horizon_index)
                    global_item = global_acc.setdefault(key, DistributionAccumulator())
                    # Merge exact moments and retained samples without revisiting tensors.
                    global_item.count += local.count
                    global_item.total += local.total
                    global_item.total_square += local.total_square
                    global_item.minimum = min(global_item.minimum, local.minimum)
                    global_item.maximum = max(global_item.maximum, local.maximum)
                    global_item.sample_chunks.append(local_retained.astype(np.float32))
                    global_item.sample_means.extend(local.sample_means)
                    global_item.within_sample_stds.extend(local.within_sample_stds)
                del model
            del test_loader
            torch.cuda.empty_cache()

    alpha_rows: List[Dict[str, object]] = []
    retained_samples: Dict[Tuple[str, str, int], np.ndarray] = {}
    for key, accumulator in sorted(global_acc.items()):
        dataset, variant, horizon = key
        retained = accumulator.retained(rng, args.max_samples_per_horizon)
        retained_samples[key] = retained
        alpha_rows.append(
            {
                "dataset": dataset,
                "dataset_label": label_by_dataset[dataset],
                "variant": variant,
                "variant_label": label_by_variant[variant],
                "horizon": horizon,
                "n_seeds": len(args.seeds),
                "seeds": "/".join(str(seed) for seed in sorted(args.seeds)),
                **accumulator.summary(retained),
            }
        )

    performance_summary = aggregate_performance(performance_rows)
    write_csv(
        output_dir / "fusion_performance_all_runs.csv",
        performance_rows,
        ("seed", "dataset", "dataset_label", "variant", "variant_label", "rmse", "mae", "effective_branch_aux_weight"),
    )
    write_csv(
        output_dir / "fusion_performance_mean_std.csv",
        performance_summary,
        ("dataset", "dataset_label", "variant", "variant_label", "n_seeds", "seeds", "rmse_mean", "rmse_std", "mae_mean", "mae_std"),
    )
    alpha_fields = (
        "dataset", "dataset_label", "variant", "variant_label", "horizon", "n_seeds", "seeds",
        "count", "mean", "std_all", "sample_mean_std", "mean_within_sample_std",
        "min", "q05", "q25", "median", "q75", "q95", "max",
    )
    write_csv(output_dir / "gate_alpha_by_horizon.csv", alpha_rows, alpha_fields)
    write_csv(
        output_dir / "gate_alpha_by_seed.csv",
        seed_acc_rows,
        ("seed", "dataset", "variant", "horizon", "count", "mean", "std_all", "sample_mean_std", "mean_within_sample_std", "min", "q05", "q25", "median", "q75", "q95", "max"),
    )
    plot_performance(performance_summary, args.datasets, args.variants, output_dir, args.dpi)
    plot_alpha(alpha_rows, retained_samples, args.datasets, args.variants, output_dir, args.dpi)
    print(
        f"COMPLETE: {len(performance_rows)} runs, {len(alpha_rows)} horizon summaries. "
        "Alpha weights the calibrated spatial branch; 1-alpha weights the temporal branch.",
        flush=True,
    )


if __name__ == "__main__":
    main()
