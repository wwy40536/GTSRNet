#!/usr/bin/env python3
"""Compare iTransformer and SegRNN per horizon on the validation split.

This script reuses completed checkpoints.  It does not train or overwrite any
existing result.  For every common seed it rebuilds the original data split,
evaluates ``best.pt`` on the validation loader, and reports horizon-wise MAE
and RMSE for iTransformer and SegRNN.

Place this file beside ``train_recent_baseline.py`` and run it from the project
root.  The existing baseline code and result checkpoints are required.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np
import torch

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import train_recent_baseline as training
from recent_data_core import build_loaders, choose_device, evaluate_predictions, set_seed


SCRIPT_DIR = Path(__file__).resolve().parent
MODELS = ("itransformer", "segrnn")
DEFAULT_DATASETS = (
    "milan_sms",
    "milan_call",
    "milan_internet",
    "mobile_bj",
    "mobile_nj",
)
PATH_ARGUMENTS = {"project_root", "data", "feature_path", "output_dir"}


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--project-root", type=Path, default=SCRIPT_DIR.parent)
    p.add_argument(
        "--results-dir", type=Path, default=Path("results_recent_baselines_4seed")
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs_itransformer_segrnn_validation"),
    )
    p.add_argument("--datasets", nargs="+", default=DEFAULT_DATASETS)
    p.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--num-workers", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--bootstrap", type=int, default=10000)
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument(
        "--relative-tie-percent",
        type=float,
        default=0.0,
        help=(
            "Treat mean errors as tied when their absolute gap divided by their "
            "mean is no larger than this percentage. Default 0 uses the exact sign."
        ),
    )
    p.add_argument("--dpi", type=int, default=400)
    return p


def resolve_under(root: Path, path: Path) -> Path:
    path = path.expanduser()
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def seed_from_dir(path: Path) -> int:
    try:
        return int(path.name.removeprefix("seed_"))
    except ValueError as exc:
        raise ValueError(f"Invalid seed directory: {path}") from exc


def discover_runs(
    results_root: Path, datasets: Sequence[str]
) -> Dict[str, Dict[str, Dict[int, Path]]]:
    discovered: Dict[str, Dict[str, Dict[int, Path]]] = {}
    missing: List[str] = []
    for dataset in datasets:
        discovered[dataset] = {}
        for model in MODELS:
            model_root = results_root / dataset / model
            runs = {}
            if model_root.is_dir():
                for run_dir in sorted(model_root.glob("seed_*")):
                    if not run_dir.is_dir():
                        continue
                    metric_path = run_dir / "metrics.json"
                    checkpoint_path = run_dir / "best.pt"
                    if metric_path.is_file() and checkpoint_path.is_file():
                        runs[seed_from_dir(run_dir)] = run_dir
                    elif metric_path.is_file() and not checkpoint_path.is_file():
                        missing.append(str(checkpoint_path))
            discovered[dataset][model] = runs
    if missing:
        raise FileNotFoundError(
            "metrics.json exists but best.pt is missing:\n  " + "\n  ".join(missing)
        )
    return discovered


def common_seeds(
    dataset_runs: Mapping[str, Mapping[int, Path]], dataset: str
) -> Tuple[int, ...]:
    seed_sets = [set(dataset_runs[model]) for model in MODELS]
    shared = tuple(sorted(set.intersection(*seed_sets))) if seed_sets else ()
    if not shared:
        counts = ", ".join(f"{model}={len(dataset_runs[model])}" for model in MODELS)
        raise FileNotFoundError(f"No common completed seeds for {dataset}: {counts}")
    return shared


def reconstruct_args(
    config: Mapping[str, object],
    project_root: Path,
    dataset: str,
    model: str,
    seed: int,
    cli: argparse.Namespace,
) -> argparse.Namespace:
    run_args = training.build_parser().parse_args([])
    for key, value in config.items():
        if key in PATH_ARGUMENTS and value is not None:
            value = Path(str(value))
        setattr(run_args, key, value)
    run_args.project_root = project_root
    run_args.dataset = dataset
    run_args.model = model
    run_args.seed = seed
    run_args.device = cli.device
    run_args.gpu = cli.gpu
    if cli.num_workers is not None:
        run_args.num_workers = cli.num_workers
    if cli.batch_size is not None:
        run_args.batch_size = cli.batch_size
    # Stored paths may be absolute paths from a different server.  The default
    # dataset paths under the current project root are the reproducible choice.
    if dataset.startswith("milan_"):
        run_args.data = None
    elif dataset == "mobile_bj":
        run_args.data = Path("dataset/MobileBJ.npy")
    elif dataset == "mobile_nj":
        run_args.data = Path("dataset/MobileNJ.npy")
    return run_args


def evaluate_run(
    run_dir: Path,
    project_root: Path,
    dataset: str,
    model_name: str,
    seed: int,
    cli: argparse.Namespace,
    device: torch.device,
) -> Tuple[List[Dict[str, object]], np.ndarray]:
    metric_path = run_dir / "metrics.json"
    payload = json.loads(metric_path.read_text(encoding="utf-8"))
    config = payload.get("config")
    if not isinstance(config, dict):
        raise ValueError(f"Missing config object in {metric_path}")
    run_args = reconstruct_args(
        config, project_root, dataset, model_name, seed, cli
    )
    training.validate(run_args)
    set_seed(seed)
    train_loader, val_loader, _test_loader, normalizer, height, width = build_loaders(
        run_args, device
    )
    del train_loader
    model = training.make_model(run_args, height, width, device)
    checkpoint = training.checkpoint_load(run_dir / "best.pt", device)
    if "model_state" not in checkpoint:
        raise KeyError(f"model_state is missing in {run_dir / 'best.pt'}")
    model.load_state_dict(checkpoint["model_state"])
    metrics, _prediction, truth = evaluate_predictions(
        model, val_loader, device, normalizer
    )
    rows: List[Dict[str, object]] = []
    for item in metrics["horizons"]:
        rows.append(
            {
                "dataset": dataset,
                "split": "validation",
                "seed": seed,
                "model": model_name,
                "horizon": int(item["horizon"]),
                "mae": float(item["mae"]),
                "rmse": float(item["rmse"]),
                "validation_samples": int(truth.shape[0]),
                "best_epoch": int(checkpoint.get("epoch", -1)),
                "best_validation_loss": float(checkpoint.get("val_loss", float("nan"))),
            }
        )
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return rows, truth


def write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    if not rows:
        raise ValueError(f"No rows to write: {path}")
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def sample_std(values: np.ndarray) -> float:
    return float(values.std(ddof=1)) if len(values) > 1 else 0.0


def bootstrap_mean_ci(
    values: np.ndarray, repeats: int, rng: np.random.Generator
) -> Tuple[float, float]:
    values = np.asarray(values, dtype=np.float64)
    if len(values) == 1:
        return float(values[0]), float(values[0])
    samples = rng.choice(values, size=(repeats, len(values)), replace=True)
    means = samples.mean(axis=1)
    return tuple(float(x) for x in np.quantile(means, (0.025, 0.975)))


def exact_sign_flip_p_value(values: np.ndarray) -> float:
    """Two-sided paired randomization p-value across common seeds."""
    values = np.asarray(values, dtype=np.float64)
    observed = abs(float(values.mean()))
    if not len(values):
        return float("nan")
    if len(values) <= 20:
        permuted = []
        for signs in itertools.product((-1.0, 1.0), repeat=len(values)):
            permuted.append(abs(float(np.mean(values * np.asarray(signs)))))
        return float(np.mean(np.asarray(permuted) >= observed - 1e-15))
    rng = np.random.default_rng(119)
    signs = rng.choice((-1.0, 1.0), size=(100000, len(values)))
    permuted = np.abs((signs * values[None, :]).mean(axis=1))
    return float((np.count_nonzero(permuted >= observed) + 1) / (len(permuted) + 1))


def summarize(
    raw_rows: Sequence[Mapping[str, object]],
    bootstrap: int,
    seed: int,
    relative_tie_percent: float,
) -> Tuple[List[Dict[str, object]], List[Dict[str, object]], List[Dict[str, object]]]:
    rng = np.random.default_rng(seed)
    datasets = tuple(dict.fromkeys(str(row["dataset"]) for row in raw_rows))
    horizons = sorted({int(row["horizon"]) for row in raw_rows})
    metrics = ("mae", "rmse")
    aggregate_rows: List[Dict[str, object]] = []
    comparison_rows: List[Dict[str, object]] = []
    switch_rows: List[Dict[str, object]] = []

    lookup: Dict[Tuple[str, str, int, int], Mapping[str, object]] = {}
    for row in raw_rows:
        lookup[(str(row["dataset"]), str(row["model"]), int(row["seed"]), int(row["horizon"]))] = row

    for dataset in datasets:
        dataset_seeds = sorted(
            {
                int(row["seed"])
                for row in raw_rows
                if str(row["dataset"]) == dataset
            }
        )
        for metric in metrics:
            winners: List[str] = []
            for horizon in horizons:
                model_values: Dict[str, np.ndarray] = {}
                for model in MODELS:
                    values = np.asarray(
                        [
                            float(lookup[(dataset, model, s, horizon)][metric])
                            for s in dataset_seeds
                        ],
                        dtype=np.float64,
                    )
                    model_values[model] = values
                    ci = bootstrap_mean_ci(values, bootstrap, rng)
                    aggregate_rows.append(
                        {
                            "dataset": dataset,
                            "split": "validation",
                            "metric": metric.upper(),
                            "model": model,
                            "horizon": horizon,
                            "seed_count": len(values),
                            "mean": float(values.mean()),
                            "std": sample_std(values),
                            "ci95_low": ci[0],
                            "ci95_high": ci[1],
                        }
                    )

                itr = model_values["itransformer"]
                seg = model_values["segrnn"]
                differences = itr - seg
                itr_mean = float(itr.mean())
                seg_mean = float(seg.mean())
                denominator = (abs(itr_mean) + abs(seg_mean)) / 2.0
                relative_gap = (
                    100.0 * (itr_mean - seg_mean) / denominator
                    if denominator > 0
                    else 0.0
                )
                if abs(relative_gap) <= relative_tie_percent:
                    winner = "tie"
                elif itr_mean < seg_mean:
                    winner = "itransformer"
                else:
                    winner = "segrnn"
                winners.append(winner)
                ci = bootstrap_mean_ci(differences, bootstrap, rng)
                comparison_rows.append(
                    {
                        "dataset": dataset,
                        "split": "validation",
                        "metric": metric.upper(),
                        "horizon": horizon,
                        "paired_seed_count": len(dataset_seeds),
                        "itransformer_mean": itr_mean,
                        "segrnn_mean": seg_mean,
                        "itransformer_minus_segrnn": float(differences.mean()),
                        "difference_std": sample_std(differences),
                        "difference_ci95_low": ci[0],
                        "difference_ci95_high": ci[1],
                        "relative_gap_percent_of_pair_mean": relative_gap,
                        "winner": winner,
                        "paired_sign_flip_p_value_two_sided": exact_sign_flip_p_value(differences),
                    }
                )

            non_ties = [(h, w) for h, w in zip(horizons, winners) if w != "tie"]
            crossover_after: List[int] = []
            for (previous_h, previous), (current_h, current) in zip(non_ties, non_ties[1:]):
                if previous != current:
                    crossover_after.append(previous_h)
            unique_winners = sorted({winner for _, winner in non_ties})
            switch_rows.append(
                {
                    "dataset": dataset,
                    "split": "validation",
                    "metric": metric.upper(),
                    "advantage_changes_with_horizon": len(unique_winners) > 1,
                    "winner_sequence_h1_to_h12": "|".join(winners),
                    "crossover_after_horizons": (
                        "|".join(str(x) for x in crossover_after) if crossover_after else ""
                    ),
                    "non_tie_winners": "|".join(unique_winners),
                }
            )
    return aggregate_rows, comparison_rows, switch_rows


def plot_dataset(
    output_dir: Path,
    dataset: str,
    aggregate_rows: Sequence[Mapping[str, object]],
    comparison_rows: Sequence[Mapping[str, object]],
    dpi: int,
) -> None:
    fig, axes = plt.subplots(
        2,
        2,
        figsize=(10.0, 6.8),
        sharex="col",
        gridspec_kw={"height_ratios": (2.1, 1.0)},
    )
    colors = {"itransformer": "#1F77B4", "segrnn": "#D62728"}
    labels = {"itransformer": "iTransformer", "segrnn": "SegRNN"}
    horizons = np.asarray(
        sorted(
            {
                int(row["horizon"])
                for row in comparison_rows
                if str(row["dataset"]) == dataset
            }
        )
    )
    for column, metric in enumerate(("MAE", "RMSE")):
        top = axes[0, column]
        bottom = axes[1, column]
        for model in MODELS:
            selected = sorted(
                (
                    row
                    for row in aggregate_rows
                    if str(row["dataset"]) == dataset
                    and str(row["metric"]) == metric
                    and str(row["model"]) == model
                ),
                key=lambda row: int(row["horizon"]),
            )
            means = np.asarray([float(row["mean"]) for row in selected])
            stds = np.asarray([float(row["std"]) for row in selected])
            top.plot(
                horizons,
                means,
                marker="o",
                linewidth=1.8,
                markersize=4,
                color=colors[model],
                label=labels[model],
            )
            top.fill_between(
                horizons,
                means - stds,
                means + stds,
                color=colors[model],
                alpha=0.15,
                linewidth=0,
            )
        selected_difference = sorted(
            (
                row
                for row in comparison_rows
                if str(row["dataset"]) == dataset and str(row["metric"]) == metric
            ),
            key=lambda row: int(row["horizon"]),
        )
        differences = np.asarray(
            [float(row["itransformer_minus_segrnn"]) for row in selected_difference]
        )
        low = np.asarray([float(row["difference_ci95_low"]) for row in selected_difference])
        high = np.asarray([float(row["difference_ci95_high"]) for row in selected_difference])
        bottom.axhline(0.0, color="#444444", linewidth=1.0, linestyle="--")
        bottom.plot(horizons, differences, marker="o", color="#6A3D9A", linewidth=1.8)
        bottom.fill_between(horizons, low, high, color="#6A3D9A", alpha=0.15, linewidth=0)
        bottom.fill_between(
            horizons,
            0,
            differences,
            where=differences < 0,
            color=colors["itransformer"],
            alpha=0.12,
            interpolate=True,
        )
        bottom.fill_between(
            horizons,
            0,
            differences,
            where=differences > 0,
            color=colors["segrnn"],
            alpha=0.12,
            interpolate=True,
        )
        top.set_title(metric)
        top.set_ylabel(f"Validation {metric}")
        top.grid(alpha=0.25)
        bottom.set_xlabel("Forecast horizon")
        bottom.set_ylabel("iTransformer - SegRNN")
        bottom.grid(alpha=0.25)
        bottom.set_xticks(horizons)
    axes[0, 0].legend(frameon=False)
    fig.suptitle(f"Validation error by forecast horizon — {dataset}")
    fig.tight_layout()
    for suffix in ("png", "pdf"):
        fig.savefig(
            output_dir / f"{dataset}_validation_horizon_comparison.{suffix}",
            dpi=dpi,
            bbox_inches="tight",
        )
    plt.close(fig)


def main() -> None:
    args = parser().parse_args()
    if args.bootstrap <= 0:
        raise ValueError("--bootstrap must be positive")
    if args.relative_tie_percent < 0:
        raise ValueError("--relative-tie-percent cannot be negative")
    project_root = args.project_root.expanduser().resolve()
    results_root = resolve_under(project_root, args.results_dir)
    output_dir = resolve_under(project_root, args.output_dir)
    if not results_root.is_dir():
        raise FileNotFoundError(f"Results directory not found: {results_root}")
    output_dir.mkdir(parents=True, exist_ok=True)
    device = choose_device(args.device, args.gpu)
    discovered = discover_runs(results_root, args.datasets)

    raw_rows: List[Dict[str, object]] = []
    truth_reference: Dict[Tuple[str, int], np.ndarray] = {}
    seed_inventory = []
    for dataset in args.datasets:
        seeds = common_seeds(discovered[dataset], dataset)
        seed_inventory.append(
            {
                "dataset": dataset,
                "common_seed_count": len(seeds),
                "common_seeds": "|".join(str(seed) for seed in seeds),
            }
        )
        for seed in seeds:
            for model_name in MODELS:
                run_dir = discovered[dataset][model_name][seed]
                print(f"EVAL {dataset}/{model_name}/seed_{seed}", flush=True)
                rows, truth = evaluate_run(
                    run_dir,
                    project_root,
                    dataset,
                    model_name,
                    seed,
                    args,
                    device,
                )
                key = (dataset, seed)
                if key in truth_reference:
                    reference = truth_reference[key]
                    if reference.shape != truth.shape or not np.allclose(
                        reference, truth, rtol=0.0, atol=1e-6
                    ):
                        raise RuntimeError(
                            f"Validation truth differs between models for {dataset}, seed={seed}"
                        )
                else:
                    truth_reference[key] = truth
                raw_rows.extend(rows)

    aggregate_rows, comparison_rows, switch_rows = summarize(
        raw_rows,
        args.bootstrap,
        args.seed,
        args.relative_tie_percent,
    )
    write_csv(output_dir / "per_run_validation_horizons.csv", raw_rows)
    write_csv(output_dir / "model_validation_horizons_mean_std.csv", aggregate_rows)
    write_csv(output_dir / "itransformer_vs_segrnn_by_horizon.csv", comparison_rows)
    write_csv(output_dir / "advantage_switch_summary.csv", switch_rows)
    write_csv(output_dir / "common_seed_inventory.csv", seed_inventory)

    for dataset in args.datasets:
        plot_dataset(
            output_dir,
            dataset,
            aggregate_rows,
            comparison_rows,
            args.dpi,
        )

    summary = {
        "split": "validation",
        "models": list(MODELS),
        "results_root": str(results_root),
        "datasets": list(args.datasets),
        "tie_threshold_percent": args.relative_tie_percent,
        "difference_definition": (
            "iTransformer minus SegRNN; negative means iTransformer has lower error"
        ),
        "seed_inventory": seed_inventory,
        "advantage_switch_summary": switch_rows,
    }
    (output_dir / "analysis_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"\nSaved validation comparison to: {output_dir}")
    for row in switch_rows:
        print(
            f"{row['dataset']} {row['metric']}: "
            f"changes={row['advantage_changes_with_horizon']}, "
            f"sequence={row['winner_sequence_h1_to_h12']}"
        )


if __name__ == "__main__":
    main()
