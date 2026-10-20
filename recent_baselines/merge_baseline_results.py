"""Merge old/new baseline metrics and GTSRNet runs; export tables and delta curves."""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np


DATASETS = ("milan_sms", "milan_call", "milan_internet", "mobile_bj", "mobile_nj")
OLD_MODELS = ("seasonal_ha", "gru_seq2seq", "mtgnn", "patchtst", "dlinear", "timemixer")
NEW_MODELS = ("itransformer", "segrnn", "stid")
ALL_MODELS = OLD_MODELS + NEW_MODELS + ("gtsrnet",)
DISPLAY = {
    "seasonal_ha": "S-HA", "gru_seq2seq": "GRU-Seq2Seq", "mtgnn": "MTGNN",
    "patchtst": "PatchTST", "dlinear": "DLinear", "timemixer": "TimeMixer",
    "itransformer": "iTransformer", "segrnn": "SegRNN", "stid": "STID",
    "gtsrnet": "GTSRNet",
}


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--old-results", type=Path, required=True)
    p.add_argument("--new-results", type=Path, required=True)
    p.add_argument("--gtsrnet-results", nargs="*", type=Path, default=())
    p.add_argument("--old-seeds", nargs="+", type=int, default=(26, 36, 66, 76))
    p.add_argument("--new-seeds", nargs="+", type=int, default=(23, 29, 103, 128))
    p.add_argument("--output-dir", type=Path, default=Path("results_baselines_merged"))
    p.add_argument("--allow-partial", action="store_true")
    p.add_argument("--dpi", type=int, default=300)
    p.add_argument("--no-plots", action="store_true")
    return p


def read_payload(path: Path) -> Mapping[str, object]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Cannot read {path}: {exc}") from exc


def row_from(path: Path, dataset: str, model: str, seed: int, source: str) -> Tuple[Dict[str, object], List[Dict[str, object]]]:
    payload = read_payload(path)
    metrics = payload["metrics"]
    history = payload.get("history", [])
    epoch_times = [float(item["seconds"]) for item in history if isinstance(item, Mapping) and "seconds" in item]
    row: Dict[str, object] = {
        "dataset": dataset, "model": model, "display_name": DISPLAY[model], "seed": seed,
        "rmse": float(metrics["rmse"]), "mae": float(metrics["mae"]),
        "parameters": int(metrics.get("parameters", 0)),
        "best_epoch": int(metrics.get("best_epoch", -1)),
        "training_seconds": float(metrics.get("training_seconds", float("nan"))),
        "seconds_per_epoch": float(np.mean(epoch_times)) if epoch_times else float("nan"),
        "inference_seconds": float(metrics.get("inference_seconds", float("nan"))),
        "source": source, "metrics_path": str(path),
    }
    steps = [
        {
            "dataset": dataset, "model": model, "display_name": DISPLAY[model], "seed": seed,
            "forecast_step": int(item["horizon"]), "rmse": float(item["rmse"]),
            "mae": float(item["mae"]), "source": source,
        }
        for item in metrics.get("horizons", [])
    ]
    return row, steps


def collect_standard(root: Path, models: Sequence[str], seeds: Sequence[int], source: str) -> Tuple[List[Dict[str, object]], List[Dict[str, object]]]:
    runs: List[Dict[str, object]] = []; steps: List[Dict[str, object]] = []
    allowed = set(seeds)
    seen = set()
    for path in sorted(root.expanduser().resolve().rglob("metrics.json")):
        if not path.parent.name.startswith("seed_"):
            continue
        dataset, model, seed_name = path.parents[2].name, path.parents[1].name, path.parent.name
        if dataset not in DATASETS or model not in models or not seed_name.startswith("seed_"):
            continue
        seed = int(seed_name[5:])
        # S-HA is deterministic and historically may have been stored for only one seed.
        if model != "seasonal_ha" and seed not in allowed:
            continue
        key = (dataset, model, seed)
        if key in seen:
            continue
        seen.add(key)
        row, per_step = row_from(path, dataset, model, seed, source)
        runs.append(row); steps.extend(per_step)
    return runs, steps


def seed_from_payload_or_path(payload: Mapping[str, object], path: Path) -> int:
    config = payload.get("config", {})
    if isinstance(config, Mapping) and "seed" in config:
        return int(config["seed"])
    match = re.search(r"seed[_-]?(\d+)", str(path), flags=re.IGNORECASE)
    if match:
        return int(match.group(1))
    raise ValueError(f"Cannot determine seed for {path}")


def collect_gtsrnet(roots: Sequence[Path], seeds: Sequence[int]) -> Tuple[List[Dict[str, object]], List[Dict[str, object]]]:
    runs: List[Dict[str, object]] = []; steps: List[Dict[str, object]] = []
    allowed = set(seeds)
    seen = set()
    for root in roots:
        root = root.expanduser().resolve()
        candidates = [root] if root.name == "metrics.json" else list(root.rglob("GTSRNet/metrics.json"))
        for path in sorted(candidates):
            payload = read_payload(path)
            metrics = payload.get("metrics", {})
            dataset = str(metrics.get("dataset", path.parents[1].name))
            seed = seed_from_payload_or_path(payload, path)
            key = (dataset, seed)
            if dataset not in DATASETS or seed not in allowed or key in seen:
                continue
            seen.add(key)
            row, per_step = row_from(path, dataset, "gtsrnet", seed, "gtsrnet")
            runs.append(row); steps.extend(per_step)
    return runs, steps


def write_csv(path: Path, rows: Sequence[Mapping[str, object]], fields: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader(); writer.writerows(rows)
    print(f"WROTE {path}", flush=True)


def mean_std(values: Iterable[float]) -> Tuple[float, float]:
    array = np.asarray(list(values), dtype=np.float64)
    return float(array.mean()), float(array.std(ddof=1)) if len(array) > 1 else 0.0


def aggregate(runs: Sequence[Mapping[str, object]]) -> List[Dict[str, object]]:
    groups: Dict[Tuple[str, str], List[Mapping[str, object]]] = defaultdict(list)
    for row in runs: groups[(str(row["dataset"]), str(row["model"]))].append(row)
    output: List[Dict[str, object]] = []
    for (dataset, model), rows in sorted(groups.items()):
        rm, rs = mean_std(float(r["rmse"]) for r in rows); mm, ms = mean_std(float(r["mae"]) for r in rows)
        output.append({
            "dataset": dataset, "model": model, "display_name": DISPLAY[model], "n_seeds": len(rows),
            "seeds": ",".join(str(r["seed"]) for r in sorted(rows, key=lambda x: int(x["seed"]))),
            "rmse_mean": rm, "rmse_std": rs, "mae_mean": mm, "mae_std": ms,
            "RMSE (mean ± std)": f"{rm:.4f} ± {rs:.4f}",
            "MAE (mean ± std)": f"{mm:.4f} ± {ms:.4f}",
            "parameters": int(round(np.mean([int(r["parameters"]) for r in rows]))),
        })
    return output


def paper_tables(summary: Sequence[Mapping[str, object]]) -> Tuple[List[Dict[str, object]], List[Dict[str, object]]]:
    lookup = {(str(r["dataset"]), str(r["model"])): r for r in summary}
    mean_rows: List[Dict[str, object]] = []; std_rows: List[Dict[str, object]] = []
    for model in ALL_MODELS:
        mean_row: Dict[str, object] = {"model": DISPLAY[model]}; std_row: Dict[str, object] = {"model": DISPLAY[model]}
        for dataset in DATASETS:
            value = lookup.get((dataset, model))
            mean_row[dataset] = f"{float(value['rmse_mean']):.3f} / {float(value['mae_mean']):.3f}" if value else "MISSING"
            std_row[dataset] = (
                f"{float(value['rmse_mean']):.3f} ± {float(value['rmse_std']):.3f} / "
                f"{float(value['mae_mean']):.3f} ± {float(value['mae_std']):.3f}"
            ) if value else "MISSING"
        mean_rows.append(mean_row); std_rows.append(std_row)
    return mean_rows, std_rows


def improvements(summary: Sequence[Mapping[str, object]]) -> List[Dict[str, object]]:
    lookup = {(str(r["dataset"]), str(r["model"])): r for r in summary}
    rows: List[Dict[str, object]] = []
    for dataset in DATASETS:
        ours = lookup.get((dataset, "gtsrnet"))
        candidates = [lookup[(dataset, model)] for model in OLD_MODELS + NEW_MODELS if (dataset, model) in lookup]
        if not ours or not candidates: continue
        best_rmse = min(candidates, key=lambda r: float(r["rmse_mean"]))
        best_mae = min(candidates, key=lambda r: float(r["mae_mean"]))
        rows.append({
            "dataset": dataset,
            "gtsrnet_rmse": float(ours["rmse_mean"]), "best_rmse_baseline": best_rmse["display_name"],
            "best_rmse": float(best_rmse["rmse_mean"]),
            "rmse_reduction_percent": (float(best_rmse["rmse_mean"]) - float(ours["rmse_mean"])) / float(best_rmse["rmse_mean"]) * 100,
            "gtsrnet_mae": float(ours["mae_mean"]), "best_mae_baseline": best_mae["display_name"],
            "best_mae": float(best_mae["mae_mean"]),
            "mae_reduction_percent": (float(best_mae["mae_mean"]) - float(ours["mae_mean"])) / float(best_mae["mae_mean"]) * 100,
        })
    return rows


def efficiency_table(runs: Sequence[Mapping[str, object]]) -> List[Dict[str, object]]:
    """Aggregate timing on the three Milan tasks (all use N=400 after cropping)."""
    groups: Dict[str, List[Mapping[str, object]]] = defaultdict(list)
    for row in runs:
        if str(row["dataset"]).startswith("milan_"):
            groups[str(row["model"])].append(row)
    output: List[Dict[str, object]] = []
    for model in ALL_MODELS:
        rows = groups.get(model, [])
        if not rows:
            continue
        train_values = [float(r["seconds_per_epoch"]) for r in rows if np.isfinite(float(r["seconds_per_epoch"]))]
        infer_values = [float(r["inference_seconds"]) for r in rows if np.isfinite(float(r["inference_seconds"]))]
        train_mean, train_std = mean_std(train_values) if train_values else (float("nan"), float("nan"))
        infer_mean, infer_std = mean_std(infer_values)
        parameters = int(round(np.mean([int(r["parameters"]) for r in rows])))
        output.append({
            "model": DISPLAY[model],
            "n_runs": len(rows),
            "training_seconds_per_epoch_mean": train_mean,
            "training_seconds_per_epoch_std": train_std,
            "training_time_table": "—" if not train_values else f"{train_mean:.3f} ± {train_std:.3f}",
            "inference_seconds_mean": infer_mean,
            "inference_seconds_std": infer_std,
            "inference_time_table": f"{infer_mean:.3f} ± {infer_std:.3f}",
            "parameters": parameters,
            "parameters_M": parameters / 1_000_000,
            "parameters_table": "0" if parameters == 0 else f"{parameters / 1_000_000:.4f}",
        })
    return output


def delta_summary(steps: Sequence[Mapping[str, object]]) -> List[Dict[str, object]]:
    by_run: Dict[Tuple[str, str, int], List[Mapping[str, object]]] = defaultdict(list)
    for row in steps: by_run[(str(row["dataset"]), str(row["model"]), int(row["seed"]))].append(row)
    grouped: Dict[Tuple[str, str, int], Dict[str, List[float]]] = defaultdict(lambda: {"rmse": [], "mae": []})
    for (dataset, model, _seed), rows in by_run.items():
        ordered = sorted(rows, key=lambda r: int(r["forecast_step"]))
        if not ordered or int(ordered[0]["forecast_step"]) != 1: continue
        first_rmse, first_mae = float(ordered[0]["rmse"]), float(ordered[0]["mae"])
        for row in ordered:
            key = (dataset, model, int(row["forecast_step"]))
            grouped[key]["rmse"].append(float(row["rmse"]) - first_rmse)
            grouped[key]["mae"].append(float(row["mae"]) - first_mae)
    output: List[Dict[str, object]] = []
    for (dataset, model, step), values in sorted(grouped.items()):
        rm, rs = mean_std(values["rmse"]); mm, ms = mean_std(values["mae"])
        output.append({"dataset": dataset, "model": model, "display_name": DISPLAY[model], "forecast_step": step,
                       "delta_rmse_mean": rm, "delta_rmse_std": rs, "delta_mae_mean": mm, "delta_mae_std": ms,
                       "n_seeds": len(values["rmse"])})
    return output


def absolute_step_summary(steps: Sequence[Mapping[str, object]]) -> List[Dict[str, object]]:
    grouped: Dict[Tuple[str, str, int], Dict[str, List[float]]] = defaultdict(lambda: {"rmse": [], "mae": []})
    for row in steps:
        key = (str(row["dataset"]), str(row["model"]), int(row["forecast_step"]))
        grouped[key]["rmse"].append(float(row["rmse"]))
        grouped[key]["mae"].append(float(row["mae"]))
    output: List[Dict[str, object]] = []
    for (dataset, model, step), values in sorted(grouped.items()):
        rm, rs = mean_std(values["rmse"]); mm, ms = mean_std(values["mae"])
        output.append({
            "dataset": dataset, "model": model, "display_name": DISPLAY[model],
            "forecast_step": step, "rmse_mean": rm, "rmse_std": rs,
            "mae_mean": mm, "mae_std": ms, "n_seeds": len(values["rmse"]),
        })
    return output


def multistep_paper_table(
    absolute_steps: Sequence[Mapping[str, object]],
    summary: Sequence[Mapping[str, object]],
) -> List[Dict[str, object]]:
    step_lookup = {
        (str(row["dataset"]), str(row["model"]), int(row["forecast_step"])): row
        for row in absolute_steps
    }
    overall_lookup = {(str(row["dataset"]), str(row["model"])): row for row in summary}
    output: List[Dict[str, object]] = []
    for model in ALL_MODELS:
        for label, step in (("Step 4", 4), ("Step 8", 8), ("Step 12", 12), ("Overall", None)):
            row: Dict[str, object] = {"model": DISPLAY[model], "forecast_horizon": label}
            for dataset in DATASETS:
                values = overall_lookup.get((dataset, model)) if step is None else step_lookup.get((dataset, model, step))
                row[dataset] = (
                    f"{float(values['rmse_mean']):.3f} / {float(values['mae_mean']):.3f}"
                    if values else "MISSING"
                )
            output.append(row)
    return output


def plot_delta(dataset: str, metric: str, rows: Sequence[Mapping[str, object]], output: Path, dpi: int) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    selected = [r for r in rows if r["dataset"] == dataset]
    if not selected: return
    fig, ax = plt.subplots(figsize=(7.2, 4.8), constrained_layout=True)
    colors = plt.get_cmap("tab10")
    for index, model in enumerate(ALL_MODELS):
        if model == "seasonal_ha":
            continue
        values = sorted(
            (
                r for r in selected
                if r["model"] == model and int(r["forecast_step"]) >= 2
            ),
            key=lambda r: int(r["forecast_step"]),
        )
        if not values: continue
        x = np.asarray([r["forecast_step"] for r in values]); mean = np.asarray([r[f"delta_{metric}_mean"] for r in values]); std = np.asarray([r[f"delta_{metric}_std"] for r in values])
        if model == "gtsrnet":
            color = "#c83e4d"
        elif model == "patchtst":
            color = "#2f6b9a"
        else:
            color = colors(index % 10)
        ax.plot(x, mean, marker="o", markersize=3.4, linewidth=2.2 if model == "gtsrnet" else 1.35, label=DISPLAY[model], color=color)
        ax.fill_between(x, mean - std, mean + std, alpha=0.12, color=color, linewidth=0)
    ax.axhline(0, color="#777777", linewidth=0.8); ax.grid(True, linestyle="--", alpha=0.25)
    ax.set_xticks(range(2, 13)); ax.set_xlabel("Forecast step h")
    ax.set_ylabel(rf"$\Delta$ {metric.upper()} from h=1")
    ax.set_title(f"Horizon-wise {metric.upper()} growth — {dataset.replace('_', ' ').title()} — mean ± std")
    ax.legend(ncol=3, fontsize=8, frameon=False)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output.with_suffix(".png"), dpi=dpi, bbox_inches="tight", facecolor="white")
    fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    plt.close(fig); print(f"WROTE {output.with_suffix('.png')} / .pdf", flush=True)


def expected_missing(
    runs: Sequence[Mapping[str, object]],
    old_seeds: Sequence[int],
    new_seeds: Sequence[int],
    have_gtsrnet: bool,
) -> List[str]:
    present = {(str(r["dataset"]), str(r["model"]), int(r["seed"])) for r in runs}
    missing: List[str] = []
    for dataset in DATASETS:
        if not any(item[0] == dataset and item[1] == "seasonal_ha" for item in present):
            missing.append(f"{dataset}/seasonal_ha/(one deterministic run)")
        for model in OLD_MODELS:
            if model == "seasonal_ha":
                continue
            for seed in old_seeds:
                if (dataset, model, seed) not in present:
                    missing.append(f"{dataset}/{model}/seed_{seed}")
        for model in NEW_MODELS:
            for seed in new_seeds:
                if (dataset, model, seed) not in present:
                    missing.append(f"{dataset}/{model}/seed_{seed}")
        if have_gtsrnet:
            for seed in new_seeds:
                if (dataset, "gtsrnet", seed) not in present:
                    missing.append(f"{dataset}/gtsrnet/seed_{seed}")
    return missing


def main() -> None:
    args = parser().parse_args()
    runs, steps = collect_standard(args.old_results, OLD_MODELS, args.old_seeds, "old_baselines")
    r, s = collect_standard(args.new_results, NEW_MODELS, args.new_seeds, "new_baselines"); runs += r; steps += s
    if args.gtsrnet_results:
        r, s = collect_gtsrnet(args.gtsrnet_results, args.new_seeds); runs += r; steps += s
    missing = expected_missing(runs, args.old_seeds, args.new_seeds, bool(args.gtsrnet_results))
    if missing and not args.allow_partial:
        raise RuntimeError("Missing dataset/model groups:\n  " + "\n  ".join(missing))
    if missing: print("WARNING partial results; missing: " + ", ".join(missing), flush=True)
    if not runs: raise FileNotFoundError("No metrics.json files were collected")
    output = args.output_dir.expanduser().resolve(); output.mkdir(parents=True, exist_ok=True)
    summary = aggregate(runs); deltas = delta_summary(steps); absolute_steps = absolute_step_summary(steps); means, mean_stds = paper_tables(summary); gains = improvements(summary); efficiency = efficiency_table(runs); multistep = multistep_paper_table(absolute_steps, summary)
    write_csv(output / "all_runs.csv", runs, ("dataset", "model", "display_name", "seed", "rmse", "mae", "parameters", "best_epoch", "training_seconds", "seconds_per_epoch", "inference_seconds", "source", "metrics_path"))
    write_csv(output / "baseline_summary.csv", summary, ("dataset", "model", "display_name", "n_seeds", "seeds", "rmse_mean", "rmse_std", "mae_mean", "mae_std", "RMSE (mean ± std)", "MAE (mean ± std)", "parameters"))
    write_csv(output / "paper_table_mean.csv", means, ("model",) + DATASETS)
    write_csv(output / "paper_table_mean_std.csv", mean_stds, ("model",) + DATASETS)
    write_csv(output / "per_horizon_delta.csv", deltas, ("dataset", "model", "display_name", "forecast_step", "delta_rmse_mean", "delta_rmse_std", "delta_mae_mean", "delta_mae_std", "n_seeds"))
    write_csv(output / "per_horizon_absolute.csv", absolute_steps, ("dataset", "model", "display_name", "forecast_step", "rmse_mean", "rmse_std", "mae_mean", "mae_std", "n_seeds"))
    write_csv(output / "paper_table_steps_4_8_12_overall.csv", multistep, ("model", "forecast_horizon") + DATASETS)
    write_csv(output / "gtsrnet_vs_best_baseline.csv", gains, ("dataset", "gtsrnet_rmse", "best_rmse_baseline", "best_rmse", "rmse_reduction_percent", "gtsrnet_mae", "best_mae_baseline", "best_mae", "mae_reduction_percent"))
    write_csv(output / "efficiency_table_milan.csv", efficiency, ("model", "n_runs", "training_seconds_per_epoch_mean", "training_seconds_per_epoch_std", "training_time_table", "inference_seconds_mean", "inference_seconds_std", "inference_time_table", "parameters", "parameters_M", "parameters_table"))
    if gains:
        avg_r = np.mean([r["rmse_reduction_percent"] for r in gains]); avg_m = np.mean([r["mae_reduction_percent"] for r in gains])
        (output / "average_improvement.txt").write_text(f"Datasets included: {len(gains)}\nAverage RMSE reduction: {avg_r:.4f}%\nAverage MAE reduction: {avg_m:.4f}%\n", encoding="utf-8")
    if not args.no_plots:
        for dataset in DATASETS:
            for metric in ("rmse", "mae"):
                plot_delta(dataset, metric, deltas, output / "figures" / f"fig_delta_{metric}_{dataset}", args.dpi)


if __name__ == "__main__":
    main()
