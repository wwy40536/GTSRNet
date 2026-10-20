"""Aggregate SGA-Seq seeds into Step 4/8/12/Overall and efficiency tables."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Mapping, Sequence, Tuple

import numpy as np


DATASETS = ("milan_sms", "milan_call", "milan_internet", "mobile_bj", "mobile_nj")
DISPLAY = {
    "milan_sms": "Milan-SMS",
    "milan_call": "Milan-Call",
    "milan_internet": "Milan-Internet",
    "mobile_bj": "MobileBJ",
    "mobile_nj": "MobileNJ",
}
STEPS = (4, 8, 12)


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--results-dir", type=Path, default=Path("results_sgaseq_10seed"))
    p.add_argument("--datasets", nargs="+", choices=DATASETS, default=DATASETS)
    p.add_argument("--seeds", nargs="+", type=int, default=(23, 26, 29, 36, 46, 56, 66, 76, 103, 128))
    p.add_argument("--allow-partial", action="store_true")
    return p


def mean_std(values: Sequence[float]) -> Tuple[float, float]:
    array = np.asarray(values, dtype=np.float64)
    return float(array.mean()), float(array.std(ddof=1)) if len(array) > 1 else 0.0


def write_csv(path: Path, rows: Sequence[Mapping[str, object]], fields: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    print(f"WROTE {path}", flush=True)


def collect(root: Path) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for path in sorted(root.glob("*/sgaseq/seed_*/metrics.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        metrics = payload["metrics"]
        horizons = {int(item["horizon"]): item for item in metrics["horizons"]}
        row: Dict[str, object] = {
            "dataset": path.parents[2].name,
            "seed": int(path.parent.name.removeprefix("seed_")),
            "rmse_overall": float(metrics["rmse"]),
            "mae_overall": float(metrics["mae"]),
            "parameters": int(metrics["parameters"]),
            "best_epoch": int(metrics["best_epoch"]),
            "training_seconds": float(metrics["training_seconds"]),
            "epoch_seconds_mean": metrics.get("epoch_seconds_mean"),
            "inference_seconds": float(metrics["inference_seconds"]),
            "metrics_path": str(path),
        }
        for step in STEPS:
            if step not in horizons:
                raise ValueError(f"{path} has no horizon {step}")
            row[f"rmse_step{step}"] = float(horizons[step]["rmse"])
            row[f"mae_step{step}"] = float(horizons[step]["mae"])
        rows.append(row)
    if not rows:
        raise FileNotFoundError(f"No SGA-Seq metrics.json found under {root}")
    return rows


def main() -> None:
    args = parser().parse_args()
    root = args.results_dir.expanduser().resolve()
    runs = collect(root)
    present = {(str(row["dataset"]), int(row["seed"])) for row in runs}
    missing = [
        f"{dataset}/sgaseq/seed_{seed}"
        for dataset in args.datasets
        for seed in args.seeds
        if (dataset, seed) not in present
    ]
    if missing and not args.allow_partial:
        raise RuntimeError("Missing expected runs:\n  " + "\n  ".join(missing))
    if missing:
        print(f"WARNING: {len(missing)} expected runs are missing", flush=True)

    selected = [
        row for row in runs
        if row["dataset"] in args.datasets and int(row["seed"]) in args.seeds
    ]
    output = root / "summary"
    run_fields = (
        "dataset", "seed", "rmse_step4", "mae_step4", "rmse_step8", "mae_step8",
        "rmse_step12", "mae_step12", "rmse_overall", "mae_overall", "parameters",
        "best_epoch", "training_seconds", "epoch_seconds_mean", "inference_seconds", "metrics_path",
    )
    write_csv(output / "all_runs.csv", selected, run_fields)

    grouped: Dict[str, List[Dict[str, object]]] = defaultdict(list)
    for row in selected:
        grouped[str(row["dataset"])].append(row)

    step_rows: List[Dict[str, object]] = []
    paper_lookup: Dict[Tuple[str, str], str] = {}
    for dataset in args.datasets:
        dataset_rows = grouped.get(dataset, [])
        if not dataset_rows:
            continue
        for label, suffix in (("Step 4", "step4"), ("Step 8", "step8"), ("Step 12", "step12"), ("Overall", "overall")):
            rmse_mean, rmse_std = mean_std([float(row[f"rmse_{suffix}"]) for row in dataset_rows])
            mae_mean, mae_std = mean_std([float(row[f"mae_{suffix}"]) for row in dataset_rows])
            step_rows.append(
                {
                    "dataset": dataset,
                    "dataset_name": DISPLAY[dataset],
                    "prediction_step": label,
                    "n_seeds": len(dataset_rows),
                    "rmse_mean": rmse_mean,
                    "rmse_std": rmse_std,
                    "mae_mean": mae_mean,
                    "mae_std": mae_std,
                    "RMSE/MAE": f"{rmse_mean:.3f} / {mae_mean:.3f}",
                    "RMSE/MAE (mean±std)": f"{rmse_mean:.3f}±{rmse_std:.3f} / {mae_mean:.3f}±{mae_std:.3f}",
                }
            )
            paper_lookup[(dataset, label)] = f"{rmse_mean:.3f} / {mae_mean:.3f}"
    write_csv(
        output / "step4_step8_step12_overall.csv",
        step_rows,
        ("dataset", "dataset_name", "prediction_step", "n_seeds", "rmse_mean", "rmse_std", "mae_mean", "mae_std", "RMSE/MAE", "RMSE/MAE (mean±std)"),
    )

    paper_rows = []
    for label in ("Step 4", "Step 8", "Step 12", "Overall"):
        row: Dict[str, object] = {"model": "SGA-Seq", "prediction_step": label}
        for dataset in args.datasets:
            row[DISPLAY[dataset]] = paper_lookup.get((dataset, label), "MISSING")
        paper_rows.append(row)
    write_csv(
        output / "paper_table_sgaseq.csv",
        paper_rows,
        ("model", "prediction_step", *(DISPLAY[dataset] for dataset in args.datasets)),
    )

    efficiency = []
    for dataset in args.datasets:
        dataset_rows = grouped.get(dataset, [])
        if not dataset_rows:
            continue
        epoch_values = [float(row["epoch_seconds_mean"]) for row in dataset_rows if row["epoch_seconds_mean"] is not None]
        inference_values = [float(row["inference_seconds"]) for row in dataset_rows]
        epoch_mean, epoch_std = mean_std(epoch_values) if epoch_values else (float("nan"), float("nan"))
        infer_mean, infer_std = mean_std(inference_values)
        efficiency.append(
            {
                "dataset": dataset,
                "dataset_name": DISPLAY[dataset],
                "n_seeds": len(dataset_rows),
                "parameters": int(round(np.mean([int(row["parameters"]) for row in dataset_rows]))),
                "parameters_M": np.mean([int(row["parameters"]) for row in dataset_rows]) / 1e6,
                "seconds_per_epoch_mean": epoch_mean,
                "seconds_per_epoch_std": epoch_std,
                "test_inference_seconds_mean": infer_mean,
                "test_inference_seconds_std": infer_std,
            }
        )
    write_csv(
        output / "efficiency_by_dataset.csv",
        efficiency,
        ("dataset", "dataset_name", "n_seeds", "parameters", "parameters_M", "seconds_per_epoch_mean", "seconds_per_epoch_std", "test_inference_seconds_mean", "test_inference_seconds_std"),
    )


if __name__ == "__main__":
    main()

