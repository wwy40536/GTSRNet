"""Create paired, per-seed spatial-grid and target-hour error-difference analyses."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


DATASETS = ("milan_sms", "milan_call", "milan_internet", "mobile_bj", "mobile_nj")
MILAN_DATASETS = ("milan_sms", "milan_call", "milan_internet")
PROTOCOL_KEYS = (
    "dataset", "seed", "close_size", "predict_size", "test_size", "val_ratio",
    "split_mode", "resolved_split_gap", "milan_loader", "crop", "rows", "cols",
    "height", "width", "batch_size", "epochs", "learning_rate", "weight_decay",
    "loss", "branch_aux_weight", "d_model", "seg_len", "dropout", "n_heads",
    "e_layers", "d_ff", "cycle_len",
)


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--results-dir", type=Path, default=Path("results_mechanism_error_maps"))
    p.add_argument("--analysis-dir", type=Path, default=None)
    p.add_argument("--spatial-datasets", nargs="+", choices=DATASETS, default=DATASETS)
    p.add_argument("--phase-datasets", nargs="+", choices=MILAN_DATASETS, default=MILAN_DATASETS)
    p.add_argument("--seeds", nargs="+", type=int, default=(16, 26, 36))
    p.add_argument("--dpi", type=int, default=220)
    return p


def run_dir(results: Path, mode: str, seed: int, dataset: str, variant: str) -> Path:
    return results / mode / f"seed_{seed}" / dataset / variant


def read_run(path: Path) -> Tuple[Dict[str, np.ndarray], Mapping[str, object]]:
    prediction_path = path / "predictions.npz"
    metrics_path = path / "metrics.json"
    if not prediction_path.is_file() or not metrics_path.is_file():
        raise FileNotFoundError(f"Missing predictions.npz or metrics.json under {path}")
    with np.load(prediction_path, allow_pickle=False) as source:
        required = {"pred", "truth", "target_hour"}
        missing = required.difference(source.files)
        if missing:
            raise KeyError(f"{prediction_path} is missing arrays: {sorted(missing)}")
        arrays = {key: np.asarray(source[key]) for key in source.files}
    payload = json.loads(metrics_path.read_text(encoding="utf-8"))
    return arrays, payload


def canonical(values: np.ndarray, label: str) -> np.ndarray:
    values = np.asarray(values)
    if values.ndim == 4:
        values = values[:, :, None, :, :]
    if values.ndim != 5:
        raise ValueError(f"{label}: expected [sample,horizon,channel,H,W], got {values.shape}")
    if not np.all(np.isfinite(values)):
        raise ValueError(f"{label}: contains NaN or infinity")
    return values.astype(np.float64, copy=False)


def compare_protocol(reference: Mapping[str, object], candidate: Mapping[str, object], label: str) -> None:
    left = reference.get("config", {})
    right = candidate.get("config", {})
    differences = []
    for key in PROTOCOL_KEYS:
        a, b = left.get(key), right.get(key)
        if isinstance(a, (int, float)) and isinstance(b, (int, float)):
            equal = bool(np.isclose(float(a), float(b), rtol=1e-12, atol=1e-12))
        else:
            equal = a == b
        if not equal:
            differences.append(f"{key}: {a!r} != {b!r}")
    if differences:
        raise ValueError(f"Protocol mismatch for {label}: " + "; ".join(differences))


def validate_pair(
    reference_arrays: Mapping[str, np.ndarray],
    candidate_arrays: Mapping[str, np.ndarray],
    label: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    pred_a = canonical(reference_arrays["pred"], label + "/reference_pred")
    truth_a = canonical(reference_arrays["truth"], label + "/reference_truth")
    pred_b = canonical(candidate_arrays["pred"], label + "/candidate_pred")
    truth_b = canonical(candidate_arrays["truth"], label + "/candidate_truth")
    if pred_a.shape != truth_a.shape or pred_b.shape != truth_b.shape or pred_a.shape != pred_b.shape:
        raise ValueError(
            f"Shape mismatch for {label}: pred/truth/reference/candidate are "
            f"{pred_a.shape}, {truth_a.shape}, {pred_b.shape}, {truth_b.shape}"
        )
    if not np.allclose(truth_a, truth_b, rtol=1e-6, atol=1e-5):
        maximum = float(np.max(np.abs(truth_a - truth_b)))
        raise ValueError(f"Truth mismatch for {label}; max absolute difference={maximum}")
    hour_a = np.asarray(reference_arrays["target_hour"], dtype=np.int64)
    hour_b = np.asarray(candidate_arrays["target_hour"], dtype=np.int64)
    if hour_a.shape != pred_a.shape[:2] or not np.array_equal(hour_a, hour_b):
        raise ValueError(f"Target-hour mismatch for {label}")
    return pred_a, pred_b, truth_a, hour_a


def write_csv(path: Path, rows: Iterable[Mapping[str, object]], fields: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def offsets(payload: Mapping[str, object]) -> Tuple[int, int]:
    config = payload.get("config", {})
    if not str(config.get("dataset", "")).startswith("milan_"):
        return 0, 0
    if not config.get("crop", False):
        return 0, 0
    rows = config.get("rows", [0, 0])
    cols = config.get("cols", [0, 0])
    return int(rows[0]), int(cols[0])


def grid_quartiles(mean_truth: np.ndarray) -> np.ndarray:
    flat_order = np.argsort(mean_truth.reshape(-1), kind="stable")
    labels = np.empty(flat_order.size, dtype=np.int64)
    for index, part in enumerate(np.array_split(flat_order, 4), start=1):
        labels[part] = index
    return labels.reshape(mean_truth.shape)


def plot_grid(
    delta_mae: np.ndarray,
    delta_rmse: np.ndarray,
    dataset: str,
    seed: int,
    path: Path,
    dpi: int,
) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(10.5, 4.3), constrained_layout=True)
    for axis, values, metric in zip(axes, (delta_mae, delta_rmse), ("MAE", "RMSE")):
        limit = max(float(np.max(np.abs(values))), np.finfo(float).eps)
        image = axis.imshow(values, cmap="RdBu", vmin=-limit, vmax=limit, origin="upper")
        improved = 100.0 * float(np.mean(values > 0))
        axis.set_title(f"$\\Delta${metric} = {metric}(wo_s) - {metric}(GTSRNet)\n{improved:.1f}% grids improved")
        axis.set_xlabel("Grid column (crop coordinates)")
        axis.set_ylabel("Grid row (crop coordinates)")
        colorbar = figure.colorbar(image, ax=axis, fraction=0.047, pad=0.03)
        colorbar.set_label("Positive: spatial branch helps")
    figure.suptitle(f"{dataset} | seed {seed} | spatial-branch contribution")
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=dpi, bbox_inches="tight")
    figure.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(figure)


def plot_hours(rows: Sequence[Mapping[str, object]], dataset: str, seed: int, path: Path, dpi: int) -> None:
    hours = np.asarray([int(row["hour"]) for row in rows])
    mae = np.asarray([float(row["delta_mae"]) for row in rows])
    rmse = np.asarray([float(row["delta_rmse"]) for row in rows])
    figure, axes = plt.subplots(2, 1, figsize=(10.5, 6.6), sharex=True, constrained_layout=True)
    for axis, values, metric in zip(axes, (mae, rmse), ("MAE", "RMSE")):
        colors = np.where(values >= 0, "#2c7fb8", "#d95f0e")
        axis.bar(hours, values, color=colors, width=0.78)
        axis.axhline(0.0, color="black", linewidth=0.9)
        axis.set_ylabel(f"$\\Delta${metric}")
        axis.grid(axis="y", alpha=0.25)
        axis.set_title(f"$\\Delta${metric} = {metric}(no_periodic) - {metric}(original_phase)")
    axes[-1].set_xticks(np.arange(24))
    axes[-1].set_xlabel("Target hour of day")
    figure.suptitle(f"{dataset} | seed {seed} | learned periodic-embedding contribution")
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=dpi, bbox_inches="tight")
    figure.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(figure)


def spatial_analysis(
    full: np.ndarray,
    wo_s: np.ndarray,
    truth: np.ndarray,
    payload: Mapping[str, object],
    dataset: str,
    seed: int,
    out: Path,
    dpi: int,
) -> Dict[str, float]:
    axes = (0, 1, 2)
    full_error = full - truth
    ablated_error = wo_s - truth
    full_mae = np.mean(np.abs(full_error), axis=axes)
    wo_mae = np.mean(np.abs(ablated_error), axis=axes)
    full_rmse = np.sqrt(np.mean(np.square(full_error), axis=axes))
    wo_rmse = np.sqrt(np.mean(np.square(ablated_error), axis=axes))
    delta_mae = wo_mae - full_mae
    delta_rmse = wo_rmse - full_rmse
    mean_truth = np.mean(truth, axis=axes)
    quartile = grid_quartiles(mean_truth)
    row_offset, col_offset = offsets(payload)
    grid_rows: List[Dict[str, object]] = []
    for row in range(delta_mae.shape[0]):
        for col in range(delta_mae.shape[1]):
            grid_rows.append({
                "dataset": dataset, "seed": seed,
                "crop_row": row, "crop_col": col,
                "original_row": row + row_offset, "original_col": col + col_offset,
                "mean_truth": float(mean_truth[row, col]),
                "traffic_quartile": int(quartile[row, col]),
                "GTSRNet_mae": float(full_mae[row, col]), "wo_s_mae": float(wo_mae[row, col]),
                "delta_mae": float(delta_mae[row, col]),
                "GTSRNet_rmse": float(full_rmse[row, col]), "wo_s_rmse": float(wo_rmse[row, col]),
                "delta_rmse": float(delta_rmse[row, col]),
                "spatial_helps_mae": int(delta_mae[row, col] > 0),
            })
    fields = tuple(grid_rows[0].keys())
    write_csv(out / "spatial_grid_differences.csv", grid_rows, fields)
    ranked = sorted(grid_rows, key=lambda item: float(item["delta_mae"]), reverse=True)
    selected = [dict(item, rank_type="most_improved", rank=i + 1) for i, item in enumerate(ranked[:10])]
    selected += [dict(item, rank_type="most_degraded", rank=i + 1) for i, item in enumerate(reversed(ranked[-10:]))]
    write_csv(out / "spatial_top_bottom_grids.csv", selected, ("rank_type", "rank", *fields))
    quartile_rows = []
    for q in range(1, 5):
        mask = quartile == q
        quartile_rows.append({
            "dataset": dataset, "seed": seed, "traffic_quartile": q,
            "grid_count": int(mask.sum()), "mean_truth": float(mean_truth[mask].mean()),
            "mean_delta_mae": float(delta_mae[mask].mean()),
            "mean_delta_rmse": float(delta_rmse[mask].mean()),
            "percent_grids_improved_mae": float(100.0 * np.mean(delta_mae[mask] > 0)),
        })
    write_csv(out / "spatial_by_traffic_quartile.csv", quartile_rows, tuple(quartile_rows[0].keys()))
    plot_grid(delta_mae, delta_rmse, dataset, seed, out / "spatial_grid_difference.png", dpi)
    global_full_mae = float(np.mean(np.abs(full_error)))
    global_wo_mae = float(np.mean(np.abs(ablated_error)))
    global_full_rmse = float(np.sqrt(np.mean(np.square(full_error))))
    global_wo_rmse = float(np.sqrt(np.mean(np.square(ablated_error))))
    return {
        "GTSRNet_mae": global_full_mae, "wo_s_mae": global_wo_mae,
        "mae_reduction_percent": 100.0 * (global_wo_mae - global_full_mae) / global_wo_mae,
        "GTSRNet_rmse": global_full_rmse, "wo_s_rmse": global_wo_rmse,
        "rmse_reduction_percent": 100.0 * (global_wo_rmse - global_full_rmse) / global_wo_rmse,
        "percent_grids_improved_mae": 100.0 * float(np.mean(delta_mae > 0)),
        "percent_grids_improved_rmse": 100.0 * float(np.mean(delta_rmse > 0)),
    }


def phase_analysis(
    phase: np.ndarray,
    no_phase: np.ndarray,
    truth: np.ndarray,
    target_hour: np.ndarray,
    dataset: str,
    seed: int,
    out: Path,
    dpi: int,
) -> Tuple[Dict[str, float], List[Dict[str, object]]]:
    error_phase = phase - truth
    error_none = no_phase - truth
    rows: List[Dict[str, object]] = []
    for hour in range(24):
        mask = target_hour == hour
        if not np.any(mask):
            raise ValueError(f"{dataset}/seed_{seed}: no test targets for hour {hour}")
        ep = error_phase[mask]
        en = error_none[mask]
        mae_p, mae_n = float(np.mean(np.abs(ep))), float(np.mean(np.abs(en)))
        rmse_p = float(np.sqrt(np.mean(np.square(ep))))
        rmse_n = float(np.sqrt(np.mean(np.square(en))))
        rows.append({
            "dataset": dataset, "seed": seed, "hour": hour,
            "target_count": int(mask.sum()),
            "original_phase_mae": mae_p, "no_periodic_mae": mae_n,
            "delta_mae": mae_n - mae_p,
            "original_phase_rmse": rmse_p, "no_periodic_rmse": rmse_n,
            "delta_rmse": rmse_n - rmse_p,
            "phase_helps_mae": int(mae_n > mae_p), "phase_helps_rmse": int(rmse_n > rmse_p),
        })
    write_csv(out / "phase_hour_differences.csv", rows, tuple(rows[0].keys()))
    plot_hours(rows, dataset, seed, out / "phase_hour_difference.png", dpi)
    mae_p = float(np.mean(np.abs(error_phase)))
    mae_n = float(np.mean(np.abs(error_none)))
    rmse_p = float(np.sqrt(np.mean(np.square(error_phase))))
    rmse_n = float(np.sqrt(np.mean(np.square(error_none))))
    summary = {
        "original_phase_mae": mae_p, "no_periodic_mae": mae_n,
        "mae_reduction_percent": 100.0 * (mae_n - mae_p) / mae_n,
        "original_phase_rmse": rmse_p, "no_periodic_rmse": rmse_n,
        "rmse_reduction_percent": 100.0 * (rmse_n - rmse_p) / rmse_n,
        "hours_improved_mae": int(sum(int(row["phase_helps_mae"]) for row in rows)),
        "hours_improved_rmse": int(sum(int(row["phase_helps_rmse"]) for row in rows)),
    }
    return summary, rows


def mean_std(values: Sequence[float]) -> str:
    array = np.asarray(values, dtype=float)
    return f"{array.mean():.4f} ± {array.std(ddof=1) if len(array) > 1 else 0.0:.4f}"


def write_report(
    path: Path,
    spatial_datasets: Sequence[str],
    phase_datasets: Sequence[str],
    seeds: Sequence[int],
    spatial_rows: Sequence[Mapping[str, object]],
    phase_rows: Sequence[Mapping[str, object]],
    hour_rows: Sequence[Mapping[str, object]],
) -> None:
    lines = [
        "# 机制级误差分析（自动生成）", "",
        "> 差值定义均为“消融模型误差 − GTSRNet误差”；正值表示对应组件有效。", "",
        "## 实验完整性", "",
        f"- 空间分析数据集：{', '.join(spatial_datasets)}",
        f"- 逐小时周期分析数据集：{', '.join(phase_datasets)}",
        f"- 独立随机种子：{', '.join(str(x) for x in seeds)}",
        "- 每个种子均单独输出图片，没有把三个种子的图合并。", "",
        "## 空间分支", "",
        "| 数据集 | MAE降低率（跨种子） | RMSE降低率（跨种子） | MAE改善网格占比 |", "|---|---:|---:|---:|",
    ]
    for dataset in spatial_datasets:
        subset = [row for row in spatial_rows if row["dataset"] == dataset]
        lines.append(
            f"| {dataset} | {mean_std([float(x['mae_reduction_percent']) for x in subset])}% "
            f"| {mean_std([float(x['rmse_reduction_percent']) for x in subset])}% "
            f"| {mean_std([float(x['percent_grids_improved_mae']) for x in subset])}% |"
        )
    lines += ["", "## 周期嵌入", "", "| 数据集 | MAE降低率（跨种子） | RMSE降低率（跨种子） | MAE改善小时数 |", "|---|---:|---:|---:|"]
    for dataset in phase_datasets:
        subset = [row for row in phase_rows if row["dataset"] == dataset]
        lines.append(
            f"| {dataset} | {mean_std([float(x['mae_reduction_percent']) for x in subset])}% "
            f"| {mean_std([float(x['rmse_reduction_percent']) for x in subset])}% "
            f"| {mean_std([float(x['hours_improved_mae']) for x in subset])}/24 |"
        )
        hour_means = []
        for hour in range(24):
            values = [float(row["delta_mae"]) for row in hour_rows if row["dataset"] == dataset and int(row["hour"]) == hour]
            hour_means.append((hour, float(np.mean(values))))
        positive = [hour for hour, value in hour_means if value > 0]
        negative = [hour for hour, value in hour_means if value <= 0]
        lines.append("")
        lines.append(f"- **{dataset}**：跨种子平均后，MAE改善小时={positive or '无'}；未改善小时={negative or '无'}。")
    lines += [
        "", "## 解释边界", "",
        "- 逐网格差值图直接回答空间分支在哪些网格有效；不能用GTSRNet自身的误差图替代。",
        "- 逐小时差值图回答周期嵌入在哪些目标小时有效。只应声称结果实际支持的小时和数据集。",
        "- 若某些网格或小时差值为负，应作为失败模式如实报告，不要删除。",
        "- 三个种子的单独图位于 `per_seed/seed_*/<dataset>/`，正文可选择代表图，全部图片放补充材料。",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parser().parse_args()
    results = args.results_dir.expanduser().resolve()
    analysis = (args.analysis_dir.expanduser().resolve() if args.analysis_dir else results / "analysis")
    analysis.mkdir(parents=True, exist_ok=True)
    spatial_summaries: List[Dict[str, object]] = []
    phase_summaries: List[Dict[str, object]] = []
    all_hour_rows: List[Dict[str, object]] = []
    for seed in args.seeds:
        for dataset in args.spatial_datasets:
            full_arrays, full_payload = read_run(run_dir(results, "original_phase", seed, dataset, "GTSRNet"))
            wo_arrays, wo_payload = read_run(run_dir(results, "original_phase", seed, dataset, "wo_s"))
            compare_protocol(full_payload, wo_payload, f"{dataset}/seed_{seed}/wo_s")
            full, wo_s, truth, hours = validate_pair(full_arrays, wo_arrays, f"{dataset}/seed_{seed}/spatial")
            destination = analysis / "per_seed" / f"seed_{seed}" / dataset
            spatial = spatial_analysis(full, wo_s, truth, full_payload, dataset, seed, destination, args.dpi)
            spatial_summaries.append({"dataset": dataset, "seed": seed, **spatial})
            print(f"DONE spatial seed={seed} dataset={dataset}: {destination}", flush=True)
        for dataset in args.phase_datasets:
            full_arrays, full_payload = read_run(run_dir(results, "original_phase", seed, dataset, "GTSRNet"))
            none_arrays, none_payload = read_run(run_dir(results, "no_periodic", seed, dataset, "GTSRNet"))
            compare_protocol(full_payload, none_payload, f"{dataset}/seed_{seed}/no_periodic")
            full2, no_phase, truth2, hours2 = validate_pair(full_arrays, none_arrays, f"{dataset}/seed_{seed}/phase")
            destination = analysis / "per_seed" / f"seed_{seed}" / dataset
            phase, hour_rows = phase_analysis(full2, no_phase, truth2, hours2, dataset, seed, destination, args.dpi)
            phase_summaries.append({"dataset": dataset, "seed": seed, **phase})
            all_hour_rows.extend(hour_rows)
            print(f"DONE phase seed={seed} dataset={dataset}: {destination}", flush=True)
    write_csv(analysis / "spatial_summary_per_seed.csv", spatial_summaries, tuple(spatial_summaries[0].keys()))
    write_csv(analysis / "phase_summary_per_seed.csv", phase_summaries, tuple(phase_summaries[0].keys()))
    write_csv(analysis / "phase_hour_all_seeds.csv", all_hour_rows, tuple(all_hour_rows[0].keys()))
    write_report(
        analysis / "MANUSCRIPT_RESULTS_TEMPLATE_CN.md",
        args.spatial_datasets,
        args.phase_datasets,
        args.seeds,
        spatial_summaries,
        phase_summaries,
        all_hour_rows,
    )
    manifest = {
        "difference_definition": "ablation_error_minus_GTSRNet_error",
        "positive_means": "component_helps",
        "spatial_datasets": list(args.spatial_datasets),
        "phase_datasets": list(args.phase_datasets),
        "seeds": list(args.seeds),
        "per_seed_figures_only": True,
        "truth_and_protocol_validation": "passed",
    }
    (analysis / "analysis_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"Analysis complete: {analysis}", flush=True)


if __name__ == "__main__":
    main()
