"""Test cross-grid predictive dependence in the five traffic datasets.

The analysis deliberately removes each grid's hour-of-day mean and linear
trend before testing cross-grid relationships.  It then combines:

1. maximum lagged cross-correlation (lags 1..L) with independent circular-
   shift permutation tests; and
2. a Granger-style nested-regression F test: does source-grid history improve
   target-grid prediction beyond the target grid's own history?

Both families are corrected with Benjamini-Hochberg FDR.  Significant results
support statistical cross-grid predictive dependence, not physical causality.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
import numpy as np
from scipy.stats import f as f_distribution


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_PROJECT_ROOT = SCRIPT_DIR.parent
DATASETS = ("milan_sms", "milan_call", "milan_internet", "mobile_bj", "mobile_nj")
DISPLAY = {
    "milan_sms": "Milan SMS",
    "milan_call": "Milan Call",
    "milan_internet": "Milan Internet",
    "mobile_bj": "MobileBJ",
    "mobile_nj": "MobileNJ",
}
SERVICE_INDEX = {"milan_sms": 0, "milan_call": 1, "milan_internet": 2}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=DEFAULT_PROJECT_ROOT)
    parser.add_argument("--datasets", nargs="+", choices=DATASETS, default=DATASETS)
    parser.add_argument("--output-dir", type=Path, default=Path("results_dataset_interactions"))
    parser.add_argument("--top-nodes", type=int, default=24)
    parser.add_argument("--max-lag", type=int, default=6)
    parser.add_argument("--granger-lags", type=int, default=6)
    parser.add_argument("--permutations", type=int, default=500)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--max-network-edges", type=int, default=30)
    parser.add_argument("--milan-rows", nargs=2, type=int, default=(40, 60))
    parser.add_argument("--milan-cols", nargs=2, type=int, default=(40, 60))
    parser.add_argument("--dpi", type=int, default=300)
    return parser


def resolve(root: Path, path: Path) -> Path:
    path = path.expanduser()
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def parse_milan_hours(values: np.ndarray) -> np.ndarray:
    hours: List[int] = []
    for value in values:
        if isinstance(value, bytes):
            value = value.decode("utf-8")
        text = str(value)
        parsed = None
        for start in (11, 8):
            token = text[start : start + 2]
            if len(token) == 2 and token.isdigit() and 0 <= int(token) <= 23:
                parsed = int(token)
                break
        if parsed is None:
            raise ValueError(f"Cannot parse hour from Milan timestamp {text!r}")
        hours.append(parsed)
    return np.asarray(hours, dtype=np.int64)


def load_dataset(
    root: Path,
    name: str,
    milan_rows: Sequence[int],
    milan_cols: Sequence[int],
) -> Tuple[np.ndarray, np.ndarray, List[Tuple[int, int]], Path]:
    if name.startswith("milan_"):
        try:
            import h5py
        except ImportError as exc:
            raise RuntimeError("Milan analysis requires h5py; install requirements.txt") from exc
        path = root / "dataset" / "data_git_version.h5"
        if not path.is_file():
            raise FileNotFoundError(path)
        with h5py.File(path, "r") as source:
            raw = np.asarray(source["data"][:, :, SERVICE_INDEX[name]], dtype=np.float64)
            hours = parse_milan_hours(np.asarray(source["idx"]))
        if raw.ndim != 2 or raw.shape[1] != 10000:
            raise ValueError(f"Expected Milan [time,10000], got {raw.shape}")
        grid = raw.reshape(-1, 100, 100)
        r0, r1 = map(int, milan_rows)
        c0, c1 = map(int, milan_cols)
        if not (0 <= r0 < r1 <= 100 and 0 <= c0 < c1 <= 100):
            raise ValueError("Invalid Milan crop")
        grid = grid[:, r0:r1, c0:c1]
        coords = [(row, col) for row in range(r0, r1) for col in range(c0, c1)]
        return grid.reshape(len(grid), -1), hours, coords, path

    if name == "mobile_bj":
        path, height, width = root / "dataset" / "MobileBJ.npy", 28, 24
    elif name == "mobile_nj":
        path, height, width = root / "dataset" / "MobileNJ.npy", 20, 28
    else:
        raise ValueError(name)
    if not path.is_file():
        raise FileNotFoundError(path)
    raw = np.load(path, allow_pickle=False)
    if raw.ndim == 3 and raw.shape[-1] == 1:
        raw = raw[..., 0]
    cells = height * width
    if raw.ndim != 2:
        raise ValueError(f"Expected 2-D mobile array, got {raw.shape}")
    if raw.shape[1] == cells:
        time_by_grid = raw
    elif raw.shape[0] == cells:
        time_by_grid = raw.T
    else:
        raise ValueError(f"Neither axis of {raw.shape} matches {height}x{width}")
    coords = [(row, col) for row in range(height) for col in range(width)]
    hours = np.arange(len(time_by_grid), dtype=np.int64) % 24
    return np.asarray(time_by_grid, dtype=np.float64), hours, coords, path


def replace_nonfinite(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64).copy()
    for column in range(values.shape[1]):
        finite = np.isfinite(values[:, column])
        replacement = float(np.median(values[finite, column])) if np.any(finite) else 0.0
        values[~finite, column] = replacement
    return values


def remove_daily_pattern_and_trend(
    traffic: np.ndarray, hours: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """Remove each cell's own daily mean and linear trend, then standardize."""

    traffic = replace_nonfinite(traffic)
    seasonal = np.zeros((24, traffic.shape[1]), dtype=np.float64)
    global_mean = traffic.mean(axis=0)
    for hour in range(24):
        mask = hours == hour
        seasonal[hour] = traffic[mask].mean(axis=0) if np.any(mask) else global_mean
    residual = traffic - seasonal[hours]
    time_axis = np.linspace(-1.0, 1.0, len(residual), dtype=np.float64)
    centered_time = time_axis - time_axis.mean()
    slope = (centered_time[:, None] * residual).sum(axis=0) / np.square(centered_time).sum()
    residual = residual - centered_time[:, None] * slope[None, :]
    residual = residual - residual.mean(axis=0, keepdims=True)
    scale = residual.std(axis=0, ddof=1)
    valid = scale > 1e-10
    standardized = np.zeros_like(residual)
    standardized[:, valid] = residual[:, valid] / scale[valid]
    return standardized, scale


def select_nodes(
    standardized: np.ndarray,
    residual_scale: np.ndarray,
    coords: Sequence[Tuple[int, int]],
    count: int,
) -> Tuple[np.ndarray, List[Tuple[int, int]], np.ndarray]:
    valid = np.flatnonzero(residual_scale > 1e-10)
    if len(valid) < 2:
        raise ValueError("Fewer than two non-constant grids remain after preprocessing")
    selected = valid[np.argsort(residual_scale[valid])[::-1][: min(count, len(valid))]]
    return standardized[:, selected], [coords[index] for index in selected], selected


def correlation_matrix(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    left = left - left.mean(axis=0, keepdims=True)
    right = right - right.mean(axis=0, keepdims=True)
    left_scale = np.sqrt(np.square(left).sum(axis=0))
    right_scale = np.sqrt(np.square(right).sum(axis=0))
    denominator = np.maximum(left_scale[:, None] * right_scale[None, :], 1e-12)
    return (left.T @ right) / denominator


def bh_fdr(p_values: np.ndarray, include: np.ndarray) -> np.ndarray:
    q_values = np.ones_like(p_values, dtype=np.float64)
    flat_indices = np.flatnonzero(include.ravel())
    if not len(flat_indices):
        return q_values
    flat_p = p_values.ravel()[flat_indices]
    order = np.argsort(flat_p)
    ranked = flat_p[order]
    adjusted = ranked * len(ranked) / np.arange(1, len(ranked) + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    adjusted = np.clip(adjusted, 0.0, 1.0)
    restored = np.empty_like(adjusted)
    restored[order] = adjusted
    q_values.ravel()[flat_indices] = restored
    return q_values


def lagged_permutation_test(
    series: np.ndarray,
    max_lag: int,
    permutations: int,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Rows are sources, columns targets; source(t-lag) predicts target(t)."""

    nodes = series.shape[1]
    best_abs = np.zeros((nodes, nodes), dtype=np.float64)
    best_corr = np.zeros((nodes, nodes), dtype=np.float64)
    best_lag = np.ones((nodes, nodes), dtype=np.int64)
    for lag in range(1, max_lag + 1):
        corr = correlation_matrix(series[:-lag], series[lag:])
        update = np.abs(corr) > best_abs
        best_abs[update] = np.abs(corr[update])
        best_corr[update] = corr[update]
        best_lag[update] = lag

    exceed = np.zeros((nodes, nodes), dtype=np.int64)
    length = len(series)
    minimum_shift = min(max(24, max_lag + 1), max(1, length // 3))
    for permutation in range(permutations):
        shifted = np.empty_like(series)
        for node in range(nodes):
            if length > 2 * minimum_shift + 1:
                shift = int(rng.integers(minimum_shift, length - minimum_shift))
            else:
                shift = int(rng.integers(1, length))
            shifted[:, node] = np.roll(series[:, node], shift)
        null_max = np.zeros((nodes, nodes), dtype=np.float64)
        for lag in range(1, max_lag + 1):
            null_max = np.maximum(
                null_max,
                np.abs(correlation_matrix(shifted[:-lag], series[lag:])),
            )
        exceed += null_max >= (best_abs - 1e-12)
        if (permutation + 1) % max(1, permutations // 10) == 0:
            print(f"  permutation {permutation + 1}/{permutations}", flush=True)
    p_values = (exceed + 1.0) / (permutations + 1.0)
    np.fill_diagonal(p_values, 1.0)
    include = ~np.eye(nodes, dtype=bool)
    q_values = bh_fdr(p_values, include)
    return best_corr, best_lag, p_values, q_values


def lag_design(series: np.ndarray, lags: int, node: int) -> np.ndarray:
    length = len(series)
    return np.column_stack(
        [series[lags - lag : length - lag, node] for lag in range(1, lags + 1)]
    )


def regression_rss(design: np.ndarray, target: np.ndarray) -> float:
    coefficients, _, _, _ = np.linalg.lstsq(design, target, rcond=None)
    residual = target - design @ coefficients
    return float(residual @ residual)


def granger_style_test(
    series: np.ndarray, lags: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    nodes = series.shape[1]
    observations = len(series) - lags
    if observations <= 2 * lags + 2:
        raise ValueError("Not enough observations for requested Granger lags")
    f_values = np.zeros((nodes, nodes), dtype=np.float64)
    gains = np.zeros((nodes, nodes), dtype=np.float64)
    p_values = np.ones((nodes, nodes), dtype=np.float64)
    ones = np.ones((observations, 1), dtype=np.float64)
    lag_cache = [lag_design(series, lags, node) for node in range(nodes)]
    for target_node in range(nodes):
        target = series[lags:, target_node]
        restricted_design = np.column_stack((ones, lag_cache[target_node]))
        restricted_rss = regression_rss(restricted_design, target)
        for source_node in range(nodes):
            if source_node == target_node:
                continue
            full_design = np.column_stack(
                (ones, lag_cache[target_node], lag_cache[source_node])
            )
            full_rss = regression_rss(full_design, target)
            numerator = max(0.0, (restricted_rss - full_rss) / lags)
            degrees = observations - full_design.shape[1]
            denominator = max(full_rss / max(degrees, 1), 1e-15)
            statistic = numerator / denominator
            f_values[source_node, target_node] = statistic
            gains[source_node, target_node] = max(
                0.0, (restricted_rss - full_rss) / max(restricted_rss, 1e-15)
            )
            p_values[source_node, target_node] = float(
                f_distribution.sf(statistic, lags, degrees)
            )
    include = ~np.eye(nodes, dtype=bool)
    q_values = bh_fdr(p_values, include)
    return f_values, gains, p_values, q_values


def labels(coords: Sequence[Tuple[int, int]]) -> List[str]:
    return [f"{index + 1}" for index in range(len(coords))]


def save_heatmap(
    matrix: np.ndarray,
    coords: Sequence[Tuple[int, int]],
    title: str,
    colorbar_label: str,
    path: Path,
    dpi: int,
    cmap: str = "coolwarm",
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
) -> None:
    node_labels = labels(coords)
    fig, axis = plt.subplots(figsize=(9.2, 8.0), constrained_layout=True)
    image = axis.imshow(matrix, cmap=cmap, aspect="equal", vmin=vmin, vmax=vmax)
    axis.set_xticks(range(len(node_labels)), node_labels, rotation=90, fontsize=7)
    axis.set_yticks(range(len(node_labels)), node_labels, fontsize=7)
    axis.set_xlabel("Target grid ID")
    axis.set_ylabel("Source grid ID")
    axis.set_title(title)
    colorbar = fig.colorbar(image, ax=axis, shrink=0.82)
    colorbar.set_label(colorbar_label)
    fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def save_period_correlations(
    series: np.ndarray,
    hours: np.ndarray,
    coords: Sequence[Tuple[int, int]],
    title: str,
    path: Path,
    dpi: int,
) -> None:
    periods = (("Night (00–05)", range(0, 6)), ("Morning (07–09)", range(7, 10)), ("Evening (17–19)", range(17, 20)))
    node_labels = labels(coords)
    fig, axes = plt.subplots(1, 3, figsize=(17, 5.4), constrained_layout=True)
    last_image = None
    for axis, (period_name, period_hours) in zip(axes, periods):
        mask = np.isin(hours, list(period_hours))
        correlation = correlation_matrix(series[mask], series[mask])
        np.fill_diagonal(correlation, np.nan)
        last_image = axis.imshow(correlation, cmap="coolwarm", vmin=-1, vmax=1, aspect="equal")
        axis.set_xticks(range(len(node_labels)), node_labels, rotation=90, fontsize=6)
        axis.set_yticks(range(len(node_labels)), node_labels, fontsize=6)
        axis.set_title(period_name)
        axis.set_xlabel("Target grid ID")
    axes[0].set_ylabel("Source grid ID")
    fig.suptitle(title)
    colorbar = fig.colorbar(last_image, ax=axes, shrink=0.82)
    colorbar.set_label("Deseasonalized Pearson correlation")
    fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def save_network(
    coords: Sequence[Tuple[int, int]],
    gains: np.ndarray,
    q_values: np.ndarray,
    alpha: float,
    max_edges: int,
    title: str,
    path: Path,
    dpi: int,
) -> None:
    significant = np.argwhere((q_values < alpha) & (~np.eye(len(coords), dtype=bool)))
    ordered = sorted(
        ((int(source), int(target), float(gains[source, target])) for source, target in significant),
        key=lambda item: item[2],
        reverse=True,
    )[:max_edges]
    fig, axis = plt.subplots(figsize=(9, 8), constrained_layout=True)
    columns = np.asarray([coord[1] for coord in coords], dtype=np.float64)
    rows = np.asarray([coord[0] for coord in coords], dtype=np.float64)
    axis.scatter(columns, rows, s=90, color="#2166ac", edgecolors="white", linewidths=0.8, zorder=3)
    maximum = max((gain for _, _, gain in ordered), default=1.0)
    for source, target, gain in ordered:
        width = 0.5 + 3.0 * gain / max(maximum, 1e-12)
        arrow = FancyArrowPatch(
            (columns[source], rows[source]),
            (columns[target], rows[target]),
            arrowstyle="-|>", mutation_scale=9, linewidth=width,
            color="#b2182b", alpha=0.30 + 0.55 * gain / max(maximum, 1e-12),
            connectionstyle="arc3,rad=0.08", zorder=2,
        )
        axis.add_patch(arrow)
    for index, (column, row) in enumerate(zip(columns, rows), start=1):
        axis.text(column, row, str(index), color="white", fontsize=6, ha="center", va="center", zorder=4)
    axis.set_xlabel("Grid column")
    axis.set_ylabel("Grid row")
    axis.set_title(title + f"\nTop {len(ordered)} FDR-significant directed links")
    axis.invert_yaxis()
    axis.grid(alpha=0.2, linestyle="--")
    fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def write_csv(path: Path, rows: Iterable[Mapping[str, object]], fields: Sequence[str]) -> None:
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def analyze_dataset(args: argparse.Namespace, root: Path, output_root: Path, name: str) -> Dict[str, object]:
    print(f"\n=== {name} ===", flush=True)
    traffic, hours, all_coords, source_path = load_dataset(
        root, name, args.milan_rows, args.milan_cols
    )
    standardized, residual_scale = remove_daily_pattern_and_trend(traffic, hours)
    series, selected_coords, selected_indices = select_nodes(
        standardized, residual_scale, all_coords, args.top_nodes
    )
    print(
        f"file={source_path}, time_steps={len(series)}, all_grids={traffic.shape[1]}, "
        f"selected_grids={series.shape[1]}",
        flush=True,
    )
    dataset_output = output_root / name
    dataset_output.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed + DATASETS.index(name))

    best_corr, best_lag, lag_p, lag_q = lagged_permutation_test(
        series, args.max_lag, args.permutations, rng
    )
    f_values, granger_gain, granger_p, granger_q = granger_style_test(
        series, args.granger_lags
    )
    off_diagonal = ~np.eye(series.shape[1], dtype=bool)
    lag_significant = (lag_q < args.alpha) & off_diagonal
    granger_significant = (granger_q < args.alpha) & off_diagonal
    both_significant = lag_significant & granger_significant

    node_rows = [
        {
            "grid_id": index + 1,
            "flat_index_in_dataset": int(selected_indices[index]),
            "row": int(coord[0]),
            "column": int(coord[1]),
            "residual_std": float(residual_scale[selected_indices[index]]),
        }
        for index, coord in enumerate(selected_coords)
    ]
    write_csv(
        dataset_output / "selected_grids.csv",
        node_rows,
        ("grid_id", "flat_index_in_dataset", "row", "column", "residual_std"),
    )

    pair_rows = []
    for source in range(series.shape[1]):
        for target in range(series.shape[1]):
            if source == target:
                continue
            pair_rows.append(
                {
                    "source_id": source + 1,
                    "source_grid": str(selected_coords[source]),
                    "target_id": target + 1,
                    "target_grid": str(selected_coords[target]),
                    "best_lag_hours": int(best_lag[source, target]),
                    "lagged_correlation": float(best_corr[source, target]),
                    "lag_permutation_p": float(lag_p[source, target]),
                    "lag_fdr_q": float(lag_q[source, target]),
                    "lag_significant": bool(lag_significant[source, target]),
                    "granger_f": float(f_values[source, target]),
                    "incremental_r2": float(granger_gain[source, target]),
                    "granger_p": float(granger_p[source, target]),
                    "granger_fdr_q": float(granger_q[source, target]),
                    "granger_significant": bool(granger_significant[source, target]),
                    "both_tests_significant": bool(both_significant[source, target]),
                }
            )
    pair_rows.sort(key=lambda row: (not row["both_tests_significant"], -float(row["incremental_r2"])))
    write_csv(
        dataset_output / "directed_grid_pairs.csv",
        pair_rows,
        (
            "source_id", "source_grid", "target_id", "target_grid", "best_lag_hours",
            "lagged_correlation", "lag_permutation_p", "lag_fdr_q", "lag_significant",
            "granger_f", "incremental_r2", "granger_p", "granger_fdr_q",
            "granger_significant", "both_tests_significant",
        ),
    )

    masked_corr = np.where(lag_significant, best_corr, 0.0)
    np.fill_diagonal(masked_corr, np.nan)
    save_heatmap(
        masked_corr, selected_coords,
        f"FDR-significant lagged cross-grid dependence — {DISPLAY[name]}",
        "Best lagged correlation", dataset_output / "fig_lagged_interaction_heatmap.png",
        args.dpi, cmap="coolwarm", vmin=-1, vmax=1,
    )
    masked_gain = np.where(granger_significant, 100.0 * granger_gain, 0.0)
    np.fill_diagonal(masked_gain, np.nan)
    save_heatmap(
        masked_gain, selected_coords,
        f"Incremental predictive influence — {DISPLAY[name]}",
        "Incremental explained variance (%)", dataset_output / "fig_granger_interaction_heatmap.png",
        args.dpi, cmap="magma", vmin=0,
    )
    save_period_correlations(
        series, hours, selected_coords,
        f"Time-varying cross-grid dependence — {DISPLAY[name]}",
        dataset_output / "fig_period_correlation_matrices.png", args.dpi,
    )
    save_network(
        selected_coords, granger_gain, granger_q, args.alpha,
        args.max_network_edges, f"Cross-grid predictive interaction — {DISPLAY[name]}",
        dataset_output / "fig_interaction_network.png", args.dpi,
    )

    possible_edges = int(off_diagonal.sum())
    summary: Dict[str, object] = {
        "dataset": name,
        "display_name": DISPLAY[name],
        "source_file": str(source_path),
        "time_steps": int(len(series)),
        "all_grids": int(traffic.shape[1]),
        "selected_grids": int(series.shape[1]),
        "selection_rule": "largest residual standard deviation after daily-pattern/trend removal",
        "max_lag": int(args.max_lag),
        "granger_lags": int(args.granger_lags),
        "permutations": int(args.permutations),
        "fdr_alpha": float(args.alpha),
        "possible_directed_edges": possible_edges,
        "significant_lagged_edges": int(lag_significant.sum()),
        "significant_lagged_ratio": float(lag_significant.sum() / possible_edges),
        "significant_granger_edges": int(granger_significant.sum()),
        "significant_granger_ratio": float(granger_significant.sum() / possible_edges),
        "significant_in_both_tests": int(both_significant.sum()),
        "both_tests_ratio": float(both_significant.sum() / possible_edges),
        "maximum_incremental_r2": float(np.max(granger_gain[off_diagonal])),
        "median_significant_incremental_r2": (
            float(np.median(granger_gain[granger_significant]))
            if np.any(granger_significant) else 0.0
        ),
        "interpretation": (
            "Evidence of cross-grid predictive dependence after removing each grid's daily pattern and trend. "
            "This is not a claim of physical causality."
            if np.any(both_significant)
            else "No edge passed both corrected tests under the current settings."
        ),
    }
    (dataset_output / "interaction_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(
        f"significant lagged={summary['significant_lagged_edges']}/{possible_edges}, "
        f"granger={summary['significant_granger_edges']}/{possible_edges}, "
        f"both={summary['significant_in_both_tests']}/{possible_edges}",
        flush=True,
    )
    return summary


def validate(args: argparse.Namespace) -> None:
    if args.top_nodes < 2:
        raise ValueError("--top-nodes must be at least 2")
    if args.max_lag < 1 or args.granger_lags < 1:
        raise ValueError("lag counts must be positive")
    if args.permutations < 99:
        raise ValueError("Use at least 99 permutations")
    if not 0.0 < args.alpha < 1.0:
        raise ValueError("--alpha must be between 0 and 1")


def main() -> None:
    args = build_parser().parse_args()
    validate(args)
    root = args.project_root.expanduser().resolve()
    output = resolve(root, args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    summaries = [analyze_dataset(args, root, output, name) for name in args.datasets]
    fields = (
        "dataset", "display_name", "source_file", "time_steps", "all_grids", "selected_grids",
        "max_lag", "granger_lags", "permutations", "fdr_alpha", "possible_directed_edges",
        "significant_lagged_edges", "significant_lagged_ratio", "significant_granger_edges",
        "significant_granger_ratio", "significant_in_both_tests", "both_tests_ratio",
        "maximum_incremental_r2", "median_significant_incremental_r2", "interpretation",
    )
    write_csv(output / "interaction_summary_all_datasets.csv", summaries, fields)
    (output / "analysis_config.json").write_text(
        json.dumps(
            {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
            ensure_ascii=False, indent=2,
        ),
        encoding="utf-8",
    )
    print(f"\nDONE. Results: {output}", flush=True)


if __name__ == "__main__":
    main()
