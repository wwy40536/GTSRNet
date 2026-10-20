#!/usr/bin/env python3
"""Generate separate correlation-vs-distance figures for Milan traffic phases.

The statistical protocol matches the original motivation experiment:

1. use only the raw training period selected by the chronological split;
2. compute a Pearson correlation matrix separately at each exact hour across days;
3. Fisher-z average the exact-hour matrices within Night, Morning, and Evening;
4. plot pairwise correlation against Manhattan distance as a hexbin density plot.

For every requested traffic channel, the script writes one PNG and one PDF for
each phase instead of combining the three phases in one figure.
"""

from __future__ import annotations

import argparse
import math
import re
from pathlib import Path
from typing import Dict, Mapping, Sequence, Tuple

import h5py
import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm


TRAFFIC_INDEX = {"sms": 0, "call": 1, "internet": 2}
PHASES: Mapping[str, Tuple[int, ...]] = {
    "night": tuple(range(0, 6)),
    "morning": (7, 8, 9),
    "evening": (17, 18, 19),
}
PHASE_TITLES = {
    "night": "Night (00-05)",
    "morning": "Morning (07-09)",
    "evening": "Evening (17-19)",
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Plot Night/Morning/Evening correlation-distance panels separately."
    )
    parser.add_argument(
        "--h5",
        type=Path,
        required=True,
        help="Milan HDF5 file containing data and idx datasets.",
    )
    parser.add_argument(
        "--traffic",
        nargs="+",
        choices=("sms", "call", "internet", "all"),
        default=("sms",),
        help="Traffic channels to process; use 'all' for all three.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("outputs_v2_separate"),
        help="Root directory for generated figures.",
    )
    parser.add_argument("--height", type=int, default=100)
    parser.add_argument("--width", type=int, default=100)
    parser.add_argument(
        "--rows", type=int, nargs=2, default=(40, 60), metavar=("START", "END")
    )
    parser.add_argument(
        "--cols", type=int, nargs=2, default=(40, 60), metavar=("START", "END")
    )
    parser.add_argument("--input-length", type=int, default=24)
    parser.add_argument("--prediction-length", type=int, default=12)
    parser.add_argument("--test-windows", type=int, default=168)
    parser.add_argument("--validation-ratio", type=float, default=0.10)
    parser.add_argument(
        "--split-gap",
        type=int,
        default=-1,
        help="Purged window gap; -1 means prediction-length minus one.",
    )
    parser.add_argument(
        "--aggregate-factor",
        type=int,
        default=1,
        help="Keep 1 for the already-hourly data_git_version.h5 file.",
    )
    parser.add_argument("--dpi", type=int, default=400)
    parser.add_argument(
        "--ymin",
        type=float,
        default=-0.7,
        help="Common lower y-axis limit for all panels (default: -0.7).",
    )
    parser.add_argument(
        "--ymax",
        type=float,
        default=1.2,
        help="Common upper y-axis limit for all panels (default: 1.2).",
    )
    parser.add_argument(
        "--combined-only",
        action="store_true",
        help="Only write the combined three-panel figure.",
    )
    return parser


def decode_timestamp(value: object) -> str:
    if isinstance(value, (bytes, np.bytes_)):
        return value.decode("utf-8")
    return str(value)


def extract_hour(timestamp: str) -> int:
    match = re.search(r"[ T](\d{2}):\d{2}", timestamp)
    if match is None:
        raise ValueError(f"Cannot extract hour from timestamp: {timestamp!r}")
    hour = int(match.group(1))
    if not 0 <= hour <= 23:
        raise ValueError(f"Invalid hour {hour} in timestamp {timestamp!r}")
    return hour


def load_milan_channel(
    path: Path,
    channel: int,
    height: int,
    width: int,
    rows: Sequence[int],
    cols: Sequence[int],
    aggregate_factor: int,
) -> Tuple[np.ndarray, np.ndarray]:
    if aggregate_factor <= 0:
        raise ValueError("aggregate-factor must be positive")
    with h5py.File(path, "r") as source:
        if "data" not in source or "idx" not in source:
            raise KeyError("The H5 file must contain 'data' and 'idx' datasets")
        data = source["data"]
        if data.ndim != 3 or data.shape[2] <= channel:
            raise ValueError(f"Unexpected data shape: {tuple(data.shape)}")
        traffic = np.asarray(data[:, :, channel], dtype=np.float64)
        timestamps = [decode_timestamp(value) for value in np.asarray(source["idx"])]

    if traffic.shape[0] != len(timestamps):
        raise ValueError("The data and idx datasets have different time lengths")
    if traffic.shape[1] != height * width:
        raise ValueError(
            f"Expected {height * width} grid cells, found {traffic.shape[1]}"
        )
    if not np.all(np.isfinite(traffic)):
        raise ValueError("The selected traffic channel contains non-finite values")

    if aggregate_factor > 1:
        usable = (len(traffic) // aggregate_factor) * aggregate_factor
        if usable == 0:
            raise ValueError("Not enough records for the requested aggregation")
        traffic = traffic[:usable].reshape(
            -1, aggregate_factor, height * width
        ).sum(axis=1)
        timestamps = timestamps[:usable:aggregate_factor]

    row_start, row_end = map(int, rows)
    col_start, col_end = map(int, cols)
    if not 0 <= row_start < row_end <= height:
        raise ValueError(f"Invalid row range: {tuple(rows)}")
    if not 0 <= col_start < col_end <= width:
        raise ValueError(f"Invalid column range: {tuple(cols)}")

    grid = traffic.reshape(-1, height, width)
    grid = grid[:, row_start:row_end, col_start:col_end]
    hours = np.asarray([extract_hour(value) for value in timestamps], dtype=np.int64)
    return grid, hours


def training_raw_end(
    time_steps: int,
    input_length: int,
    prediction_length: int,
    test_windows: int,
    validation_ratio: float,
    split_gap: int,
) -> int:
    sample_count = time_steps - input_length - prediction_length + 1
    if sample_count <= test_windows:
        raise ValueError("Not enough windows for the requested test set")
    if not 0.0 < validation_ratio < 1.0:
        raise ValueError("validation-ratio must be between 0 and 1")
    gap = prediction_length - 1 if split_gap < 0 else split_gap
    test_start = sample_count - test_windows
    validation_pool_end = test_start - gap
    validation_count = max(1, int(math.floor(validation_pool_end * validation_ratio)))
    validation_start = validation_pool_end - validation_count
    train_count = validation_start - gap
    if train_count <= 0:
        raise ValueError("The split leaves no training windows")
    return train_count + input_length + prediction_length - 1


def correlation_matrix(values: np.ndarray) -> np.ndarray:
    if values.ndim != 2 or values.shape[0] < 3:
        raise ValueError(f"Need at least 3 observations, got {values.shape}")
    centered = values - values.mean(axis=0, keepdims=True)
    norms = np.sqrt(np.sum(centered * centered, axis=0))
    valid = norms > 1e-12
    corr = np.full((values.shape[1], values.shape[1]), np.nan, dtype=np.float64)
    numerator = centered[:, valid].T @ centered[:, valid]
    denominator = np.outer(norms[valid], norms[valid])
    corr[np.ix_(valid, valid)] = numerator / np.maximum(denominator, 1e-12)
    corr = np.clip(corr, -1.0, 1.0)
    valid_indices = np.flatnonzero(valid)
    corr[valid_indices, valid_indices] = 1.0
    return corr


def fisher_average(correlations: Sequence[np.ndarray]) -> np.ndarray:
    stack = np.stack(correlations, axis=0)
    clipped = np.clip(stack, -1.0 + 1e-7, 1.0 - 1e-7)
    z_values = np.arctanh(clipped)
    finite = np.isfinite(stack)
    counts = finite.sum(axis=0)
    summed = np.where(finite, z_values, 0.0).sum(axis=0)
    averaged_z = np.full(stack.shape[1:], np.nan, dtype=np.float64)
    np.divide(summed, counts, out=averaged_z, where=counts > 0)
    averaged = np.tanh(averaged_z)
    diagonal = np.arange(averaged.shape[0])
    valid_diagonal = counts[diagonal, diagonal] > 0
    averaged[diagonal[valid_diagonal], diagonal[valid_diagonal]] = 1.0
    return np.clip(averaged, -1.0, 1.0)


def phase_correlations(
    flat_train: np.ndarray, train_hours: np.ndarray
) -> Dict[str, np.ndarray]:
    result: Dict[str, np.ndarray] = {}
    for phase, selected_hours in PHASES.items():
        exact_hour_matrices = []
        for hour in selected_hours:
            mask = train_hours == hour
            if int(mask.sum()) < 3:
                raise ValueError(f"Hour {hour} has fewer than 3 training observations")
            exact_hour_matrices.append(correlation_matrix(flat_train[mask]))
        result[phase] = fisher_average(exact_hour_matrices)
    return result


def grid_coordinates(height: int, width: int) -> np.ndarray:
    return np.asarray([(row, col) for row in range(height) for col in range(width)])


def pair_data(
    correlation: np.ndarray, coordinates: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    pairwise_distance = np.abs(
        coordinates[:, None, :] - coordinates[None, :, :]
    ).sum(axis=-1)
    upper = np.triu_indices(len(coordinates), k=1)
    distance = pairwise_distance[upper].astype(np.float64)
    corr = correlation[upper]
    finite = np.isfinite(corr)
    return distance[finite], corr[finite]


def binned_median(
    distance: np.ndarray, correlation: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    unique = np.unique(distance.astype(np.int64))
    medians = np.asarray(
        [np.nanmedian(correlation[distance == value]) for value in unique],
        dtype=np.float64,
    )
    return unique, medians


def set_plot_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "legend.fontsize": 9,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def save_phase_figure(
    phase: str,
    correlation: np.ndarray,
    coordinates: np.ndarray,
    output_dir: Path,
    dpi: int,
    ymin: float,
    ymax: float,
) -> None:
    distance, pair_corr = pair_data(correlation, coordinates)
    max_distance = int(
        np.abs(coordinates[:, None, :] - coordinates[None, :, :]).sum(axis=-1).max()
    )

    fig, ax = plt.subplots(figsize=(5.0, 4.0), constrained_layout=True)
    density = ax.hexbin(
        distance,
        pair_corr,
        gridsize=(max(18, max_distance), 30),
        extent=(0, max_distance, -1, 1),
        mincnt=1,
        bins="log",
        cmap="Blues",
    )
    bins, medians = binned_median(distance, pair_corr)
    ax.plot(bins, medians, color="#b2182b", linewidth=1.8, label="Median")
    ax.axhline(0, color="0.5", linewidth=0.8)
    ax.set_xlim(0, max_distance)
    ax.set_ylim(ymin, ymax)
    ax.set_title(PHASE_TITLES[phase])
    ax.set_xlabel("Manhattan distance")
    ax.set_ylabel("Pearson correlation")
    ax.legend(frameon=False, loc="upper right")
    colorbar = fig.colorbar(density, ax=ax, shrink=0.90, pad=0.03)
    colorbar.set_label("log10(pair count)")

    output_base = output_dir / f"correlation_vs_distance_{phase}"
    fig.savefig(output_base.with_suffix(".png"), dpi=dpi, bbox_inches="tight")
    fig.savefig(output_base.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def save_combined_figure(
    correlations: Mapping[str, np.ndarray],
    coordinates: np.ndarray,
    output_dir: Path,
    dpi: int,
    ymin: float,
    ymax: float,
) -> None:
    """Write one paper-ready Night/Morning/Evening three-panel figure."""
    max_distance = int(
        np.abs(coordinates[:, None, :] - coordinates[None, :, :])
        .sum(axis=-1)
        .max()
    )

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(13.2, 4.25),
        sharex=True,
        sharey=True,
        constrained_layout=True,
    )
    densities = []
    for index, (phase, ax) in enumerate(zip(PHASES, axes)):
        distance, pair_corr = pair_data(correlations[phase], coordinates)
        density = ax.hexbin(
            distance,
            pair_corr,
            gridsize=(max(18, max_distance), 30),
            extent=(0, max_distance, -1, 1),
            mincnt=1,
            norm=LogNorm(vmin=1),
            cmap="Blues",
        )
        densities.append(density)

        bins, medians = binned_median(distance, pair_corr)
        ax.plot(
            bins,
            medians,
            color="#b2182b",
            linewidth=1.8,
            label="Median",
            zorder=4,
        )
        ax.axhline(0, color="0.5", linewidth=0.8, zorder=3)
        ax.set_xlim(0, max_distance)
        ax.set_ylim(ymin, ymax)
        ax.set_title(PHASE_TITLES[phase])
        ax.set_xlabel("Manhattan distance")
        if index == 0:
            ax.set_ylabel("Pearson correlation")
        ax.legend(frameon=False, loc="upper right")

    # Force all three hexbins to use the same density scale so their colors are
    # directly comparable, then attach one shared color bar.
    global_max = max(float(np.max(item.get_array())) for item in densities)
    for item in densities:
        item.set_clim(1, max(1.0, global_max))
    colorbar = fig.colorbar(densities[-1], ax=axes, shrink=0.88, pad=0.02)
    colorbar.set_label("Pair count (log scale)")

    output_dir.mkdir(parents=True, exist_ok=True)
    output_base = output_dir / "correlation_vs_distance_centered"
    fig.savefig(output_base.with_suffix(".png"), dpi=dpi, bbox_inches="tight")
    fig.savefig(output_base.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def normalize_traffic(values: Sequence[str]) -> Tuple[str, ...]:
    if "all" in values:
        return tuple(TRAFFIC_INDEX)
    return tuple(dict.fromkeys(values))


def main() -> None:
    args = build_parser().parse_args()
    if not args.h5.is_file():
        raise FileNotFoundError(f"H5 file not found: {args.h5}")
    if args.dpi <= 0:
        raise ValueError("dpi must be positive")
    if not args.ymin < args.ymax:
        raise ValueError("ymin must be smaller than ymax")

    set_plot_style()
    for traffic in normalize_traffic(args.traffic):
        grid, hours = load_milan_channel(
            args.h5,
            TRAFFIC_INDEX[traffic],
            args.height,
            args.width,
            args.rows,
            args.cols,
            args.aggregate_factor,
        )
        train_end = training_raw_end(
            len(grid),
            args.input_length,
            args.prediction_length,
            args.test_windows,
            args.validation_ratio,
            args.split_gap,
        )
        flat_train = grid[:train_end].reshape(train_end, -1)
        correlations = phase_correlations(flat_train, hours[:train_end])
        coordinates = grid_coordinates(grid.shape[1], grid.shape[2])

        traffic_dir = args.output_root / f"motivation_{traffic}"
        save_combined_figure(
            correlations,
            coordinates,
            traffic_dir,
            args.dpi,
            args.ymin,
            args.ymax,
        )

        if not args.combined_only:
            separate_dir = traffic_dir / "separate_panels"
            separate_dir.mkdir(parents=True, exist_ok=True)
            for phase, correlation in correlations.items():
                save_phase_figure(
                    phase,
                    correlation,
                    coordinates,
                    separate_dir,
                    args.dpi,
                    args.ymin,
                    args.ymax,
                )
            print(
                f"[{traffic}] wrote the combined figure and 3 separate panels "
                f"to {traffic_dir.resolve()}"
            )
        else:
            print(
                f"[{traffic}] wrote correlation_vs_distance_centered.png/.pdf "
                f"to {traffic_dir.resolve()}"
            )


if __name__ == "__main__":
    main()
