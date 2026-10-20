"""Training-only ACF/PACF and period analysis for the five traffic datasets."""

from __future__ import annotations

import argparse
import csv
import warnings
from pathlib import Path
from typing import Dict, List, Mapping, Sequence, Tuple

import numpy as np


DATASETS = ("milan_sms", "milan_call", "milan_internet", "mobile_bj", "mobile_nj")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Analyze hourly dependence using only the pre-test portion of each dataset."
    )
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parent.parent)
    parser.add_argument("--output-dir", type=Path, default=Path("results_GTSRNet_sensitivity/dependence"))
    parser.add_argument("--datasets", nargs="+", choices=DATASETS, default=DATASETS)
    parser.add_argument("--max-lag", type=int, default=72)
    parser.add_argument("--max-period", type=int, default=168)
    parser.add_argument("--mobile-step-hours", type=float, default=1.0)
    parser.add_argument("--rows", nargs=2, type=int, default=(40, 60))
    parser.add_argument("--cols", nargs=2, type=int, default=(40, 60))
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--no-plot", action="store_true", help="Write CSV files only.")
    return parser


def resolve_under(root: Path, value: Path) -> Path:
    value = value.expanduser()
    return value.resolve() if value.is_absolute() else (root / value).resolve()


def decode_timestamps(values: np.ndarray) -> np.ndarray:
    texts = [item.decode("utf-8") if isinstance(item, bytes) else str(item) for item in values]
    try:
        return np.asarray(texts, dtype="datetime64[m]")
    except ValueError as exc:
        raise ValueError("Milan timestamps cannot be parsed as minute-resolution datetimes") from exc


def validate_hourly_timestamps(values: np.ndarray) -> float:
    timestamps = decode_timestamps(values)
    deltas = np.diff(timestamps).astype("timedelta64[m]").astype(np.int64)
    if len(deltas) == 0 or np.any(deltas <= 0):
        raise ValueError("Milan timestamps must be strictly increasing")
    median_minutes = float(np.median(deltas))
    if not np.all(deltas == 60):
        counts = {int(value): int(np.sum(deltas == value)) for value in np.unique(deltas)}
        warnings.warn(
            f"Milan cadence is not uniformly hourly (minute deltas: {counts}); "
            "window lengths must be reported as steps unless this is resolved."
        )
    return median_minutes / 60.0


def load_milan(project_root: Path, dataset: str, rows: Sequence[int], cols: Sequence[int]) -> Tuple[np.ndarray, float]:
    try:
        import h5py
    except ImportError as exc:
        raise RuntimeError("h5py is required to read dataset/data_git_version.h5") from exc
    path = project_root / "dataset" / "data_git_version.h5"
    if not path.is_file():
        raise FileNotFoundError(path)
    channel = {"milan_sms": 0, "milan_call": 1, "milan_internet": 2}[dataset]
    with h5py.File(str(path), "r") as source:
        data = np.asarray(source["data"][:, :, channel], dtype=np.float64)
        timestamps = np.asarray(source["idx"])
    if data.ndim != 2 or data.shape[1] != 100 * 100:
        raise ValueError(f"Expected Milan [time,10000], got {data.shape}")
    row_start, row_end = rows
    col_start, col_end = cols
    if not (0 <= row_start < row_end <= 100 and 0 <= col_start < col_end <= 100):
        raise ValueError("Invalid Milan crop")
    data = data.reshape(-1, 100, 100)[:, row_start:row_end, col_start:col_end]
    return data.reshape(len(data), -1), validate_hourly_timestamps(timestamps)


def load_mobile(project_root: Path, dataset: str, step_hours: float) -> Tuple[np.ndarray, float]:
    filename, height, width = {
        "mobile_bj": ("MobileBJ.npy", 28, 24),
        "mobile_nj": ("MobileNJ.npy", 20, 28),
    }[dataset]
    path = project_root / "dataset" / filename
    if not path.is_file():
        raise FileNotFoundError(path)
    raw = np.load(path, allow_pickle=False)
    if raw.ndim == 3 and raw.shape[-1] == 1:
        raw = raw[..., 0]
    cells = height * width
    if raw.ndim != 2:
        raise ValueError(f"Expected 2-D mobile data, got {raw.shape}")
    if raw.shape[1] == cells:
        data = raw
    elif raw.shape[0] == cells:
        data = raw.T
    else:
        raise ValueError(f"Neither axis of {raw.shape} equals {cells} cells")
    return np.asarray(data, dtype=np.float64), step_hours


def load_dataset(args: argparse.Namespace, project_root: Path, dataset: str) -> Tuple[np.ndarray, float, int]:
    if dataset.startswith("milan_"):
        data, step_hours = load_milan(project_root, dataset, args.rows, args.cols)
        test_steps = 24 * 7
    else:
        data, step_hours = load_mobile(project_root, dataset, args.mobile_step_hours)
        test_steps = 24 if dataset == "mobile_bj" else 24 * 7
    if step_hours <= 0:
        raise ValueError("Step duration must be positive")
    if abs(step_hours - 1.0) > 1e-9:
        warnings.warn(
            f"{dataset}: one stored step is {step_hours:g} hours; requested lags are "
            "converted from hours and may not equal the paper's current step counts."
        )
    if len(data) <= test_steps + args.max_lag + 2:
        raise ValueError(f"{dataset} is too short after reserving {test_steps} test steps")
    data = data[:-test_steps]
    if not np.all(np.isfinite(data)):
        raise ValueError(f"{dataset} contains NaN or infinite values")
    return data, step_hours, test_steps


def acf_columns(data: np.ndarray, max_lag: int) -> np.ndarray:
    centered = data - data.mean(axis=0, keepdims=True)
    denominator = np.sum(centered * centered, axis=0)
    valid = denominator > np.finfo(np.float64).eps
    if not np.any(valid):
        raise ValueError("All spatial series are constant")
    centered = centered[:, valid]
    denominator = denominator[valid]
    result = np.empty((max_lag + 1, centered.shape[1]), dtype=np.float64)
    result[0] = 1.0
    for lag in range(1, max_lag + 1):
        result[lag] = np.sum(centered[:-lag] * centered[lag:], axis=0) / denominator
    return result


def acf_1d(values: np.ndarray, max_lag: int) -> np.ndarray:
    centered = np.asarray(values, dtype=np.float64) - float(np.mean(values))
    denominator = float(np.dot(centered, centered))
    if denominator <= np.finfo(np.float64).eps:
        raise ValueError("Global time series is constant")
    result = np.empty(max_lag + 1, dtype=np.float64)
    result[0] = 1.0
    for lag in range(1, max_lag + 1):
        result[lag] = float(np.dot(centered[:-lag], centered[lag:]) / denominator)
    return result


def pacf_yule_walker(acf: np.ndarray) -> np.ndarray:
    pacf = np.empty_like(acf)
    pacf[0] = 1.0
    for order in range(1, len(acf)):
        indices = np.abs(np.subtract.outer(np.arange(order), np.arange(order)))
        toeplitz = acf[indices]
        target = acf[1 : order + 1]
        try:
            coefficients = np.linalg.solve(toeplitz + np.eye(order) * 1e-10, target)
            pacf[order] = coefficients[-1]
        except np.linalg.LinAlgError:
            pacf[order] = np.nan
    return pacf


def periodogram(values: np.ndarray, step_hours: float, max_period: int) -> List[Dict[str, float]]:
    centered = np.asarray(values, dtype=np.float64) - float(np.mean(values))
    frequencies = np.fft.rfftfreq(len(centered), d=step_hours)
    power = np.abs(np.fft.rfft(centered)) ** 2
    valid = frequencies > 0
    periods = 1.0 / frequencies[valid]
    power = power[valid]
    valid_period = (periods >= 2.0) & (periods <= float(max_period))
    periods = periods[valid_period]
    power = power[valid_period]
    if len(power) == 0:
        return []
    total = float(power.sum())
    order = np.argsort(power)[::-1]
    selected: List[Dict[str, float]] = []
    for index in order:
        period = float(periods[index])
        # Keep distinct peaks rather than adjacent FFT bins of one broad peak.
        if any(abs(period - item["period_hours"]) / item["period_hours"] < 0.03 for item in selected):
            continue
        selected.append(
            {
                "rank": len(selected) + 1,
                "period_hours": period,
                "relative_power": float(power[index] / total) if total else 0.0,
            }
        )
        if len(selected) == 5:
            break
    return selected


def write_csv(path: Path, rows: Sequence[Mapping[str, object]], fields: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    print(f"WROTE {path}", flush=True)


def plot_dependence(dataset: str, rows: Sequence[Mapping[str, object]], output: Path, dpi: int) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    lags = np.asarray([row["lag_hours"] for row in rows], dtype=float)
    median = np.asarray([row["spatial_median_acf"] for row in rows], dtype=float)
    lower = np.asarray([row["spatial_q25_acf"] for row in rows], dtype=float)
    upper = np.asarray([row["spatial_q75_acf"] for row in rows], dtype=float)
    global_acf = np.asarray([row["global_acf"] for row in rows], dtype=float)
    global_pacf = np.asarray([row["global_pacf"] for row in rows], dtype=float)

    fig, axes = plt.subplots(2, 1, figsize=(9.0, 7.0), sharex=True, constrained_layout=True)
    axes[0].fill_between(lags, lower, upper, color="#8DB3E2", alpha=0.35, label="Cell ACF IQR")
    axes[0].plot(lags, median, color="#2E64A2", linewidth=1.8, label="Median cell ACF")
    axes[0].plot(lags, global_acf, color="#B44A4A", linewidth=1.3, label="Global ACF")
    axes[0].set_ylabel("Autocorrelation")
    axes[0].legend(frameon=False, ncol=3, fontsize=9)
    stems = axes[1].stem(lags, global_pacf, markerfmt=" ", basefmt=" ")
    plt.setp(stems.stemlines, color="#4D8B58", linewidth=1.1)
    axes[1].set_ylabel("Global PACF")
    axes[1].set_xlabel("Lag (hours)")
    for axis in axes:
        axis.axhline(0, color="black", linewidth=0.7)
        axis.axvline(24, color="#E18C32", linestyle="--", linewidth=1.2, label="24 h")
        axis.grid(alpha=0.2, linestyle="--")
    fig.suptitle(f"Training-only temporal dependence — {dataset.replace('_', ' ').title()}")
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"WROTE {output}", flush=True)


def main() -> None:
    args = build_parser().parse_args()
    if args.max_lag < 24:
        raise ValueError("--max-lag must be at least 24")
    project_root = args.project_root.expanduser().resolve()
    output_dir = resolve_under(project_root, args.output_dir)
    all_lags: List[Dict[str, object]] = []
    all_periods: List[Dict[str, object]] = []
    summaries: List[Dict[str, object]] = []

    for dataset in args.datasets:
        data, step_hours, test_steps = load_dataset(args, project_root, dataset)
        max_lag_steps = int(round(args.max_lag / step_hours))
        spatial_acf = acf_columns(data, max_lag_steps)
        global_series = data.mean(axis=1)
        global_acf = acf_1d(global_series, max_lag_steps)
        global_pacf = pacf_yule_walker(global_acf)
        rows: List[Dict[str, object]] = []
        for lag_step in range(max_lag_steps + 1):
            row = {
                "dataset": dataset,
                "lag_steps": lag_step,
                "lag_hours": lag_step * step_hours,
                "spatial_median_acf": float(np.median(spatial_acf[lag_step])),
                "spatial_q25_acf": float(np.quantile(spatial_acf[lag_step], 0.25)),
                "spatial_q75_acf": float(np.quantile(spatial_acf[lag_step], 0.75)),
                "global_acf": float(global_acf[lag_step]),
                "global_pacf": float(global_pacf[lag_step]),
            }
            rows.append(row)
            all_lags.append(row)
        peaks = periodogram(global_series, step_hours, args.max_period)
        for peak in peaks:
            all_periods.append({"dataset": dataset, **peak})
        lag24_step = int(round(24.0 / step_hours))
        candidate = spatial_acf[2:]
        strongest_offset = int(np.nanargmax(np.median(candidate, axis=1))) + 2
        summaries.append(
            {
                "dataset": dataset,
                "observations_used": len(data),
                "spatial_cells": data.shape[1],
                "step_hours": step_hours,
                "reserved_test_steps": test_steps,
                "median_acf_24h": float(np.median(spatial_acf[lag24_step])),
                "global_acf_24h": float(global_acf[lag24_step]),
                "global_pacf_24h": float(global_pacf[lag24_step]),
                "strongest_median_acf_lag_hours": strongest_offset * step_hours,
                "strongest_spectral_period_hours": peaks[0]["period_hours"] if peaks else float("nan"),
            }
        )
        if not args.no_plot:
            plot_dependence(dataset, rows, output_dir / "figures" / f"fig_acf_pacf_{dataset}.png", args.dpi)
        print(
            f"{dataset}: pre-test observations={len(data)}, cells={data.shape[1]}, "
            f"step={step_hours:g} h, median ACF(24h)={summaries[-1]['median_acf_24h']:.4f}",
            flush=True,
        )

    write_csv(
        output_dir / "acf_pacf.csv",
        all_lags,
        ("dataset", "lag_steps", "lag_hours", "spatial_median_acf", "spatial_q25_acf", "spatial_q75_acf", "global_acf", "global_pacf"),
    )
    write_csv(
        output_dir / "spectral_peaks.csv",
        all_periods,
        ("dataset", "rank", "period_hours", "relative_power"),
    )
    write_csv(
        output_dir / "dependence_summary.csv",
        summaries,
        ("dataset", "observations_used", "spatial_cells", "step_hours", "reserved_test_steps", "median_acf_24h", "global_acf_24h", "global_pacf_24h", "strongest_median_acf_lag_hours", "strongest_spectral_period_hours"),
    )


if __name__ == "__main__":
    main()
