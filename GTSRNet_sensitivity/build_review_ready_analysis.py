"""Build robustness tables and readable two-scale CDFs without matplotlib."""

from __future__ import annotations

import argparse
import csv
import html
import json
from pathlib import Path
from typing import Dict, List, Mapping, Sequence, Tuple

import numpy as np


COLORS = (
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b",
    "#e377c2", "#7f7f7f", "#bcbd22", "#17becf", "#3366cc", "#e6550d",
)


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description="Prepare reviewer-ready tail analysis.")
    result.add_argument("--predictions-dir", type=Path, required=True)
    result.add_argument("--output-dir", type=Path, required=True)
    result.add_argument("--datasets", nargs="+", default=None)
    result.add_argument("--zoom-quantile", type=float, default=0.995)
    return result


def error_metrics(error: np.ndarray) -> Tuple[float, float]:
    return float(np.mean(np.abs(error))), float(np.sqrt(np.mean(np.square(error))))


def write_csv(path: Path, rows: Sequence[Mapping[str, object]], fields: Sequence[str]) -> None:
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def curve(values: np.ndarray, points: int = 1600) -> Tuple[np.ndarray, np.ndarray]:
    ordered = np.sort(np.asarray(values, dtype=np.float64).reshape(-1))
    indices = np.unique(np.linspace(0, len(ordered) - 1, min(points, len(ordered))).astype(int))
    return ordered[indices], (indices + 1) / len(ordered)


def svg_path(x: np.ndarray, y: np.ndarray, left: float, top: float, width: float, height: float,
             xmax: float, log_scale: bool) -> str:
    if log_scale:
        denominator = np.log10(1.0 + xmax)
        px = left + np.log10(1.0 + np.clip(x, 0, xmax)) / denominator * width
    else:
        px = left + np.clip(x, 0, xmax) / xmax * width
    py = top + (1.0 - np.clip(y, 0, 1)) * height
    return " ".join(("M" if index == 0 else "L") + f"{a:.2f},{b:.2f}" for index, (a, b) in enumerate(zip(px, py)))


def nice_ticks(maximum: float, count: int = 5) -> List[float]:
    if maximum <= 0:
        return [0.0]
    raw = maximum / count
    magnitude = 10 ** np.floor(np.log10(raw))
    step = min((1, 2, 5, 10), key=lambda value: abs(raw - value * magnitude)) * magnitude
    ticks = list(np.arange(0.0, maximum + step, step))
    return [float(value) for value in ticks if value <= maximum * 1.001]


def cdf_svg(dataset: str, errors: Sequence[np.ndarray], output: Path, zoom_quantile: float) -> None:
    pooled = np.concatenate([np.asarray(values).reshape(-1) for values in errors])
    zoom_max = float(np.quantile(pooled, zoom_quantile))
    full_max = float(np.max(pooled))
    p95, p99 = (float(np.quantile(pooled, value)) for value in (0.95, 0.99))
    width, height = 1600, 760
    panel_w, panel_h, top = 620, 500, 105
    panel_lefts = (105, 850)
    pieces = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<style>text{font-family:Arial,sans-serif;fill:#222}.tick{font-size:17px}.label{font-size:21px}.title{font-size:28px;font-weight:600}.subtitle{font-size:20px;font-weight:600}</style>',
        f'<text x="800" y="42" text-anchor="middle" class="title">Absolute-error empirical CDF — {html.escape(dataset.replace("_", " ").title())}</text>',
    ]
    for panel, (left, xmax, log_scale, title) in enumerate(
        ((panel_lefts[0], zoom_max, False, f"Central distribution (0–P{zoom_quantile * 100:g})"),
         (panel_lefts[1], full_max, True, "Full tail (log-scaled x-axis)"))
    ):
        pieces.append(f'<text x="{left + panel_w / 2}" y="78" text-anchor="middle" class="subtitle">{title}</text>')
        for value in np.linspace(0, 1, 6):
            y = top + (1 - value) * panel_h
            pieces.append(f'<line x1="{left}" y1="{y}" x2="{left + panel_w}" y2="{y}" stroke="#dddddd" stroke-dasharray="5 5"/>')
            pieces.append(f'<text x="{left - 12}" y="{y + 6}" text-anchor="end" class="tick">{value:.1f}</text>')
        if log_scale:
            xticks = [0.0] + [10.0 ** power for power in range(0, int(np.ceil(np.log10(full_max))) + 1) if 10.0 ** power <= full_max]
            positions = [left + np.log10(1 + value) / np.log10(1 + full_max) * panel_w for value in xticks]
        else:
            xticks = nice_ticks(xmax)
            positions = [left + value / xmax * panel_w for value in xticks]
        for value, x in zip(xticks, positions):
            pieces.append(f'<line x1="{x}" y1="{top}" x2="{x}" y2="{top + panel_h}" stroke="#eeeeee"/>')
            pieces.append(f'<text x="{x}" y="{top + panel_h + 27}" text-anchor="middle" class="tick">{value:g}</text>')
        pieces.append(f'<rect x="{left}" y="{top}" width="{panel_w}" height="{panel_h}" fill="none" stroke="#222" stroke-width="2"/>')
        if not log_scale:
            for value, label in ((p95, "P95"), (p99, "P99")):
                x = left + min(value, xmax) / xmax * panel_w
                pieces.append(f'<line x1="{x}" y1="{top}" x2="{x}" y2="{top + panel_h}" stroke="#555" stroke-dasharray="8 5"/>')
                pieces.append(f'<text x="{x + 5}" y="{top + 22}" class="tick">{label}</text>')
        for step, values in enumerate(errors):
            x_values, probabilities = curve(values)
            pieces.append(
                f'<path d="{svg_path(x_values, probabilities, left, top, panel_w, panel_h, xmax, log_scale)}" '
                f'fill="none" stroke="{COLORS[step % len(COLORS)]}" stroke-width="2" opacity="0.9"/>'
            )
        pieces.append(f'<text x="{left + panel_w / 2}" y="{top + panel_h + 60}" text-anchor="middle" class="label">Absolute error</text>')
    pieces.append('<text x="26" y="355" transform="rotate(-90 26 355)" text-anchor="middle" class="label">Empirical CDF</text>')
    legend_y = 710
    for step in range(len(errors)):
        x = 105 + (step % 6) * 125
        y = legend_y + (step // 6) * 28
        pieces.append(f'<line x1="{x}" y1="{y - 5}" x2="{x + 30}" y2="{y - 5}" stroke="{COLORS[step]}" stroke-width="3"/>')
        pieces.append(f'<text x="{x + 37}" y="{y}" class="tick">t+{step + 1}</text>')
    pieces.append('</svg>')
    output.write_text("\n".join(pieces), encoding="utf-8")


def main() -> None:
    args = parser().parse_args()
    if not 0.9 < args.zoom_quantile < 1.0:
        raise ValueError("--zoom-quantile must be between 0.9 and 1")
    source, output = args.predictions_dir.resolve(), args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    event_rows: List[Dict[str, object]] = []
    hotspot_rows: List[Dict[str, object]] = []
    cases: Dict[str, object] = {}
    paths = sorted(source.glob("*/predictions.npz"))
    if args.datasets:
        paths = [path for path in paths if path.parent.name in args.datasets]
    for path in paths:
        dataset = path.parent.name
        archive = np.load(path, allow_pickle=False)
        prediction = archive["pred"].astype(np.float64)
        truth = archive["truth"].astype(np.float64)
        error = prediction - truth
        absolute = np.abs(error)
        base_mae, base_rmse = error_metrics(error)
        timestamps = archive["target_time"]
        unique_times = np.unique(timestamps)
        time_sse = []
        for timestamp in unique_times:
            mask2 = timestamps == timestamp
            mask5 = np.broadcast_to(mask2[:, :, None, None, None], error.shape)
            time_sse.append((float(np.sum(np.square(error[mask5]))), str(timestamp), mask5))
        _sse, worst_time, mask = max(time_sse, key=lambda item: item[0])
        kept = ~mask
        robust_mae, robust_rmse = error_metrics(error[kept])
        element_fraction = float(mask.sum() / mask.size * 100.0)
        mae_share = float(absolute[mask].sum() / absolute.sum() * 100.0)
        sse_share = float(np.square(error[mask]).sum() / np.square(error).sum() * 100.0)
        event_rows.append({
            "dataset": dataset, "excluded_target_time": worst_time,
            "excluded_element_percent": element_fraction,
            "original_mae": base_mae, "robust_mae": robust_mae,
            "mae_change_percent": (robust_mae / base_mae - 1) * 100,
            "original_rmse": base_rmse, "robust_rmse": robust_rmse,
            "rmse_change_percent": (robust_rmse / base_rmse - 1) * 100,
            "event_absolute_error_share_percent": mae_share,
            "event_sse_share_percent": sse_share,
        })
        grid_mae = absolute.mean(axis=(0, 1, 2))
        worst_grid = np.unravel_index(int(np.argmax(grid_mae)), grid_mae.shape)
        grid_mask = np.zeros_like(error, dtype=bool)
        grid_mask[:, :, :, worst_grid[0], worst_grid[1]] = True
        grid_sse_share = float(np.square(error[grid_mask]).sum() / np.square(error).sum() * 100.0)
        hotspot_rows.append({
            "dataset": dataset, "crop_row": int(worst_grid[0]), "crop_col": int(worst_grid[1]),
            "source_row": int(worst_grid[0] + 40), "source_col": int(worst_grid[1] + 40),
            "worst_grid_mae": float(grid_mae[worst_grid]),
            "median_grid_mae": float(np.median(grid_mae)),
            "worst_to_median_mae_ratio": float(grid_mae[worst_grid] / np.median(grid_mae)),
            "worst_grid_sse_share_percent": grid_sse_share,
        })
        max_index = np.unravel_index(int(np.argmax(absolute)), absolute.shape)
        cases[dataset] = {
            "worst_target_time_by_total_sse": worst_time,
            "maximum_error_target_time": str(timestamps[max_index[0], max_index[1]]),
            "maximum_error_forecast_step": int(max_index[1] + 1),
            "maximum_error_source_grid": [int(max_index[3] + 40), int(max_index[4] + 40)],
            "maximum_absolute_error": float(absolute[max_index]),
            "prediction": float(prediction[max_index]), "truth": float(truth[max_index]),
        }
        cdf_svg(dataset, [absolute[:, step] for step in range(absolute.shape[1])], output / f"fig_cdf_two_scale_{dataset}.svg", args.zoom_quantile)
    write_csv(output / "table_event_robustness.csv", event_rows, tuple(event_rows[0].keys()))
    write_csv(output / "table_spatial_concentration.csv", hotspot_rows, tuple(hotspot_rows[0].keys()))
    (output / "anomaly_cases.json").write_text(json.dumps(cases, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
