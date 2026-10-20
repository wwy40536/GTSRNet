"""Plot horizon-wise error growth relative to forecast step 1.

For every run and horizon h, this script computes
  delta_MAE_h  = MAE_h  - MAE_1
  delta_RMSE_h = RMSE_h - RMSE_1
and the corresponding relative increases in percent. Deltas are computed per
seed before mean/std aggregation so that different initial errors do not bias
the comparison.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Mapping, Sequence, Tuple

import numpy as np


DEFAULT_MODELS = ("seasonal_ha", "gru_seq2seq", "mtgnn", "patchtst", "dlinear", "timemixer", "GTSRNet")
DEFAULT_DATASETS = ("milan_sms", "milan_call", "milan_internet", "mobile_bj", "mobile_nj")
DISPLAY = {
    "seasonal_ha": "S-HA",
    "gru_seq2seq": "GRU-Seq2Seq",
    "mtgnn": "MTGNN",
    "patchtst": "PatchTST",
    "dlinear": "DLinear",
    "timemixer": "TimeMixer",
    "GTSRNet": "GTSRNet",
}
COLORS = {
    "seasonal_ha": "#8C8C8C",
    "gru_seq2seq": "#4C78A8",
    "mtgnn": "#F58518",
    "patchtst": "#54A24B",
    "dlinear": "#B279A2",
    "timemixer": "#72B7B2",
    "GTSRNet": "#C44E52",
}
MARKERS = {
    "seasonal_ha": "o",
    "gru_seq2seq": "s",
    "mtgnn": "^",
    "patchtst": "D",
    "dlinear": "v",
    "timemixer": "P",
    "GTSRNet": "*",
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=Path("results_GTSRNet_baselines"))
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--models", nargs="+", choices=DEFAULT_MODELS, default=DEFAULT_MODELS)
    parser.add_argument("--datasets", nargs="+", choices=DEFAULT_DATASETS, default=DEFAULT_DATASETS)
    parser.add_argument("--seeds", nargs="+", type=int, default=(16, 26, 36))
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--formats", nargs="+", choices=("png", "pdf"), default=("png", "pdf"))
    parser.add_argument("--skip-per-seed", action="store_true")
    return parser


def mean_std(values: Sequence[float]) -> Tuple[float, float]:
    array = np.asarray(values, dtype=np.float64)
    return float(array.mean()), float(array.std(ddof=1)) if len(array) > 1 else 0.0


def collect_runs(results_dir: Path, models: Sequence[str], datasets: Sequence[str], seeds: Sequence[int]) -> List[Dict[str, object]]:
    selected_models, selected_datasets, selected_seeds = set(models), set(datasets), set(seeds)
    rows: List[Dict[str, object]] = []
    for path in sorted(results_dir.glob("*/*/seed_*/metrics.json")):
        dataset, model = path.parents[2].name, path.parents[1].name
        seed = int(path.parent.name.removeprefix("seed_"))
        if dataset not in selected_datasets or model not in selected_models:
            continue
        if model != "seasonal_ha" and seed not in selected_seeds:
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        horizons = payload["metrics"].get("horizons", [])
        if not horizons:
            raise ValueError(f"No per-horizon metrics in {path}")
        by_horizon = sorted(horizons, key=lambda item: int(item["horizon"]))
        base_rmse, base_mae = float(by_horizon[0]["rmse"]), float(by_horizon[0]["mae"])
        if int(by_horizon[0]["horizon"]) != 1:
            raise ValueError(f"First saved horizon is not 1 in {path}")
        for item in by_horizon:
            rmse, mae = float(item["rmse"]), float(item["mae"])
            rows.append(
                {
                    "dataset": dataset,
                    "model": model,
                    "display_name": DISPLAY[model],
                    "seed": seed,
                    "forecast_step": int(item["horizon"]),
                    "rmse": rmse,
                    "mae": mae,
                    "delta_rmse": rmse - base_rmse,
                    "delta_mae": mae - base_mae,
                    "pct_rmse": 100.0 * (rmse - base_rmse) / base_rmse,
                    "pct_mae": 100.0 * (mae - base_mae) / base_mae,
                }
            )
    if not rows:
        raise FileNotFoundError(f"No usable metrics.json files found under {results_dir}")
    return rows


def aggregate(rows: Sequence[Mapping[str, object]]) -> List[Dict[str, object]]:
    groups: Dict[Tuple[str, str, int], List[Mapping[str, object]]] = defaultdict(list)
    for row in rows:
        groups[(str(row["dataset"]), str(row["model"]), int(row["forecast_step"]))].append(row)
    output: List[Dict[str, object]] = []
    metrics = ("rmse", "mae", "delta_rmse", "delta_mae", "pct_rmse", "pct_mae")
    for (dataset, model, step), group in sorted(groups.items()):
        result: Dict[str, object] = {
            "dataset": dataset,
            "model": model,
            "display_name": DISPLAY[model],
            "forecast_step": step,
            "n_seeds": len(group),
        }
        for metric in metrics:
            mean, std = mean_std([float(row[metric]) for row in group])
            result[f"{metric}_mean"] = mean
            result[f"{metric}_std"] = std
        output.append(result)
    return output


def write_csv(path: Path, rows: Sequence[Mapping[str, object]], fields: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    print(f"WROTE {path}", flush=True)


def series_from_summary(
    rows: Sequence[Mapping[str, object]], dataset: str, model: str, metric: str
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    selected = sorted(
        (row for row in rows if row["dataset"] == dataset and row["model"] == model),
        key=lambda row: int(row["forecast_step"]),
    )
    x = np.asarray([int(row["forecast_step"]) for row in selected])
    mean = np.asarray([float(row[f"{metric}_mean"]) for row in selected])
    std = np.asarray([float(row[f"{metric}_std"]) for row in selected])
    return x, mean, std


def plot_matplotlib(
    dataset: str,
    data: Mapping[str, Mapping[str, Tuple[np.ndarray, np.ndarray, np.ndarray]]],
    models: Sequence[str],
    metrics: Sequence[Tuple[str, str]],
    title_suffix: str,
    output_stem: Path,
    dpi: int,
    formats: Sequence[str],
    show_std: bool,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(12.8, 4.8), constrained_layout=True)
    for axis, (metric, ylabel) in zip(axes, metrics):
        for model in models:
            if model not in data or metric not in data[model]:
                continue
            x, mean, std = data[model][metric]
            width = 2.6 if model == "GTSRNet" else 1.5
            size = 8 if model == "GTSRNet" else 4
            axis.plot(x, mean, label=DISPLAY[model], color=COLORS[model], marker=MARKERS[model], linewidth=width, markersize=size)
            if show_std and np.any(std > 0):
                axis.fill_between(x, mean - std, mean + std, color=COLORS[model], alpha=0.10)
        axis.axhline(0.0, color="#333333", linewidth=0.8)
        axis.set_xticks(range(1, 13))
        axis.set_xlabel("Forecast step h")
        axis.set_ylabel(ylabel)
        axis.grid(linestyle="--", alpha=0.25)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, -0.02), ncol=7, frameon=False)
    fig.suptitle(f"Horizon-wise error growth — {dataset.replace('_', ' ').title()} — {title_suffix}")
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    for extension in formats:
        output = output_stem.with_suffix(f".{extension}")
        fig.savefig(output, dpi=dpi, bbox_inches="tight", facecolor="white")
        print(f"WROTE {output}", flush=True)
    plt.close(fig)


def plot_pillow(
    dataset: str,
    data: Mapping[str, Mapping[str, Tuple[np.ndarray, np.ndarray, np.ndarray]]],
    models: Sequence[str],
    metrics: Sequence[Tuple[str, str]],
    title_suffix: str,
    output_stem: Path,
    dpi: int,
    formats: Sequence[str],
    show_std: bool,
) -> None:
    from PIL import Image, ImageDraw, ImageFont

    image = Image.new("RGB", (2600, 1050), "white")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default(size=24)
    small = ImageFont.load_default(size=19)
    title_font = ImageFont.load_default(size=36)
    title = f"Horizon-wise error growth - {dataset.replace('_', ' ').title()} - {title_suffix}"
    box = draw.textbbox((0, 0), title, font=title_font)
    draw.text(((2600 - box[2] + box[0]) / 2, 20), title, fill="#222222", font=title_font)
    for panel, (metric, ylabel) in enumerate(metrics):
        left, right = 120 + panel * 1270, 1200 + panel * 1270
        top, bottom = 130, 790
        all_values = []
        for model in models:
            if model not in data or metric not in data[model]:
                continue
            _, mean_values, std_values = data[model][metric]
            all_values.extend(float(value) for value in mean_values)
            if show_std:
                all_values.extend(float(value) for value in mean_values - std_values)
                all_values.extend(float(value) for value in mean_values + std_values)
        low, high = min(0.0, min(all_values)), max(0.0, max(all_values))
        padding = max((high - low) * 0.12, 1e-9)
        low, high = low - padding, high + padding
        map_x = lambda h: left + (right - left) * (h - 1) / 11
        map_y = lambda value: bottom - (bottom - top) * (value - low) / (high - low)
        for tick in range(6):
            value = low + (high - low) * tick / 5
            y = map_y(value)
            draw.line((left, y, right, y), fill="#DDDDDD", width=1)
            draw.text((left - 100, y - 10), f"{value:.3g}", fill="#555555", font=small)
        draw.line((left, map_y(0), right, map_y(0)), fill="#333333", width=2)
        draw.text((left, 88), ylabel, fill="#222222", font=font)
        for h in range(1, 13):
            x = map_x(h)
            draw.text((x - 7, bottom + 15), str(h), fill="#444444", font=small)
        draw.text(((left + right) / 2 - 70, bottom + 52), "Forecast step h", fill="#222222", font=font)
        for model in models:
            if model not in data or metric not in data[model]:
                continue
            x_values, mean, _ = data[model][metric]
            points = [(map_x(int(h)), map_y(float(value))) for h, value in zip(x_values, mean)]
            if show_std:
                std = data[model][metric][2]
                if np.any(std > 0):
                    upper = [(map_x(int(h)), map_y(float(value + spread))) for h, value, spread in zip(x_values, mean, std)]
                    lower = [(map_x(int(h)), map_y(float(value - spread))) for h, value, spread in zip(x_values, mean, std)]
                    shade = Image.new("RGBA", image.size, (255, 255, 255, 0))
                    shade_draw = ImageDraw.Draw(shade)
                    rgb = tuple(int(COLORS[model][index:index + 2], 16) for index in (1, 3, 5))
                    shade_draw.polygon(upper + list(reversed(lower)), fill=rgb + (28,))
                    image = Image.alpha_composite(image.convert("RGBA"), shade).convert("RGB")
                    draw = ImageDraw.Draw(image)
            width = 6 if model == "GTSRNet" else 3
            draw.line(points, fill=COLORS[model], width=width, joint="curve")
            radius = 7 if model == "GTSRNet" else 4
            for x, y in points:
                draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=COLORS[model])
    legend_y = 930
    legend_width = 310
    start_x = (2600 - legend_width * len(models)) / 2
    for index, model in enumerate(models):
        x = start_x + index * legend_width
        draw.line((x, legend_y, x + 55, legend_y), fill=COLORS[model], width=6 if model == "GTSRNet" else 3)
        draw.text((x + 65, legend_y - 13), DISPLAY[model], fill="#222222", font=small)
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    for extension in formats:
        output = output_stem.with_suffix(f".{extension}")
        image.save(output, "PDF" if extension == "pdf" else "PNG", resolution=dpi, dpi=(dpi, dpi))
        print(f"WROTE {output} (Pillow fallback)", flush=True)


def make_plot(*args, **kwargs) -> None:
    try:
        import matplotlib  # noqa: F401
    except ModuleNotFoundError:
        plot_pillow(*args, **kwargs)
        return
    plot_matplotlib(*args, **kwargs)


def summary_data(summary: Sequence[Mapping[str, object]], dataset: str, models: Sequence[str], metrics: Sequence[str]):
    output = {}
    for model in models:
        if not any(row["dataset"] == dataset and row["model"] == model for row in summary):
            continue
        output[model] = {metric: series_from_summary(summary, dataset, model, metric) for metric in metrics}
    return output


def seed_data(
    rows: Sequence[Mapping[str, object]], dataset: str, seed: int, models: Sequence[str], metrics: Sequence[str]
):
    output = {}
    for model in models:
        selected = [row for row in rows if row["dataset"] == dataset and row["model"] == model and int(row["seed"]) == seed]
        if not selected and model == "seasonal_ha":
            selected = [row for row in rows if row["dataset"] == dataset and row["model"] == model]
        selected.sort(key=lambda row: int(row["forecast_step"]))
        if not selected:
            continue
        x = np.asarray([int(row["forecast_step"]) for row in selected])
        zeros = np.zeros(len(selected), dtype=np.float64)
        output[model] = {
            metric: (x, np.asarray([float(row[metric]) for row in selected]), zeros) for metric in metrics
        }
    return output


def main() -> None:
    args = build_parser().parse_args()
    results_dir = args.results_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve() if args.output_dir else results_dir / "summary" / "horizon_error_growth"
    rows = collect_runs(results_dir, args.models, args.datasets, args.seeds)
    summary = aggregate(rows)
    run_fields = ("dataset", "model", "display_name", "seed", "forecast_step", "rmse", "mae", "delta_rmse", "delta_mae", "pct_rmse", "pct_mae")
    summary_fields = ("dataset", "model", "display_name", "forecast_step", "n_seeds") + tuple(
        f"{metric}_{stat}" for metric in ("rmse", "mae", "delta_rmse", "delta_mae", "pct_rmse", "pct_mae") for stat in ("mean", "std")
    )
    write_csv(output_dir / "horizon_error_growth_by_run.csv", rows, run_fields)
    write_csv(output_dir / "horizon_error_growth_summary.csv", summary, summary_fields)
    plot_specs = (
        (("delta_rmse", "Delta RMSE from h=1"), ("delta_mae", "Delta MAE from h=1"), "absolute", "mean +/- std"),
        (("pct_rmse", "Relative RMSE increase (%)"), ("pct_mae", "Relative MAE increase (%)"), "percent", "mean +/- std"),
    )
    for dataset in args.datasets:
        for left, right, folder, title in plot_specs:
            metrics = (left, right)
            data = summary_data(summary, dataset, args.models, (left[0], right[0]))
            make_plot(dataset, data, args.models, metrics, title, output_dir / f"figures_{folder}_mean_std" / f"fig_{folder}_{dataset}", args.dpi, args.formats, show_std=True)
        if not args.skip_per_seed:
            for seed in args.seeds:
                data = seed_data(rows, dataset, seed, args.models, ("delta_rmse", "delta_mae"))
                make_plot(
                    dataset,
                    data,
                    args.models,
                    (("delta_rmse", "Delta RMSE from h=1"), ("delta_mae", "Delta MAE from h=1")),
                    f"seed {seed}",
                    output_dir / "figures_absolute_by_seed" / f"seed_{seed}" / f"fig_absolute_{dataset}_seed_{seed}",
                    args.dpi,
                    args.formats,
                    show_std=False,
                )
    print(f"DONE: horizon-growth outputs are under {output_dir}", flush=True)


if __name__ == "__main__":
    main()
