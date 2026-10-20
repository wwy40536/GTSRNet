"""Plot one baseline-comparison figure for every dataset and random seed.

Unlike summarize_baselines.py, this script never averages trainable-model runs.
The deterministic Seasonal Historical Average (S-HA) result is repeated as a
reference in every seed figure because it has no random initialization.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Mapping, Sequence, Tuple

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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=Path("results_GTSRNet_baselines"))
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--models", nargs="+", choices=DEFAULT_MODELS, default=DEFAULT_MODELS)
    parser.add_argument("--datasets", nargs="+", choices=DEFAULT_DATASETS, default=DEFAULT_DATASETS)
    parser.add_argument("--seeds", nargs="+", type=int, default=(16, 26, 36))
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--formats", nargs="+", choices=("png", "pdf"), default=("png",))
    return parser


def read_metrics(path: Path) -> Dict[str, float]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    metrics = payload["metrics"]
    return {"rmse": float(metrics["rmse"]), "mae": float(metrics["mae"])}


def collect_seed(
    results_dir: Path,
    dataset: str,
    seed: int,
    models: Sequence[str],
) -> Tuple[Dict[str, Mapping[str, float]], Sequence[str]]:
    values: Dict[str, Mapping[str, float]] = {}
    missing = []
    for model in models:
        path = results_dir / dataset / model / f"seed_{seed}" / "metrics.json"
        if model == "seasonal_ha" and not path.is_file():
            candidates = sorted((results_dir / dataset / model).glob("seed_*/metrics.json"))
            if candidates:
                path = candidates[0]
        if not path.is_file():
            missing.append(f"{dataset}/{model}/seed_{seed}")
            continue
        values[model] = read_metrics(path)
    return values, missing


def add_value_labels(axis, bars) -> None:
    for bar in bars:
        height = float(bar.get_height())
        axis.annotate(
            f"{height:.4g}",
            xy=(bar.get_x() + bar.get_width() / 2.0, height),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=7,
            rotation=45,
        )


def plot_seed(
    dataset: str,
    seed: int,
    values: Mapping[str, Mapping[str, float]],
    models: Sequence[str],
    output_stem: Path,
    dpi: int,
    formats: Sequence[str],
) -> None:
    try:
        import matplotlib
    except ModuleNotFoundError:
        plot_seed_with_pillow(dataset, seed, values, models, output_stem, dpi, formats)
        return

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    available = [model for model in models if model in values]
    if not available:
        return
    labels = [DISPLAY[model] for model in available]
    x = np.arange(len(available))
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8), constrained_layout=True)
    for axis, metric, title in ((axes[0], "rmse", "RMSE"), (axes[1], "mae", "MAE")):
        scores = [float(values[model][metric]) for model in available]
        colors = ["#C35A4A" if model == "GTSRNet" else "#4F77AA" for model in available]
        bars = axis.bar(x, scores, color=colors, edgecolor="white")
        axis.set_xticks(x, labels, rotation=30, ha="right")
        axis.set_ylabel(title)
        axis.grid(axis="y", linestyle="--", alpha=0.25)
        axis.margins(y=0.18)
        add_value_labels(axis, bars)
    readable_dataset = dataset.replace("_", " ").title()
    fig.suptitle(f"Baseline comparison — {readable_dataset} — seed {seed}")
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    for file_format in formats:
        output = output_stem.with_suffix(f".{file_format}")
        fig.savefig(output, dpi=dpi, bbox_inches="tight", facecolor="white")
        print(f"WROTE {output}", flush=True)
    plt.close(fig)


def plot_seed_with_pillow(
    dataset: str,
    seed: int,
    values: Mapping[str, Mapping[str, float]],
    models: Sequence[str],
    output_stem: Path,
    dpi: int,
    formats: Sequence[str],
) -> None:
    """Dependency-light fallback used when Matplotlib is unavailable."""
    from PIL import Image, ImageDraw, ImageFont

    available = [model for model in models if model in values]
    if not available:
        return
    width, height = 2500, 960
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default(size=26)
    small = ImageFont.load_default(size=21)
    title_font = ImageFont.load_default(size=38)
    title = f"Baseline comparison - {dataset.replace('_', ' ').title()} - seed {seed}"
    title_box = draw.textbbox((0, 0), title, font=title_font)
    draw.text(((width - (title_box[2] - title_box[0])) / 2, 25), title, fill="#222222", font=title_font)

    for panel, (metric, metric_name) in enumerate((("rmse", "RMSE"), ("mae", "MAE"))):
        left = 110 + panel * 1240
        right = left + 1110
        top, bottom = 135, 770
        scores = [float(values[model][metric]) for model in available]
        maximum = max(scores) * 1.18 if scores else 1.0
        draw.line((left, top, left, bottom), fill="#333333", width=3)
        draw.line((left, bottom, right, bottom), fill="#333333", width=3)
        draw.text((left, 90), metric_name, fill="#222222", font=font)
        for tick in range(6):
            y = bottom - (bottom - top) * tick / 5
            value = maximum * tick / 5
            draw.line((left, y, right, y), fill="#DDDDDD", width=1)
            draw.text((left - 100, y - 12), f"{value:.3g}", fill="#555555", font=small)
        slot = (right - left) / len(available)
        bar_width = slot * 0.62
        for index, (model, score) in enumerate(zip(available, scores)):
            center = left + slot * (index + 0.5)
            bar_top = bottom - (bottom - top) * score / maximum
            color = "#C35A4A" if model == "GTSRNet" else "#4F77AA"
            draw.rectangle((center - bar_width / 2, bar_top, center + bar_width / 2, bottom), fill=color)
            score_text = f"{score:.4g}"
            score_box = draw.textbbox((0, 0), score_text, font=small)
            draw.text((center - (score_box[2] - score_box[0]) / 2, bar_top - 31), score_text, fill="#222222", font=small)
            label = DISPLAY[model]
            label_box = draw.textbbox((0, 0), label, font=small)
            draw.text((center - (label_box[2] - label_box[0]) / 2, bottom + 18), label, fill="#222222", font=small)

    output_stem.parent.mkdir(parents=True, exist_ok=True)
    for file_format in formats:
        output = output_stem.with_suffix(f".{file_format}")
        if file_format == "pdf":
            image.save(output, "PDF", resolution=dpi)
        else:
            image.save(output, dpi=(dpi, dpi))
        print(f"WROTE {output} (Pillow fallback)", flush=True)


def main() -> None:
    args = build_parser().parse_args()
    results_dir = args.results_dir.expanduser().resolve()
    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else results_dir / "summary" / "figures_by_seed"
    )
    missing_all = []
    count = 0
    for seed in args.seeds:
        for dataset in args.datasets:
            values, missing = collect_seed(results_dir, dataset, seed, args.models)
            missing_all.extend(missing)
            plot_seed(
                dataset=dataset,
                seed=seed,
                values=values,
                models=args.models,
                output_stem=output_dir / f"seed_{seed}" / f"fig_baselines_{dataset}_seed_{seed}",
                dpi=args.dpi,
                formats=args.formats,
            )
            count += 1
    if missing_all:
        print("WARNING: missing runs (figures contain only available models):", flush=True)
        for item in missing_all:
            print(f"  {item}", flush=True)
    print(f"DONE: generated {count} dataset/seed figures under {output_dir}", flush=True)


if __name__ == "__main__":
    main()
