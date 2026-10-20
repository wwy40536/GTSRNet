"""Diagnose horizon-wise prediction variance collapse from predictions.npz."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", required=True)
    parser.add_argument(
        "--output-prefix",
        required=True,
        help="Output path without extension; .csv and .png are added",
    )
    return parser.parse_args()


def correlation(x, y):
    x = x.astype(np.float64)
    y = y.astype(np.float64)
    x_centered = x - np.mean(x)
    y_centered = y - np.mean(y)
    denominator = np.sqrt(
        np.sum(x_centered ** 2) * np.sum(y_centered ** 2)
    )
    if denominator <= 1e-12:
        return 0.0
    return float(np.sum(x_centered * y_centered) / denominator)


def main():
    args = parse_args()
    archive = np.load(args.predictions)
    required = {"truth", "prediction"}
    missing = required.difference(archive.files)
    if missing:
        raise KeyError(
            "predictions archive is missing keys: {}. Available: {}".format(
                sorted(missing), archive.files
            )
        )

    truth = np.asarray(archive["truth"])
    prediction = np.asarray(archive["prediction"])
    if truth.shape != prediction.shape:
        raise ValueError(
            "truth/prediction shape mismatch: {} vs {}".format(
                truth.shape, prediction.shape
            )
        )
    if truth.ndim < 2:
        raise ValueError("expected arrays with a horizon dimension at axis 1")

    horizon = truth.shape[1]
    rows = []
    for step in range(horizon):
        truth_h = truth[:, step].reshape(-1).astype(np.float64)
        pred_h = prediction[:, step].reshape(-1).astype(np.float64)
        truth_mean = float(np.mean(truth_h))
        pred_mean = float(np.mean(pred_h))
        truth_std = float(np.std(truth_h))
        pred_std = float(np.std(pred_h))
        std_ratio = float(pred_std / max(truth_std, 1e-12))
        variance_ratio = float(
            (pred_std ** 2) / max(truth_std ** 2, 1e-12)
        )
        error = pred_h - truth_h
        rows.append(
            {
                "step": step + 1,
                "truth_mean": truth_mean,
                "prediction_mean": pred_mean,
                "mean_bias": pred_mean - truth_mean,
                "truth_std": truth_std,
                "prediction_std": pred_std,
                "std_ratio": std_ratio,
                "variance_ratio": variance_ratio,
                "correlation": correlation(pred_h, truth_h),
                "rmse": float(np.sqrt(np.mean(error ** 2))),
                "mae": float(np.mean(np.abs(error))),
            }
        )

    prefix = Path(args.output_prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    csv_path = Path(str(prefix) + ".csv")
    png_path = Path(str(prefix) + ".png")

    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    steps = [row["step"] for row in rows]
    truth_std = [row["truth_std"] for row in rows]
    pred_std = [row["prediction_std"] for row in rows]
    std_ratio = [row["std_ratio"] for row in rows]
    corr = [row["correlation"] for row in rows]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), dpi=160)
    axes[0].plot(steps, truth_std, marker="o", linewidth=2, label="Truth std")
    axes[0].plot(steps, pred_std, marker="s", linewidth=2, label="Prediction std")
    axes[0].set_title("Horizon-wise standard deviation")
    axes[0].set_ylabel("Standard deviation")

    axes[1].plot(steps, std_ratio, marker="o", linewidth=2, label="Std ratio")
    axes[1].plot(steps, corr, marker="s", linewidth=2, label="Correlation")
    axes[1].axhline(1.0, color="gray", linestyle="--", alpha=0.6)
    axes[1].set_title("Variance retention and correlation")
    axes[1].set_ylabel("Ratio / correlation")

    for axis in axes:
        axis.set_xlabel("Forecasting step")
        axis.grid(True, alpha=0.25)
        axis.legend()
    fig.tight_layout()
    fig.savefig(str(png_path), bbox_inches="tight")

    late_start = max(0, horizon // 2)
    late_ratio = float(np.mean(std_ratio[late_start:]))
    late_corr = float(np.mean(corr[late_start:]))
    print("Input shape:", truth.shape)
    print("Saved:", csv_path)
    print("Saved:", png_path)
    print("Late-horizon mean std ratio: {:.4f}".format(late_ratio))
    print("Late-horizon mean correlation: {:.4f}".format(late_corr))
    if late_ratio < 0.2:
        print("WARNING: strong late-horizon variance collapse is likely")
    elif late_ratio < 0.5:
        print("CAUTION: late-horizon predictions are substantially under-dispersed")
    else:
        print("No severe late-horizon variance collapse detected by std ratio")


if __name__ == "__main__":
    main()
