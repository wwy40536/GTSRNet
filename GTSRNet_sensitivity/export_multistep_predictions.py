"""Re-evaluate existing GTSRNet checkpoints and export predictions without retraining."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch

import train_sensitivity as training


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Export raw-unit test predictions from existing GTSRNet best.pt files."
    )
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--datasets", nargs="+", default=None)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="cuda")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=None, help="Optional inference-only override.")
    parser.add_argument("--force", action="store_true")
    return parser


def resolve_under(root: Path, value: Path) -> Path:
    value = value.expanduser()
    return value.resolve() if value.is_absolute() else (root / value).resolve()


def discover_datasets(results_dir: Path) -> List[str]:
    return sorted(
        path.parent.parent.name
        for path in results_dir.glob("*/GTSRNet/best.pt")
        if path.is_file()
    )


def namespace_from_saved_config(
    config: Dict[str, object],
    project_root: Path,
    dataset: str,
    cli: argparse.Namespace,
) -> argparse.Namespace:
    args = training.build_parser().parse_args([])
    for key, value in config.items():
        if hasattr(args, key):
            setattr(args, key, value)
    for key in ("data", "feature_path", "output_dir"):
        value = getattr(args, key, None)
        if value is not None and not isinstance(value, Path):
            setattr(args, key, Path(value))
    args.project_root = project_root
    args.dataset = dataset
    args.variant = "GTSRNet"
    args.device = cli.device
    args.gpu = cli.gpu
    if cli.batch_size is not None:
        args.batch_size = cli.batch_size
    return args


def load_checkpoint(path: Path, device: torch.device) -> Dict[str, object]:
    try:
        return torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=device)


def milan_target_times(
    project_root: Path, sample_count: int, predict_size: int
) -> np.ndarray:
    """Recover the exact target timestamps for the final chronological test windows."""
    import h5py

    with h5py.File(str(project_root / "dataset" / "data_git_version.h5"), "r") as source:
        raw = np.asarray(source["idx"])
    text = np.asarray(
        [item.decode("utf-8") if isinstance(item, bytes) else str(item) for item in raw],
        dtype=str,
    )
    first_origin = len(text) - predict_size - sample_count + 1
    indices = np.arange(first_origin, first_origin + sample_count)[:, None]
    indices = indices + np.arange(predict_size)[None, :]
    if indices.min() < 0 or indices.max() >= len(text):
        raise RuntimeError("Cannot align Milan test predictions with timestamps")
    return text[indices]


def main() -> None:
    cli = build_parser().parse_args()
    project_root = cli.project_root.expanduser().resolve()
    results_dir = resolve_under(project_root, cli.results_dir)
    output_dir = resolve_under(
        project_root,
        cli.output_dir or (results_dir / "multistep_diagnostics"),
    )
    datasets = cli.datasets or discover_datasets(results_dir)
    if not datasets:
        raise FileNotFoundError(f"No */GTSRNet/best.pt found below {results_dir}")

    device = training.choose_device(cli.device, cli.gpu)
    for dataset in datasets:
        variant_dir = results_dir / dataset / "GTSRNet"
        checkpoint_path = variant_dir / "best.pt"
        metrics_path = variant_dir / "metrics.json"
        destination = output_dir / dataset / "predictions.npz"
        manifest_path = output_dir / dataset / "evaluation_manifest.json"
        if destination.is_file() and manifest_path.is_file() and not cli.force:
            print(f"SKIP {dataset}: {destination} already exists", flush=True)
            continue
        if not checkpoint_path.is_file() or not metrics_path.is_file():
            raise FileNotFoundError(
                f"{dataset}: both {checkpoint_path} and {metrics_path} are required"
            )
        saved = json.loads(metrics_path.read_text(encoding="utf-8"))
        args = namespace_from_saved_config(saved["config"], project_root, dataset, cli)
        if int(args.predict_size) != 12:
            raise ValueError(
                f"{dataset}: reviewer requested steps 1-12, checkpoint predicts {args.predict_size}"
            )
        training.set_seed(int(args.seed))
        (
            train_loader,
            val_loader,
            test_loader,
            normalizer,
            height,
            width,
            mase_scale,
        ) = training.build_loaders(args, device)
        del train_loader, val_loader
        model = training.make_model(args, "GTSRNet", height, width).to(device)
        checkpoint = load_checkpoint(checkpoint_path, device)
        if checkpoint.get("variant") != "GTSRNet":
            raise ValueError(f"{checkpoint_path} is not a GTSRNet checkpoint")
        model.load_state_dict(checkpoint["model_state"])
        computed, prediction, truth = training.evaluate_predictions(
            model,
            test_loader,
            device,
            normalizer,
            mase_scale,
            int(args.mase_seasonality),
        )
        original = saved["metrics"]
        rmse_delta = abs(float(computed["rmse"]) - float(original["rmse"]))
        mae_delta = abs(float(computed["mae"]) - float(original["mae"]))
        tolerance = max(1e-6, 1e-6 * max(abs(float(original["rmse"])), 1.0))
        if rmse_delta > tolerance or mae_delta > tolerance:
            raise RuntimeError(
                f"{dataset}: reproduced metrics do not match saved test metrics; "
                f"RMSE delta={rmse_delta:.6g}, MAE delta={mae_delta:.6g}. "
                "Do not analyze predictions from a mismatched split/configuration."
            )
        destination.parent.mkdir(parents=True, exist_ok=True)
        arrays = {"pred": prediction, "truth": truth}
        if dataset.startswith("milan_") and args.split_mode == "chronological":
            arrays["target_time"] = milan_target_times(
                project_root, len(prediction), int(args.predict_size)
            )
        np.savez_compressed(destination, **arrays)
        manifest = {
            "dataset": dataset,
            "source_checkpoint": str(checkpoint_path),
            "source_metrics": str(metrics_path),
            "prediction_file": str(destination),
            "prediction_shape": list(prediction.shape),
            "rmse": float(computed["rmse"]),
            "mae": float(computed["mae"]),
            "saved_rmse": float(original["rmse"]),
            "saved_mae": float(original["mae"]),
            "rmse_delta": rmse_delta,
            "mae_delta": mae_delta,
            "checkpoint_epoch": int(checkpoint["epoch"]),
            "config": saved["config"],
        }
        manifest_path.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(
            f"WROTE {destination} shape={prediction.shape}; "
            f"RMSE={computed['rmse']:.6f}, MAE={computed['mae']:.6f}",
            flush=True,
        )


if __name__ == "__main__":
    main()
