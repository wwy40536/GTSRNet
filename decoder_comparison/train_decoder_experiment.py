"""Leakage-safe, directly supervised PMF-vs-RMF decoder experiment."""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader


SCRIPT_DIR = Path(__file__).resolve().parent


def bootstrap_project_root() -> Path:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--project-root", type=Path, default=SCRIPT_DIR.parent)
    known, _ = parser.parse_known_args()
    root = known.project_root.expanduser().resolve()
    for path in (SCRIPT_DIR, root):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))
    return root


PROJECT_ROOT = bootstrap_project_root()

from decoder_model import SegRNNGridDecoderAblation  # noqa: E402
from train_embedding_ablation import (  # noqa: E402
    build_loaders,
    choose_device,
    set_seed,
)


DATASETS = ("milan_sms", "milan_call", "milan_internet", "mobile_bj", "mobile_nj")


class TemporalOnlyModel(nn.Module):
    def __init__(
        self,
        input_len: int,
        horizon: int,
        height: int,
        width: int,
        d_model: int,
        seg_len: int,
        dropout: float,
        decoder_mode: str,
        baseline_mode: str = "last",
        seasonal_period: int = 24,
    ) -> None:
        super().__init__()
        self.horizon = horizon
        self.height = height
        self.width = width
        self.decoder_mode = decoder_mode
        self.decoder = SegRNNGridDecoderAblation(
            Tin=input_len,
            Tout=horizon,
            H=height,
            W=width,
            d_model=d_model,
            seg_len=seg_len,
            dropout=dropout,
            decoder_mode=decoder_mode,
            baseline_mode=baseline_mode,
            seasonal_period=seasonal_period,
        )

    def forward(self, x: torch.Tensor, feedback_noise_std: float = 0.0) -> torch.Tensor:
        flat = self.decoder(x, feedback_noise_std=feedback_noise_std)
        return flat.view(x.shape[0], self.horizon, 1, self.height, self.width)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--decoder", choices=("pmf", "rmf"), required=True)
    parser.add_argument("--dataset", choices=DATASETS, default="milan_call")
    parser.add_argument("--data", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=Path("results_decoder_reviewer"))
    parser.add_argument("--close-size", "--input-len", dest="close_size", type=int, default=24)
    parser.add_argument("--predict-size", "--horizon", dest="predict_size", type=int, default=12)
    parser.add_argument("--seg-len", type=int, default=6)
    parser.add_argument(
        "--baseline", choices=("last", "seasonal"), default="last",
        help="Output baseline shared by PMF and RMF: last value (original) or x(t+h-P).",
    )
    parser.add_argument("--seasonal-period", type=int, default=24)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", "--lr", dest="learning_rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--loss", choices=("l1", "l2"), default="l1")
    parser.add_argument("--patience", type=int, default=40)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--split-mode", choices=("chronological",), default="chronological")
    parser.add_argument("--split-gap", type=int, default=-1)
    parser.add_argument("--milan-loader", choices=("clean",), default="clean")
    parser.add_argument("--test-size", type=int, default=0)
    parser.add_argument("--seed", type=int, default=45)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--save-predictions", action="store_true")
    parser.add_argument(
        "--noise-levels",
        nargs="+",
        type=float,
        default=(0.0, 0.01, 0.03, 0.05, 0.10),
        help="Feedback-noise std values on the normalized scale. PMF is a negative control.",
    )
    parser.add_argument("--noise-seed", type=int, default=20260914)
    parser.add_argument("--latency-warmup", type=int, default=10)
    parser.add_argument("--latency-repeats", type=int, default=50)

    # Shared clean-loader arguments.
    parser.add_argument("--traffic", choices=("sms", "call", "internet"), default="call")
    parser.add_argument("--feature-path", type=Path, default=Path("dataset/crawled_feature.csv"))
    parser.add_argument("--nb-flow", type=int, default=1)
    parser.add_argument("--height", type=int, default=100)
    parser.add_argument("--width", type=int, default=100)
    parser.add_argument("--cluster", type=int, default=3)
    parser.add_argument("--crop", dest="crop", action="store_true")
    parser.add_argument("--no-crop", dest="crop", action="store_false")
    parser.set_defaults(crop=True)
    parser.add_argument("--rows", nargs=2, type=int, default=(40, 60))
    parser.add_argument("--cols", nargs=2, type=int, default=(40, 60))
    parser.add_argument("--cycle-len", type=int, default=24)
    # Unused model fields required only because the loader module has one shared parser API.
    parser.set_defaults(
        variant="GTSRNet",
        embedding_variant="joint",
        branch_aux_weight=0.1,
        n_heads=4,
        e_layers=2,
        d_ff=256,
    )
    return parser


def resolve_output(project_root: Path, output: Path) -> Path:
    output = output.expanduser()
    return output.resolve() if output.is_absolute() else (project_root / output).resolve()


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: Optional[torch.optim.Optimizer] = None,
) -> float:
    training = optimizer is not None
    model.train(training)
    total = 0.0
    samples = 0
    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for x, _calendar, target in loader:
            x = x.to(device=device, dtype=torch.float32, non_blocking=True)
            target = target.to(device=device, dtype=torch.float32, non_blocking=True)
            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
            prediction = model(x)
            loss = criterion(prediction, target)
            if optimizer is not None:
                loss.backward()
                optimizer.step()
            total += float(loss.detach()) * x.shape[0]
            samples += x.shape[0]
    return total / max(samples, 1)


@torch.no_grad()
def collect_predictions(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    feedback_noise_std: float,
    noise_seed: int,
) -> Tuple[np.ndarray, np.ndarray]:
    model.eval()
    torch.manual_seed(noise_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(noise_seed)
    predictions: List[np.ndarray] = []
    truths: List[np.ndarray] = []
    for x, _calendar, target in loader:
        x = x.to(device=device, dtype=torch.float32, non_blocking=True)
        prediction = model(x, feedback_noise_std=feedback_noise_std)
        predictions.append(prediction.cpu().numpy())
        truths.append(np.asarray(target))
    return np.concatenate(predictions, axis=0), np.concatenate(truths, axis=0)


def metric_summary(prediction: np.ndarray, truth: np.ndarray) -> Dict[str, object]:
    error = prediction.astype(np.float64) - truth.astype(np.float64)
    reduce_axes = (0,) + tuple(range(2, error.ndim))
    rmse_h = np.sqrt(np.mean(np.square(error), axis=reduce_axes))
    mae_h = np.mean(np.abs(error), axis=reduce_axes)
    steps = np.arange(1, len(rmse_h) + 1, dtype=np.float64)
    return {
        "rmse": float(np.sqrt(np.mean(np.square(error)))),
        "mae": float(np.mean(np.abs(error))),
        "first_rmse": float(rmse_h[0]),
        "last_rmse": float(rmse_h[-1]),
        "rmse_growth_ratio": float(rmse_h[-1] / max(rmse_h[0], 1e-12)),
        "rmse_growth_slope": float(np.polyfit(steps, rmse_h, 1)[0]) if len(steps) > 1 else 0.0,
        "rmse_h": rmse_h.tolist(),
        "mae_h": mae_h.tolist(),
    }


def delta_summary(noisy: np.ndarray, clean: np.ndarray) -> Dict[str, object]:
    delta = noisy.astype(np.float64) - clean.astype(np.float64)
    reduce_axes = (0,) + tuple(range(2, delta.ndim))
    rmse_h = np.sqrt(np.mean(np.square(delta), axis=reduce_axes))
    mae_h = np.mean(np.abs(delta), axis=reduce_axes)
    return {
        "delta_rmse": float(np.sqrt(np.mean(np.square(delta)))),
        "delta_mae": float(np.mean(np.abs(delta))),
        "delta_rmse_h": rmse_h.tolist(),
        "delta_mae_h": mae_h.tolist(),
    }


@torch.no_grad()
def profile_inference(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    warmup: int,
    repeats: int,
) -> Tuple[float, float]:
    model.eval()
    sample, _calendar, _target = next(iter(loader))
    sample = sample.to(device=device, dtype=torch.float32)
    for _ in range(warmup):
        model(sample)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    for _ in range(repeats):
        model(sample)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        peak_mb = torch.cuda.max_memory_allocated(device) / (1024.0 ** 2)
    else:
        peak_mb = float("nan")
    latency_ms = (time.perf_counter() - started) * 1000.0 / repeats
    return latency_ms, peak_mb


def write_horizon_csv(path: Path, metrics: Dict[str, object]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=("step", "rmse", "mae"))
        writer.writeheader()
        for index, (rmse, mae) in enumerate(zip(metrics["rmse_h"], metrics["mae_h"])):
            writer.writerow({"step": index + 1, "rmse": rmse, "mae": mae})


def write_noise_outputs(
    run_dir: Path,
    decoder: str,
    noise_results: Sequence[Dict[str, object]],
) -> None:
    (run_dir / "noise_sweep.json").write_text(
        json.dumps(list(noise_results), indent=2, ensure_ascii=False), encoding="utf-8"
    )
    overall_fields = (
        "decoder", "noise_std", "rmse", "mae", "first_rmse", "last_rmse",
        "rmse_growth_ratio", "delta_rmse", "delta_mae",
    )
    with (run_dir / "noise_sweep.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=overall_fields, extrasaction="ignore")
        writer.writeheader()
        for result in noise_results:
            writer.writerow({"decoder": decoder, **result})
    horizon_fields = ("decoder", "noise_std", "step", "rmse", "mae", "delta_rmse", "delta_mae")
    with (run_dir / "noise_horizon_metrics.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=horizon_fields)
        writer.writeheader()
        for result in noise_results:
            for index in range(len(result["rmse_h"])):
                writer.writerow(
                    {
                        "decoder": decoder,
                        "noise_std": result["noise_std"],
                        "step": index + 1,
                        "rmse": result["rmse_h"][index],
                        "mae": result["mae_h"][index],
                        "delta_rmse": result["delta_rmse_h"][index],
                        "delta_mae": result["delta_mae_h"][index],
                    }
                )


def main() -> None:
    args = build_parser().parse_args()
    if args.close_size % args.seg_len or args.predict_size % args.seg_len:
        raise ValueError("--seg-len must divide both input and prediction lengths")
    if args.baseline == "seasonal" and not 0 < args.seasonal_period <= args.close_size:
        raise ValueError("--baseline seasonal requires 0 < --seasonal-period <= input length")
    if any(level < 0 for level in args.noise_levels):
        raise ValueError("--noise-levels must be non-negative")
    if args.latency_warmup < 0 or args.latency_repeats <= 0:
        raise ValueError("Invalid latency warmup/repeats")
    project_root = args.project_root.expanduser().resolve()
    device = choose_device(args.device, args.gpu)
    set_seed(args.seed)
    train_loader, val_loader, test_loader, normalizer, height, width = build_loaders(args, device)
    model = TemporalOnlyModel(
        args.close_size, args.predict_size, height, width, args.d_model,
        args.seg_len, args.dropout, args.decoder,
        baseline_mode=args.baseline, seasonal_period=args.seasonal_period,
    ).to(device)
    parameters = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    output_root = resolve_output(project_root, args.output_dir)
    run_dir = output_root / args.dataset / args.decoder / f"H{args.predict_size}" / f"seed_{args.seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = run_dir / "best.pt"
    history_path = run_dir / "training_history.json"

    criterion: nn.Module = nn.L1Loss() if args.loss == "l1" else nn.MSELoss()
    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    milestones = sorted({max(1, args.epochs // 4), max(1, 3 * args.epochs // 4)})
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=0.5)

    if not args.eval_only:
        best_val = float("inf")
        stale = 0
        history: List[Dict[str, float]] = []
        for epoch in range(1, args.epochs + 1):
            started = time.perf_counter()
            train_loss = run_epoch(model, train_loader, criterion, device, optimizer)
            val_loss = run_epoch(model, val_loader, criterion, device)
            scheduler.step()
            history.append(
                {
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "seconds": time.perf_counter() - started,
                }
            )
            if val_loss < best_val:
                best_val = val_loss
                stale = 0
                torch.save(
                    {"model_state": model.state_dict(), "epoch": epoch, "val_loss": val_loss},
                    checkpoint_path,
                )
            else:
                stale += 1
            if epoch == 1 or epoch % args.log_every == 0:
                print(
                    f"[{args.dataset}/{args.decoder}/H{args.predict_size}/seed_{args.seed}] "
                    f"epoch={epoch:03d} train={train_loss:.6f} val={val_loss:.6f}",
                    flush=True,
                )
            if args.patience > 0 and stale >= args.patience:
                print(f"early stop at epoch {epoch}", flush=True)
                break
        history_path.write_text(json.dumps(history, indent=2), encoding="utf-8")
    elif not checkpoint_path.is_file():
        raise FileNotFoundError(f"Missing checkpoint for --eval-only: {checkpoint_path}")

    try:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    except TypeError:
        checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state"])

    clean_norm, truth_norm = collect_predictions(model, test_loader, device, 0.0, args.noise_seed)
    clean = normalizer.inverse_transform(clean_norm)
    truth = normalizer.inverse_transform(truth_norm)
    clean_metrics = metric_summary(clean, truth)
    latency_ms, peak_memory_mb = profile_inference(
        model, test_loader, device, args.latency_warmup, args.latency_repeats
    )
    write_horizon_csv(run_dir / "horizon_metrics.csv", clean_metrics)
    if args.save_predictions:
        np.savez_compressed(run_dir / "predictions.npz", truth=truth, prediction=clean)

    config = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
    }
    summary = {
        "objective": "directly_supervised_temporal_decoder",
        "dataset": args.dataset,
        "decoder": args.decoder,
        "input_len": args.close_size,
        "horizon": args.predict_size,
        "seg_len": args.seg_len,
        "baseline": args.baseline,
        "seasonal_period": args.seasonal_period,
        "seed": args.seed,
        "parameters": parameters,
        "best_epoch": int(checkpoint["epoch"]),
        "best_val_loss": float(checkpoint["val_loss"]),
        "latency_ms_per_batch": latency_ms,
        "peak_cuda_memory_mb": peak_memory_mb,
        "metrics": clean_metrics,
        "config": config,
    }
    (run_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    noise_results: List[Dict[str, object]] = []
    for index, level in enumerate(sorted(set(args.noise_levels))):
        if level == 0.0:
            noisy_norm = clean_norm
        else:
            noisy_norm, _ = collect_predictions(
                model, test_loader, device, level, args.noise_seed + index
            )
        noisy = normalizer.inverse_transform(noisy_norm)
        result = {
            "noise_std": float(level),
            **metric_summary(noisy, truth),
            **delta_summary(noisy, clean),
        }
        noise_results.append(result)
        print(
            f"noise={level:.4f} rmse={result['rmse']:.6f} "
            f"delta_rmse={result['delta_rmse']:.6f}",
            flush=True,
        )
    write_noise_outputs(run_dir, args.decoder, noise_results)
    print(f"DONE: {run_dir}", flush=True)


if __name__ == "__main__":
    main()
