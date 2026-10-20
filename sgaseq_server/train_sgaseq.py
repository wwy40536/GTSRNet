"""Train/evaluate one SGA-Seq run under the existing unified protocol."""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from torch import nn

from full_v2_training_core import (
    PROJECT_ROOT,
    batch_loss,
    build_loaders,
    choose_device,
    evaluate_predictions,
    resolve_under_project,
    set_seed,
)
from sgaseq_model import SGASeq, count_parameters


DATASETS = ("milan_sms", "milan_call", "milan_internet", "mobile_bj", "mobile_nj")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--dataset", choices=DATASETS, default="milan_sms")
    parser.add_argument("--data", type=Path, default=None)
    parser.add_argument("--feature-path", type=Path, default=Path("dataset/crawled_feature.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("results_sgaseq_10seed"))
    parser.add_argument("--traffic", choices=("sms", "call", "internet"), default="sms")

    parser.add_argument("--close-size", type=int, default=24)
    parser.add_argument("--predict-size", type=int, default=12)
    parser.add_argument("--nb-flow", type=int, default=1)
    parser.add_argument("--height", type=int, default=100)
    parser.add_argument("--width", type=int, default=100)
    parser.add_argument("--cluster", type=int, default=3)
    parser.add_argument("--crop", dest="crop", action="store_true")
    parser.add_argument("--no-crop", dest="crop", action="store_false")
    parser.set_defaults(crop=True)
    parser.add_argument("--rows", nargs=2, type=int, default=(40, 60))
    parser.add_argument("--cols", nargs=2, type=int, default=(40, 60))

    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--loss", choices=("l1", "l2"), default="l1")
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--split-mode", choices=("chronological", "random"), default="chronological")
    parser.add_argument("--split-gap", type=int, default=-1)
    parser.add_argument("--milan-loader", choices=("clean", "legacy"), default="clean")
    parser.add_argument("--test-size", type=int, default=0)
    parser.add_argument("--patience", type=int, default=0)
    parser.add_argument("--seed", type=int, default=23)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--log-every", type=int, default=1)

    # SGA-Seq model settings. Defaults follow the public implementation where possible.
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--embed-dim", type=int, default=14)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--kernel", type=int, default=3)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--graph-neighbors", type=int, default=8)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--cycle-len", type=int, default=24)

    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--save-predictions", action="store_true")
    parser.add_argument("--smoke-test", action="store_true")
    return parser


def load_checkpoint(path: Path, device: torch.device):
    try:
        return torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=device)


def configuration(args: argparse.Namespace) -> Dict[str, object]:
    return {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}


def make_model(args: argparse.Namespace, height: int, width: int) -> SGASeq:
    return SGASeq(
        in_steps=args.close_size,
        out_steps=args.predict_size,
        height=height,
        width=width,
        hidden_dim=args.hidden_dim,
        embed_dim=args.embed_dim,
        layers=args.num_layers,
        kernel=args.kernel,
        num_heads=args.num_heads,
        graph_neighbors=args.graph_neighbors,
        cycle_len=args.cycle_len,
        dropout=args.dropout,
    )


def write_history(path: Path, history: List[Dict[str, float]]) -> None:
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=("epoch", "train_loss", "val_loss", "seconds"))
        writer.writeheader()
        writer.writerows(history)


def smoke_test(args: argparse.Namespace) -> None:
    device = choose_device(args.device, args.gpu)
    set_seed(args.seed)
    model = make_model(args, 3, 4).to(device)
    x = torch.randn(2, args.close_size, 1, 3, 4, device=device)
    phase = torch.tensor([0, 7], dtype=torch.long, device=device)
    target = torch.randn(2, args.predict_size, 1, 3, 4, device=device)
    prediction = model(x, phase)
    loss = torch.mean((prediction - target) ** 2)
    loss.backward()
    if prediction.shape != target.shape or not torch.isfinite(loss):
        raise RuntimeError("SGA-Seq smoke test failed")
    print(f"PASS SGA-Seq shape={tuple(prediction.shape)} params={count_parameters(model):,}")


def run(args: argparse.Namespace) -> None:
    device = choose_device(args.device, args.gpu)
    set_seed(args.seed)
    train_loader, val_loader, test_loader, normalizer, height, width = build_loaders(args, device)
    model = make_model(args, height, width).to(device)
    parameter_count = count_parameters(model)
    output_root = resolve_under_project(args.output_dir)
    run_dir = output_root / args.dataset / "sgaseq" / f"seed_{args.seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = run_dir / "best.pt"
    metrics_path = run_dir / "metrics.json"
    history_path = run_dir / "history.csv"
    config = configuration(args)
    history: List[Dict[str, float]] = []

    print(f"[{args.dataset}/sgaseq/seed_{args.seed}] SGA-Seq 24->12 unified protocol", flush=True)
    print(f"device={device}, trainable_parameters={parameter_count:,}", flush=True)
    criterion: nn.Module = nn.L1Loss() if args.loss == "l1" else nn.MSELoss()
    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    milestones = sorted({max(1, int(args.epochs * 0.25)), max(1, int(args.epochs * 0.75))})
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=0.5)
    best_val = float("inf")
    best_epoch = -1
    wait = 0
    training_started = time.perf_counter()

    if not args.eval_only:
        for epoch in range(1, args.epochs + 1):
            epoch_started = time.perf_counter()
            train_loss = batch_loss(model, train_loader, criterion, device, optimizer)
            val_loss = batch_loss(model, val_loader, criterion, device, None)
            scheduler.step()
            seconds = time.perf_counter() - epoch_started
            history.append(
                {"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss, "seconds": seconds}
            )
            if val_loss < best_val:
                best_val = val_loss
                best_epoch = epoch
                wait = 0
                torch.save(
                    {"model_state": model.state_dict(), "epoch": epoch, "val_loss": val_loss, "args": config},
                    checkpoint_path,
                )
            else:
                wait += 1
            if epoch == 1 or epoch % args.log_every == 0:
                print(
                    f"epoch {epoch:03d} | train={train_loss:.6f} | val={val_loss:.6f} | {seconds:.2f}s",
                    flush=True,
                )
            if args.patience and wait >= args.patience:
                print(f"early stop at epoch {epoch}; best epoch={best_epoch}", flush=True)
                break
        write_history(history_path, history)
    elif not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    checkpoint = load_checkpoint(checkpoint_path, device)
    model.load_state_dict(checkpoint["model_state"])
    best_epoch = int(checkpoint["epoch"])
    best_val = float(checkpoint["val_loss"])
    training_seconds = time.perf_counter() - training_started
    metrics, predictions, truth = evaluate_predictions(model, test_loader, device, normalizer)
    epoch_times = [float(item["seconds"]) for item in history]
    metrics.update(
        {
            "model": "sgaseq",
            "description": "SGA-Seq adapted to 24-input/12-output unified protocol",
            "dataset": args.dataset,
            "seed": args.seed,
            "parameters": parameter_count,
            "best_epoch": best_epoch,
            "best_val_loss": best_val,
            "training_seconds": training_seconds,
            "epoch_seconds_mean": float(np.mean(epoch_times)) if epoch_times else None,
            "epoch_seconds_std": float(np.std(epoch_times, ddof=1)) if len(epoch_times) > 1 else 0.0,
        }
    )
    if args.eval_only and metrics_path.is_file():
        print(f"PRESERVED {metrics_path}", flush=True)
    else:
        metrics_path.write_text(
            json.dumps({"metrics": metrics, "config": config, "history": history}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    if args.save_predictions:
        np.savez_compressed(run_dir / "predictions.npz", pred=predictions, truth=truth)
    print(
        f"RESULT {args.dataset}/sgaseq/seed_{args.seed}: RMSE={metrics['rmse']:.6f}, MAE={metrics['mae']:.6f}",
        flush=True,
    )


def validate(args: argparse.Namespace) -> None:
    if args.close_size <= 0 or args.predict_size <= 0:
        raise ValueError("input/output lengths must be positive")
    if args.epochs <= 0 or args.batch_size <= 0:
        raise ValueError("epochs and batch size must be positive")
    if args.kernel > args.close_size:
        raise ValueError("--kernel cannot exceed --close-size")
    if args.hidden_dim % args.num_heads:
        raise ValueError("--hidden-dim must be divisible by --num-heads")
    if args.hidden_dim % 2:
        raise ValueError("--hidden-dim must be even")
    if args.graph_neighbors <= 0:
        raise ValueError("--graph-neighbors must be positive")
    if args.patience < 0 or args.split_gap < -1:
        raise ValueError("invalid patience or split gap")


def main() -> None:
    args = build_parser().parse_args()
    validate(args)
    if args.smoke_test:
        smoke_test(args)
    else:
        run(args)


if __name__ == "__main__":
    main()

