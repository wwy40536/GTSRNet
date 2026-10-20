"""Train or evaluate one baseline under the corrected GTSRNet data protocol."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from torch import nn

# Import the data core first: it resolves --project-root and places the original
# project's ``utils`` package on sys.path before model adapters import it.
from training_core import (
    PROJECT_ROOT,
    batch_loss,
    build_loaders,
    choose_device,
    evaluate_predictions,
    resolve_under_project,
    set_seed,
)
from baseline_models import (
    MODEL_DESCRIPTIONS,
    MODEL_NAMES,
    SeasonalHistoricalAverage,
    build_model,
    count_parameters,
)


DATASETS = ("milan_sms", "milan_call", "milan_internet", "mobile_bj", "mobile_nj")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--model", choices=("all",) + MODEL_NAMES, default="dlinear")
    parser.add_argument("--dataset", choices=DATASETS, default="milan_sms")
    parser.add_argument("--data", type=Path, default=None)
    parser.add_argument("--feature-path", type=Path, default=Path("dataset/crawled_feature.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("results_GTSRNet_baselines"))
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
    parser.add_argument("--branch-aux-weight", type=float, default=0.1)
    parser.add_argument("--loss", choices=("l1", "l2"), default="l1")
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--split-mode", choices=("chronological", "random"), default="chronological")
    parser.add_argument("--split-gap", type=int, default=-1)
    parser.add_argument("--milan-loader", choices=("clean", "legacy"), default="clean")
    parser.add_argument("--test-size", type=int, default=0)
    parser.add_argument("--patience", type=int, default=0)
    parser.add_argument("--seed", type=int, default=15)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--log-every", type=int, default=1)

    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--seg-len", type=int, default=6)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--e-layers", type=int, default=2)
    parser.add_argument("--d-ff", type=int, default=256)
    parser.add_argument("--cycle-len", type=int, default=24)

    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--save-predictions", action="store_true")
    parser.add_argument("--smoke-test", action="store_true")
    return parser


def checkpoint_load(path: Path, device: torch.device):
    try:
        return torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=device)


def config_dict(args: argparse.Namespace) -> Dict[str, object]:
    return {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
    }


def run_smoke_test(args: argparse.Namespace) -> None:
    from torch.utils.data import DataLoader, TensorDataset

    device = choose_device(args.device, args.gpu)
    set_seed(args.seed)
    batch, t_in, t_out, height, width = 2, 24, 12, 2, 3
    x = torch.randn(batch, t_in, 1, height, width)
    phase = torch.tensor([0, 7], dtype=torch.long)
    target = torch.randn(batch, t_out, 1, height, width)
    loader = DataLoader(TensorDataset(x, phase, target), batch_size=batch)
    selected = MODEL_NAMES if args.model == "all" else (args.model,)
    for name in selected:
        model = build_model(
            name, t_in, t_out, height, width, device,
            d_model=16, seg_len=6, dropout=0.0, n_heads=4,
            e_layers=1, d_ff=32, cycle_len=24,
        ).to(device)
        if isinstance(model, SeasonalHistoricalAverage):
            model.fit(loader)
        prediction = model(x.to(device), phase.to(device))
        loss = torch.mean((prediction - target.to(device)) ** 2)
        if count_parameters(model):
            loss.backward()
        if prediction.shape != target.shape or not torch.isfinite(loss):
            raise RuntimeError(f"Smoke test failed for {name}")
        print(f"PASS {name:16s} shape={tuple(prediction.shape)} params={count_parameters(model):,}")


def train(args: argparse.Namespace) -> None:
    device = choose_device(args.device, args.gpu)
    set_seed(args.seed)
    train_loader, val_loader, test_loader, normalizer, height, width = build_loaders(args, device)
    model = build_model(
        args.model,
        args.close_size,
        args.predict_size,
        height,
        width,
        device,
        d_model=args.d_model,
        seg_len=args.seg_len,
        dropout=args.dropout,
        n_heads=args.n_heads,
        e_layers=args.e_layers,
        d_ff=args.d_ff,
        cycle_len=args.cycle_len,
    ).to(device)
    parameters = count_parameters(model)
    output_root = resolve_under_project(args.output_dir)
    run_dir = output_root / args.dataset / args.model / f"seed_{args.seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = run_dir / "best.pt"
    configuration = config_dict(args)
    history: List[Dict[str, float]] = []
    started_training = time.perf_counter()

    print(f"[{args.dataset}/{args.model}/seed_{args.seed}] {MODEL_DESCRIPTIONS[args.model]}")
    print(f"device={device}, trainable_parameters={parameters:,}")

    if isinstance(model, SeasonalHistoricalAverage):
        if args.eval_only and not checkpoint_path.is_file():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        if args.eval_only:
            model.load_state_dict(checkpoint_load(checkpoint_path, device)["model_state"])
            model.fitted = True
        else:
            model.fit(train_loader)
            torch.save({"model_state": model.state_dict(), "args": configuration}, checkpoint_path)
        best_epoch = 0
        best_val = float("nan")
    else:
        criterion: nn.Module = nn.L1Loss() if args.loss == "l1" else nn.MSELoss()
        optimizer = torch.optim.Adam(
            model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
        )
        milestones = sorted({max(1, int(args.epochs * 0.25)), max(1, int(args.epochs * 0.75))})
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=0.5)
        best_val = float("inf")
        best_epoch = -1
        wait = 0
        if not args.eval_only:
            for epoch in range(1, args.epochs + 1):
                epoch_started = time.perf_counter()
                train_loss = batch_loss(
                    model,
                    train_loader,
                    criterion,
                    device,
                    optimizer,
                    branch_aux_weight=(args.branch_aux_weight if args.model == "GTSRNet" else 0.0),
                )
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
                        {
                            "model_state": model.state_dict(),
                            "epoch": epoch,
                            "val_loss": val_loss,
                            "args": configuration,
                        },
                        checkpoint_path,
                    )
                else:
                    wait += 1
                if epoch == 1 or epoch % args.log_every == 0:
                    print(
                        f"epoch {epoch:03d} | train={train_loss:.6f} | "
                        f"val={val_loss:.6f} | {seconds:.2f}s",
                        flush=True,
                    )
                if args.patience and wait >= args.patience:
                    print(f"early stop at epoch {epoch}; best epoch={best_epoch}", flush=True)
                    break
        elif not checkpoint_path.is_file():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        checkpoint = checkpoint_load(checkpoint_path, device)
        model.load_state_dict(checkpoint["model_state"])
        best_epoch = int(checkpoint["epoch"])
        best_val = float(checkpoint["val_loss"])

    train_seconds = time.perf_counter() - started_training
    metrics, predictions, truth = evaluate_predictions(model, test_loader, device, normalizer)
    metrics.update(
        {
            "model": args.model,
            "description": MODEL_DESCRIPTIONS[args.model],
            "dataset": args.dataset,
            "seed": args.seed,
            "parameters": parameters,
            "best_epoch": best_epoch,
            "best_val_loss": best_val,
            "training_seconds": train_seconds,
        }
    )
    metrics_path = run_dir / "metrics.json"
    if args.eval_only and metrics_path.is_file():
        # Keep the original training history/timing when exporting predictions.
        print(f"PRESERVED {metrics_path}", flush=True)
    else:
        metrics_path.write_text(
            json.dumps({"metrics": metrics, "config": configuration, "history": history}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    if args.save_predictions:
        np.savez_compressed(run_dir / "predictions.npz", pred=predictions, truth=truth)
    print(
        f"RESULT {args.dataset}/{args.model}/seed_{args.seed}: "
        f"RMSE={metrics['rmse']:.6f}, MAE={metrics['mae']:.6f}",
        flush=True,
    )


def validate(args: argparse.Namespace) -> None:
    if args.model == "all" and not args.smoke_test:
        raise ValueError("--model all is only valid together with --smoke-test")
    if args.close_size <= 0 or args.predict_size <= 0:
        raise ValueError("--close-size and --predict-size must be positive")
    if args.model == "GTSRNet" and (
        args.close_size % args.seg_len or args.predict_size % args.seg_len
    ):
        raise ValueError("GTSRNet requires --seg-len to divide input and output lengths")
    if args.epochs <= 0 or args.batch_size <= 0:
        raise ValueError("--epochs and --batch-size must be positive")
    if args.patience < 0 or args.branch_aux_weight < 0:
        raise ValueError("--patience and --branch-aux-weight must be non-negative")
    if args.split_gap < -1:
        raise ValueError("--split-gap must be -1 or non-negative")


def main() -> None:
    # ``all`` is accepted only for smoke testing; argparse choices stay strict
    # for real runs, so add it before parsing.
    parser = build_parser()
    for action in parser._actions:
        if action.dest == "model":
            action.choices = ("all",) + MODEL_NAMES
    args = parser.parse_args()
    validate(args)
    if args.smoke_test:
        run_smoke_test(args)
    elif args.model == "all":
        raise ValueError("--model all is only valid with --smoke-test")
    else:
        train(args)


if __name__ == "__main__":
    main()
