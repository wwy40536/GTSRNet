#!/usr/bin/env python3
"""Train GSTRNet on a large Milan grid using shared local windows.

The original spatial branch uses full attention over all H*W cells.  This
script keeps the original 20x20 GSTRNet architecture and shares it across
non-overlapping windows of a larger grid.  Every (time window, spatial tile)
pair is a training example.  Validation and test metrics cover every tile, so
for stride == window_size they equal metrics over the complete large grid.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
import time
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset


SCRIPT_DIR = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=SCRIPT_DIR.parent)
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=Path("GTSRNet_gate_ablation"),
        help="Directory containing gtsrnet_model.py.",
    )
    parser.add_argument(
        "--dataset",
        choices=("milan_sms", "milan_call", "milan_internet"),
        default="milan_internet",
    )
    parser.add_argument("--data", type=Path, default=Path("dataset/data_git_version.h5"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--grid-size", type=int, default=100)
    parser.add_argument("--window-size", type=int, default=20)
    parser.add_argument("--stride", type=int, default=20)
    parser.add_argument("--close-size", type=int, default=24)
    parser.add_argument("--predict-size", type=int, default=12)
    parser.add_argument("--test-size", type=int, default=168)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--split-gap", type=int, default=-1)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--loss", choices=("l1", "l2"), default="l1")
    parser.add_argument("--branch-aux-weight", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=23)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--log-every", type=int, default=1)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--seg-len", type=int, default=6)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--e-layers", type=int, default=2)
    parser.add_argument("--d-ff", type=int, default=256)
    parser.add_argument("--cycle-len", type=int, default=24)
    parser.add_argument("--amp", action="store_true", help="Use CUDA mixed precision.")
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--smoke-test", action="store_true")
    return parser.parse_args()


def resolve(root: Path, value: Path) -> Path:
    value = value.expanduser()
    return value.resolve() if value.is_absolute() else (root / value).resolve()


def prepare_imports(project_root: Path, model_dir: Path):
    resolved_model_dir = resolve(project_root, model_dir)
    model_file = resolved_model_dir / "gtsrnet_model.py"
    if not model_file.is_file():
        raise FileNotFoundError(
            f"Cannot find {model_file}. Upload GTSRNet_gate_ablation first."
        )
    for path in (project_root, resolved_model_dir):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))
    from gtsrnet_model import GTSRNet, count_trainable_parameters

    return GTSRNet, count_trainable_parameters


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def make_split_indices(
    sample_count: int,
    test_size: int,
    val_ratio: float,
    split_gap: int,
) -> tuple[list[int], list[int], list[int]]:
    if not 0.0 < val_ratio < 1.0:
        raise ValueError("--val-ratio must be between 0 and 1")
    if not 0 < test_size < sample_count:
        raise ValueError("--test-size must be positive and smaller than sample_count")
    test_start = sample_count - test_size
    val_pool_end = test_start - split_gap
    val_count = max(1, int(math.floor(val_pool_end * val_ratio)))
    val_start = val_pool_end - val_count
    train_end = val_start - split_gap
    if train_end <= 0:
        raise ValueError("Chronological split leaves no training samples")
    return (
        list(range(0, train_end)),
        list(range(val_start, val_pool_end)),
        list(range(test_start, sample_count)),
    )


def parse_hour(value: object) -> int:
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    text = str(value)
    try:
        return int(text[11:13])
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Cannot parse hour from timestamp {text!r}") from exc


def load_milan(
    path: Path,
    dataset: str,
    grid_size: int,
) -> tuple[np.ndarray, np.ndarray, tuple[int, int]]:
    try:
        import h5py
    except ImportError as exc:
        raise RuntimeError("h5py is required") from exc

    traffic_index = {"milan_sms": 0, "milan_call": 1, "milan_internet": 2}[dataset]
    with h5py.File(str(path), "r") as source:
        traffic = np.asarray(source["data"][:, :, traffic_index], dtype=np.float32)
        timestamps = np.asarray(source["idx"])
    if traffic.ndim != 2:
        raise ValueError(f"Expected [time,cells], got {traffic.shape}")
    side = int(round(math.sqrt(traffic.shape[1])))
    if side * side != traffic.shape[1]:
        raise ValueError(f"Cell count {traffic.shape[1]} is not a square grid")
    if not 0 < grid_size <= side:
        raise ValueError(f"--grid-size must be in [1,{side}]")
    offset = (side - grid_size) // 2
    raw = traffic.reshape(-1, side, side)[:, offset:offset + grid_size, offset:offset + grid_size]
    return np.ascontiguousarray(raw), timestamps, (offset, offset + grid_size)


def tile_positions(grid_size: int, window_size: int, stride: int) -> list[tuple[int, int]]:
    if stride != window_size:
        raise ValueError(
            "This accuracy script requires --stride == --window-size so each cell is "
            "counted exactly once. Use the benchmark script for other layouts."
        )
    if grid_size % window_size != 0:
        raise ValueError("--grid-size must be divisible by --window-size")
    return [
        (row, col)
        for row in range(0, grid_size, stride)
        for col in range(0, grid_size, stride)
    ]


class TiledMilanDataset(Dataset):
    def __init__(
        self,
        raw: np.ndarray,
        phases: np.ndarray,
        starts: Sequence[int],
        positions: Sequence[tuple[int, int]],
        close_size: int,
        predict_size: int,
        window_size: int,
    ) -> None:
        self.raw = raw
        self.phases = phases
        self.starts = np.asarray(starts, dtype=np.int64)
        self.positions = tuple(positions)
        self.close_size = close_size
        self.predict_size = predict_size
        self.window_size = window_size

    def __len__(self) -> int:
        return int(len(self.starts) * len(self.positions))

    def __getitem__(self, index: int):
        time_slot, tile_index = divmod(index, len(self.positions))
        start = int(self.starts[time_slot])
        row, col = self.positions[tile_index]
        target_start = start + self.close_size
        window = np.s_[row:row + self.window_size, col:col + self.window_size]
        x = np.ascontiguousarray(self.raw[start:target_start, window[0], window[1]])[:, None]
        y = np.ascontiguousarray(
            self.raw[target_start:target_start + self.predict_size, window[0], window[1]]
        )[:, None]
        return x, np.int64(self.phases[start]), y


def build_data(args: argparse.Namespace, project_root: Path, device: torch.device):
    data_path = resolve(project_root, args.data)
    if not data_path.is_file():
        raise FileNotFoundError(data_path)
    raw, timestamps, source_rows = load_milan(data_path, args.dataset, args.grid_size)
    sample_count = len(raw) - args.close_size - args.predict_size + 1
    if sample_count <= 1:
        raise ValueError("Not enough time steps")
    gap = args.predict_size - 1 if args.split_gap < 0 else args.split_gap
    train_starts, val_starts, test_starts = make_split_indices(
        sample_count, args.test_size, args.val_ratio, gap
    )
    phases = np.asarray(
        [parse_hour(timestamps[start + args.close_size]) for start in range(sample_count)],
        dtype=np.int64,
    )

    # Chronological training indices are contiguous.  The union below is exactly
    # the raw time interval touched by all training input and target windows.
    fit_start = train_starts[0]
    fit_end = train_starts[-1] + args.close_size + args.predict_size
    train_observations = raw[fit_start:fit_end]
    data_min = float(np.nanmin(train_observations))
    data_max = float(np.nanmax(train_observations))
    if not np.isfinite(data_min) or not np.isfinite(data_max) or data_max <= data_min:
        raise ValueError("Invalid training min/max")
    raw = np.ascontiguousarray((raw - data_min) / (data_max - data_min), dtype=np.float32)

    positions = tile_positions(args.grid_size, args.window_size, args.stride)
    common = dict(
        raw=raw,
        phases=phases,
        positions=positions,
        close_size=args.close_size,
        predict_size=args.predict_size,
        window_size=args.window_size,
    )
    train_set = TiledMilanDataset(starts=train_starts, **common)
    val_set = TiledMilanDataset(starts=val_starts, **common)
    test_set = TiledMilanDataset(starts=test_starts, **common)
    generator = torch.Generator().manual_seed(args.seed)
    loader_common = dict(
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
    )
    loaders = (
        DataLoader(train_set, shuffle=True, generator=generator, **loader_common),
        DataLoader(val_set, shuffle=False, **loader_common),
        DataLoader(test_set, shuffle=False, **loader_common),
    )
    metadata = {
        "data_path": str(data_path),
        "source_rows": list(source_rows),
        "normalization_min": data_min,
        "normalization_max": data_max,
        "normalization_range": data_max - data_min,
        "sample_count": sample_count,
        "train_time_windows": len(train_starts),
        "val_time_windows": len(val_starts),
        "test_time_windows": len(test_starts),
        "tiles_per_grid": len(positions),
        "train_patch_samples": len(train_set),
        "val_patch_samples": len(val_set),
        "test_patch_samples": len(test_set),
        "split_gap": gap,
    }
    return loaders, metadata


def make_model(args: argparse.Namespace, model_class):
    return model_class(
        t_in=args.close_size,
        t_out=args.predict_size,
        height=args.window_size,
        width=args.window_size,
        variant="GTSRNet",
        d_model=args.d_model,
        seg_len=args.seg_len,
        dropout=args.dropout,
        n_heads=args.n_heads,
        e_layers=args.e_layers,
        d_ff=args.d_ff,
        cycle_len=args.cycle_len,
    )


def primary_and_total_loss(
    model: nn.Module,
    x: torch.Tensor,
    phase: torch.Tensor,
    target: torch.Tensor,
    criterion: nn.Module,
    branch_aux_weight: float,
    training: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    if training and branch_aux_weight > 0:
        prediction, branches = model(x, phase, return_branches=True)
        branch_loss = 0.5 * (
            criterion(branches["spatial"], target) + criterion(branches["temporal"], target)
        )
        primary = criterion(prediction, target)
        return primary, primary + branch_aux_weight * branch_loss
    prediction = model(x, phase)
    primary = criterion(prediction, target)
    return primary, primary


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
    scaler,
    amp: bool,
    branch_aux_weight: float,
) -> float:
    training = optimizer is not None
    model.train(training)
    total = 0.0
    count = 0
    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for x, phase, target in loader:
            x = x.to(device=device, dtype=torch.float32, non_blocking=True)
            phase = phase.to(device=device, dtype=torch.long, non_blocking=True)
            target = target.to(device=device, dtype=torch.float32, non_blocking=True)
            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=amp):
                primary, loss = primary_and_total_loss(
                    model, x, phase, target, criterion, branch_aux_weight, training
                )
            if optimizer is not None:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            batch = x.shape[0]
            total += float(primary.detach()) * batch
            count += batch
    return total / max(count, 1)


def evaluate_raw_metrics(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    value_range: float,
    amp: bool,
    full_grid_count: int,
) -> dict[str, object]:
    model.eval()
    abs_sum = 0.0
    square_sum = 0.0
    element_count = 0
    horizon_abs = np.zeros(model.t_out, dtype=np.float64)
    horizon_square = np.zeros(model.t_out, dtype=np.float64)
    horizon_count = np.zeros(model.t_out, dtype=np.int64)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    with torch.no_grad():
        for x, phase, target in loader:
            x = x.to(device=device, dtype=torch.float32, non_blocking=True)
            phase = phase.to(device=device, dtype=torch.long, non_blocking=True)
            target = target.to(device=device, dtype=torch.float32, non_blocking=True)
            with torch.cuda.amp.autocast(enabled=amp):
                prediction = model(x, phase)
            error = (prediction.float() - target) * value_range
            abs_sum += float(error.abs().sum())
            square_sum += float(error.square().sum())
            element_count += error.numel()
            reduced_dims = (0, 2, 3, 4)
            horizon_abs += error.abs().sum(dim=reduced_dims).cpu().numpy()
            horizon_square += error.square().sum(dim=reduced_dims).cpu().numpy()
            per_horizon_count = error[:, 0].numel()
            horizon_count += per_horizon_count
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started
    peak_mb = (
        float(torch.cuda.max_memory_allocated(device)) / (1024 ** 2)
        if device.type == "cuda"
        else 0.0
    )
    return {
        "mae": abs_sum / element_count,
        "rmse": math.sqrt(square_sum / element_count),
        "inference_seconds": elapsed,
        "inference_ms_per_full_grid": 1000.0 * elapsed / max(full_grid_count, 1),
        "evaluation_peak_memory_mb": peak_mb,
        "horizons": [
            {
                "horizon": index + 1,
                "mae": float(horizon_abs[index] / horizon_count[index]),
                "rmse": float(math.sqrt(horizon_square[index] / horizon_count[index])),
            }
            for index in range(model.t_out)
        ],
    }


def smoke_test(args: argparse.Namespace, model_class, device: torch.device) -> None:
    model = make_model(args, model_class).to(device)
    x = torch.randn(2, args.close_size, 1, args.window_size, args.window_size, device=device)
    phase = torch.tensor([0, 7], device=device)
    target = torch.randn(2, args.predict_size, 1, args.window_size, args.window_size, device=device)
    prediction, branches = model(x, phase, return_branches=True)
    loss = (prediction - target).abs().mean()
    loss = loss + 0.05 * (
        (branches["spatial"] - target).abs().mean()
        + (branches["temporal"] - target).abs().mean()
    )
    loss.backward()
    print(f"PASS shape={tuple(prediction.shape)} loss={float(loss):.6f}")


def write_summary_csv(path: Path, metrics: dict[str, object]) -> None:
    fields = (
        "dataset", "seed", "grid_size", "window_size", "tiles_per_grid", "parameters",
        "best_epoch", "trained_epochs", "mae", "rmse", "training_peak_memory_mb",
        "evaluation_peak_memory_mb", "inference_ms_per_full_grid",
    )
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerow(metrics)


def main() -> None:
    args = parse_args()
    project_root = args.project_root.expanduser().resolve()
    model_class, count_parameters = prepare_imports(project_root, args.model_dir)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    if args.gpu < 0 or args.gpu >= torch.cuda.device_count():
        raise ValueError(f"Invalid GPU {args.gpu}; found {torch.cuda.device_count()}")
    if args.close_size % args.seg_len or args.predict_size % args.seg_len:
        raise ValueError("--seg-len must divide close-size and predict-size")
    if args.patience < 0 or args.log_every <= 0:
        raise ValueError("Invalid patience/log-every")
    device = torch.device(f"cuda:{args.gpu}")
    set_seed(args.seed)
    if args.smoke_test:
        smoke_test(args, model_class, device)
        return

    output_root = resolve(project_root, args.output_dir)
    run_dir = output_root / args.dataset / f"seed_{args.seed}"
    metrics_path = run_dir / "metrics.json"
    if metrics_path.is_file() and not args.force and not args.eval_only:
        print(f"SKIP completed run: {metrics_path}")
        return
    run_dir.mkdir(parents=True, exist_ok=True)
    loaders, data_meta = build_data(args, project_root, device)
    train_loader, val_loader, test_loader = loaders
    model = make_model(args, model_class).to(device)
    parameters = int(count_parameters(model))
    criterion: nn.Module = nn.L1Loss() if args.loss == "l1" else nn.MSELoss()
    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    milestones = sorted({max(1, int(args.epochs * 0.25)), max(1, int(args.epochs * 0.75))})
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=0.5)
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp)
    checkpoint_path = run_dir / "best.pt"
    best_val = float("inf")
    best_epoch = -1
    wait = 0
    history: list[dict[str, float]] = []
    config = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    print(
        f"dataset={args.dataset} seed={args.seed} grid={args.grid_size}x{args.grid_size} "
        f"window={args.window_size}x{args.window_size} tiles={data_meta['tiles_per_grid']} "
        f"train_patches={data_meta['train_patch_samples']} params={parameters:,}",
        flush=True,
    )

    torch.cuda.reset_peak_memory_stats(device)
    if not args.eval_only:
        for epoch in range(1, args.epochs + 1):
            started = time.perf_counter()
            train_loss = run_epoch(
                model, train_loader, criterion, device, optimizer, scaler, args.amp,
                args.branch_aux_weight,
            )
            val_loss = run_epoch(
                model, val_loader, criterion, device, None, scaler, args.amp, 0.0
            )
            scheduler.step()
            seconds = time.perf_counter() - started
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
                        "config": config,
                        "data": data_meta,
                    },
                    checkpoint_path,
                )
            else:
                wait += 1
            if epoch == 1 or epoch % args.log_every == 0:
                print(
                    f"epoch {epoch:03d} train={train_loss:.6f} val={val_loss:.6f} "
                    f"seconds={seconds:.2f} wait={wait}/{args.patience}",
                    flush=True,
                )
            if args.patience > 0 and wait >= args.patience:
                print(f"EARLY STOP epoch={epoch}, best={best_epoch}", flush=True)
                break
    elif not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)

    training_peak_mb = float(torch.cuda.max_memory_allocated(device)) / (1024 ** 2)
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    except TypeError:
        checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state"])
    raw_metrics = evaluate_raw_metrics(
        model,
        test_loader,
        device,
        float(data_meta["normalization_range"]),
        args.amp,
        int(data_meta["test_time_windows"]),
    )
    metrics: dict[str, object] = {
        "dataset": args.dataset,
        "seed": args.seed,
        "method": "GSTRNet-windowed-shared",
        "grid_size": args.grid_size,
        "window_size": args.window_size,
        "stride": args.stride,
        "tiles_per_grid": data_meta["tiles_per_grid"],
        "parameters": parameters,
        "best_epoch": int(checkpoint["epoch"]),
        "trained_epochs": len(history),
        "best_val_loss": float(checkpoint["val_loss"]),
        "training_peak_memory_mb": training_peak_mb,
        **raw_metrics,
    }
    payload = {"metrics": metrics, "config": config, "data": data_meta, "history": history}
    metrics_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    write_summary_csv(run_dir / "summary.csv", metrics)
    print(
        f"COMPLETE RMSE={metrics['rmse']:.6f} MAE={metrics['mae']:.6f} "
        f"best_epoch={metrics['best_epoch']} peak={training_peak_mb:.1f}MB",
        flush=True,
    )
    print(f"WROTE {metrics_path}")


if __name__ == "__main__":
    main()
