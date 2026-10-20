#!/usr/bin/env python3
"""Benchmark original-global and shared-window GSTRNet scaling on synthetic grids."""

from __future__ import annotations

import argparse
import csv
import gc
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn


SCRIPT_DIR = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=SCRIPT_DIR.parent)
    parser.add_argument("--model-dir", type=Path, default=Path("GTSRNet_gate_ablation"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--grid-sizes", nargs="+", type=int, default=(20, 40, 60, 80, 100))
    parser.add_argument("--window-size", type=int, default=20)
    parser.add_argument("--tile-batch-size", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--close-size", type=int, default=24)
    parser.add_argument("--predict-size", type=int, default=12)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--seg-len", type=int, default=6)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--e-layers", type=int, default=2)
    parser.add_argument("--d-ff", type=int, default=256)
    parser.add_argument("--cycle-len", type=int, default=24)
    parser.add_argument("--branch-aux-weight", type=float, default=0.1)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument(
        "--global-max-size",
        type=int,
        default=40,
        help="Benchmark quadratic global attention only up to this size; 0 disables it.",
    )
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--amp", action="store_true")
    return parser.parse_args()


def resolve(root: Path, value: Path) -> Path:
    value = value.expanduser()
    return value.resolve() if value.is_absolute() else (root / value).resolve()


def prepare_imports(project_root: Path, model_dir: Path):
    model_dir = resolve(project_root, model_dir)
    if not (model_dir / "gtsrnet_model.py").is_file():
        raise FileNotFoundError(model_dir / "gtsrnet_model.py")
    for path in (project_root, model_dir):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))
    from gtsrnet_model import GTSRNet, count_trainable_parameters

    return GTSRNet, count_trainable_parameters


def make_model(args: argparse.Namespace, model_class, side: int):
    return model_class(
        t_in=args.close_size,
        t_out=args.predict_size,
        height=side,
        width=side,
        variant="GTSRNet",
        d_model=args.d_model,
        seg_len=args.seg_len,
        dropout=args.dropout,
        n_heads=args.n_heads,
        e_layers=args.e_layers,
        d_ff=args.d_ff,
        cycle_len=args.cycle_len,
    )


def extract_tiles(values: torch.Tensor, window_size: int) -> torch.Tensor:
    batch, steps, channels, height, width = values.shape
    if height % window_size or width % window_size:
        raise ValueError("Grid dimensions must be divisible by window-size")
    tiles = values.unfold(3, window_size, window_size).unfold(4, window_size, window_size)
    # [B,T,C,rows,cols,ws,ws] -> [B*rows*cols,T,C,ws,ws]
    return (
        tiles.permute(0, 3, 4, 1, 2, 5, 6)
        .contiguous()
        .view(batch * (height // window_size) * (width // window_size), steps, channels, window_size, window_size)
    )


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def tiled_train_step(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    x_tiles: torch.Tensor,
    y_tiles: torch.Tensor,
    phase_tiles: torch.Tensor,
    chunk: int,
    amp: bool,
    branch_aux_weight: float,
) -> None:
    model.train()
    optimizer.zero_grad(set_to_none=True)
    scaler = torch.cuda.amp.GradScaler(enabled=amp)
    total_tiles = x_tiles.shape[0]
    for start in range(0, total_tiles, chunk):
        stop = min(start + chunk, total_tiles)
        x = x_tiles[start:stop]
        target = y_tiles[start:stop]
        phase = phase_tiles[start:stop]
        weight = (stop - start) / total_tiles
        with torch.cuda.amp.autocast(enabled=amp):
            prediction, branches = model(x, phase, return_branches=True)
            main = (prediction - target).abs().mean()
            branch = 0.5 * (
                (branches["spatial"] - target).abs().mean()
                + (branches["temporal"] - target).abs().mean()
            )
            loss = weight * (main + branch_aux_weight * branch)
        scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()


def tiled_inference(
    model: nn.Module,
    x_tiles: torch.Tensor,
    phase_tiles: torch.Tensor,
    chunk: int,
    amp: bool,
) -> None:
    model.eval()
    with torch.no_grad():
        for start in range(0, x_tiles.shape[0], chunk):
            stop = min(start + chunk, x_tiles.shape[0])
            with torch.cuda.amp.autocast(enabled=amp):
                model(x_tiles[start:stop], phase_tiles[start:stop])


def benchmark_tiled(
    args: argparse.Namespace,
    model_class,
    count_parameters,
    device: torch.device,
    grid_size: int,
) -> dict[str, object]:
    model = make_model(args, model_class, args.window_size).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    x = torch.randn(
        args.batch_size, args.close_size, 1, grid_size, grid_size, device=device
    )
    y = torch.randn(
        args.batch_size, args.predict_size, 1, grid_size, grid_size, device=device
    )
    phase = torch.zeros(args.batch_size, dtype=torch.long, device=device)
    x_tiles = extract_tiles(x, args.window_size)
    y_tiles = extract_tiles(y, args.window_size)
    repeats_per_sample = (grid_size // args.window_size) ** 2
    phase_tiles = phase.repeat_interleave(repeats_per_sample)
    chunk = max(1, args.tile_batch_size)

    for _ in range(args.warmup):
        tiled_train_step(
            model, optimizer, x_tiles, y_tiles, phase_tiles, chunk, args.amp,
            args.branch_aux_weight,
        )
        tiled_inference(model, x_tiles, phase_tiles, chunk, args.amp)
    synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)
    train_times = []
    inference_times = []
    for _ in range(args.repeats):
        synchronize(device)
        started = time.perf_counter()
        tiled_train_step(
            model, optimizer, x_tiles, y_tiles, phase_tiles, chunk, args.amp,
            args.branch_aux_weight,
        )
        synchronize(device)
        train_times.append(1000.0 * (time.perf_counter() - started))

        synchronize(device)
        started = time.perf_counter()
        tiled_inference(model, x_tiles, phase_tiles, chunk, args.amp)
        synchronize(device)
        inference_times.append(1000.0 * (time.perf_counter() - started))
    return {
        "mode": "windowed_shared",
        "grid_size": grid_size,
        "cells": grid_size * grid_size,
        "window_size": args.window_size,
        "tiles_per_grid": repeats_per_sample,
        "batch_size": args.batch_size,
        "tile_batch_size": chunk,
        "parameters": int(count_parameters(model)),
        "train_step_ms_mean": float(np.mean(train_times)),
        "train_step_ms_std": float(np.std(train_times, ddof=1)) if len(train_times) > 1 else 0.0,
        "inference_ms_mean": float(np.mean(inference_times)),
        "inference_ms_std": float(np.std(inference_times, ddof=1)) if len(inference_times) > 1 else 0.0,
        "peak_memory_mb": float(torch.cuda.max_memory_allocated(device)) / (1024 ** 2),
        "status": "ok",
        "error": "",
    }


def global_train_step(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    x: torch.Tensor,
    y: torch.Tensor,
    phase: torch.Tensor,
    amp: bool,
    branch_aux_weight: float,
) -> None:
    model.train()
    optimizer.zero_grad(set_to_none=True)
    scaler = torch.cuda.amp.GradScaler(enabled=amp)
    with torch.cuda.amp.autocast(enabled=amp):
        prediction, branches = model(x, phase, return_branches=True)
        main = (prediction - y).abs().mean()
        branch = 0.5 * (
            (branches["spatial"] - y).abs().mean()
            + (branches["temporal"] - y).abs().mean()
        )
        loss = main + branch_aux_weight * branch
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()


def global_inference(model, x, phase, amp: bool) -> None:
    model.eval()
    with torch.no_grad(), torch.cuda.amp.autocast(enabled=amp):
        model(x, phase)


def benchmark_global(
    args: argparse.Namespace,
    model_class,
    count_parameters,
    device: torch.device,
    grid_size: int,
) -> dict[str, object]:
    base = {
        "mode": "original_global",
        "grid_size": grid_size,
        "cells": grid_size * grid_size,
        "window_size": grid_size,
        "tiles_per_grid": 1,
        "batch_size": args.batch_size,
        "tile_batch_size": args.batch_size,
    }
    if args.global_max_size <= 0 or grid_size > args.global_max_size:
        return {**base, "parameters": "", "train_step_ms_mean": "", "train_step_ms_std": "",
                "inference_ms_mean": "", "inference_ms_std": "", "peak_memory_mb": "",
                "status": "skipped", "error": f"above global-max-size={args.global_max_size}"}
    try:
        model = make_model(args, model_class, grid_size).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        x = torch.randn(
            args.batch_size, args.close_size, 1, grid_size, grid_size, device=device
        )
        y = torch.randn(
            args.batch_size, args.predict_size, 1, grid_size, grid_size, device=device
        )
        phase = torch.zeros(args.batch_size, dtype=torch.long, device=device)
        for _ in range(args.warmup):
            global_train_step(model, optimizer, x, y, phase, args.amp, args.branch_aux_weight)
            global_inference(model, x, phase, args.amp)
        synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
        train_times = []
        inference_times = []
        for _ in range(args.repeats):
            synchronize(device)
            started = time.perf_counter()
            global_train_step(model, optimizer, x, y, phase, args.amp, args.branch_aux_weight)
            synchronize(device)
            train_times.append(1000.0 * (time.perf_counter() - started))
            synchronize(device)
            started = time.perf_counter()
            global_inference(model, x, phase, args.amp)
            synchronize(device)
            inference_times.append(1000.0 * (time.perf_counter() - started))
        return {
            **base,
            "parameters": int(count_parameters(model)),
            "train_step_ms_mean": float(np.mean(train_times)),
            "train_step_ms_std": float(np.std(train_times, ddof=1)) if len(train_times) > 1 else 0.0,
            "inference_ms_mean": float(np.mean(inference_times)),
            "inference_ms_std": float(np.std(inference_times, ddof=1)) if len(inference_times) > 1 else 0.0,
            "peak_memory_mb": float(torch.cuda.max_memory_allocated(device)) / (1024 ** 2),
            "status": "ok",
            "error": "",
        }
    except torch.cuda.OutOfMemoryError as exc:
        return {
            **base,
            "parameters": "",
            "train_step_ms_mean": "",
            "train_step_ms_std": "",
            "inference_ms_mean": "",
            "inference_ms_std": "",
            "peak_memory_mb": "",
            "status": "oom",
            "error": str(exc).splitlines()[0],
        }
    finally:
        # Local tensor references are released when this function returns.
        pass


def main() -> None:
    args = parse_args()
    project_root = args.project_root.expanduser().resolve()
    output_dir = resolve(project_root, args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    device = torch.device(f"cuda:{args.gpu}")
    model_class, count_parameters = prepare_imports(project_root, args.model_dir)
    rows: list[dict[str, object]] = []
    for grid_size in args.grid_sizes:
        if grid_size % args.window_size:
            raise ValueError(f"grid size {grid_size} is not divisible by window-size")
        print(f"BENCH windowed grid={grid_size}x{grid_size}", flush=True)
        windowed = benchmark_tiled(args, model_class, count_parameters, device, grid_size)
        rows.append(windowed)
        print(json.dumps(windowed, ensure_ascii=False), flush=True)
        gc.collect()
        torch.cuda.empty_cache()
        print(f"BENCH global grid={grid_size}x{grid_size}", flush=True)
        global_row = benchmark_global(args, model_class, count_parameters, device, grid_size)
        rows.append(global_row)
        print(json.dumps(global_row, ensure_ascii=False), flush=True)
        gc.collect()
        torch.cuda.empty_cache()

    csv_path = output_dir / "scalability_benchmark.csv"
    fields = (
        "mode", "grid_size", "cells", "window_size", "tiles_per_grid", "batch_size",
        "tile_batch_size", "parameters", "train_step_ms_mean", "train_step_ms_std",
        "inference_ms_mean", "inference_ms_std", "peak_memory_mb", "status", "error",
    )
    with csv_path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    (output_dir / "scalability_benchmark.json").write_text(
        json.dumps({"config": vars(args), "rows": rows}, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    print(f"WROTE {csv_path}")


if __name__ == "__main__":
    main()
