"""Train one reviewer-requested embedding condition with the full-v2 model."""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, Subset


SCRIPT_DIR = Path(__file__).resolve().parent


def bootstrap_project_root() -> Path:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--project-root", type=Path, default=SCRIPT_DIR.parent)
    known, _ = parser.parse_known_args()
    root = known.project_root.expanduser().resolve()
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    return root


PROJECT_ROOT = bootstrap_project_root()

from embedding_ablation_model import (  # noqa: E402
    EMBEDDING_DESCRIPTIONS,
    EMBEDDING_VARIANTS,
    Trainduobu1Ablation,
    VARIANT_DESCRIPTIONS,
    count_trainable_parameters,
)


class ArrayWindowDataset(Dataset):
    """Avoid duplicating the already large window arrays in memory."""

    def __init__(self, x: np.ndarray, calendar: np.ndarray, y: np.ndarray) -> None:
        if not (len(x) == len(calendar) == len(y)):
            raise ValueError("x, calendar and y must contain the same number of samples")
        self.x = x
        self.calendar = calendar
        self.y = y

    def __len__(self) -> int:
        return len(self.x)

    def __getitem__(self, index: int):
        return self.x[index], self.calendar[index], self.y[index]


class MinMaxNormalizer:
    """Small NumPy-compatible normalizer fitted on training observations only."""

    def __init__(self) -> None:
        self.minimum = 0.0
        self.maximum = 1.0

    def fit(self, values: np.ndarray) -> None:
        self.set_limits(float(np.nanmin(values)), float(np.nanmax(values)))

    def set_limits(self, minimum: float, maximum: float) -> None:
        self.minimum = float(minimum)
        self.maximum = float(maximum)
        if not np.isfinite(self.minimum) or not np.isfinite(self.maximum):
            raise ValueError("Dataset contains no finite min/max values")
        if self.maximum <= self.minimum:
            raise ValueError("Dataset is constant; min-max normalization is undefined")

    def transform(self, values: np.ndarray) -> np.ndarray:
        return (values - self.minimum) / (self.maximum - self.minimum)

    def inverse_transform(self, values: np.ndarray) -> np.ndarray:
        return values * (self.maximum - self.minimum) + self.minimum


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train one full-v2 embedding-ablation condition."
    )
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--variant", choices=("full_v2",), default="full_v2")
    parser.add_argument(
        "--embedding-variant",
        choices=EMBEDDING_VARIANTS,
        default="joint",
        help="The only model factor varied by this controlled experiment.",
    )
    parser.add_argument(
        "--dataset",
        choices=("milan_sms", "milan_call", "milan_internet", "mobile_bj", "mobile_nj"),
        default="milan_sms",
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=None,
        help="Optional dataset override; defaults are selected from --dataset.",
    )
    parser.add_argument("--feature-path", type=Path, default=Path("dataset/crawled_feature.csv"))
    parser.add_argument(
        "--output-dir", type=Path, default=Path("results_embedding_reviewer")
    )
    parser.add_argument("--traffic", choices=("sms", "call", "internet"), default="sms")

    # Names below are also consumed by the original utils.dataset.read_data.
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

    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument(
        "--branch-aux-weight",
        type=float,
        default=0.1,
        help=(
            "Training-only auxiliary weight for full_v2 branch losses; "
            "0 disables it. Checkpoint selection uses fusion loss only."
        ),
    )
    parser.add_argument(
        "--loss", choices=("l1", "l2"), default="l1",
        help="trainduobu1 used L1 by default; use l2 only as a separate experiment.",
    )
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument(
        "--split-mode",
        choices=("chronological",),
        default="chronological",
        help=(
            "Reviewer experiments require validation/test to remain in the future."
        ),
    )
    parser.add_argument(
        "--split-gap",
        type=int,
        default=-1,
        help=(
            "Purged sample gap between chronological train/validation/test blocks. "
            "-1 uses predict_size-1; ignored for random splitting."
        ),
    )
    parser.add_argument(
        "--milan-loader",
        choices=("clean",),
        default="clean",
        help=(
            "Only the leakage-safe forward-time loader is allowed in this package."
        ),
    )
    parser.add_argument(
        "--test-size",
        type=int,
        default=0,
        help="0 selects dataset defaults: Milan/NJ=168 windows, BJ=24 windows.",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=0,
        help="Early-stop patience; 0 disables early stopping and runs all epochs.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--log-every", type=int, default=1)

    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--seg-len", type=int, default=6)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument(
        "--e-layers",
        type=int,
        default=2,
        help="The archived trainduobu1 spatial branch used two Transformer layers.",
    )
    parser.add_argument("--d-ff", type=int, default=256)
    parser.add_argument("--cycle-len", type=int, default=24)

    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--save-predictions", action="store_true")
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Run shape/backprop checks on synthetic data without loading a dataset.",
    )
    return parser


def resolve_under_project(path: Path) -> Path:
    path = path.expanduser()
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def choose_device(name: str, gpu: int) -> torch.device:
    if name == "cpu":
        return torch.device("cpu")
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda was requested, but CUDA is unavailable")
    if name in ("cuda", "auto") and torch.cuda.is_available():
        if gpu < 0 or gpu >= torch.cuda.device_count():
            raise ValueError(f"GPU index {gpu} is invalid; found {torch.cuda.device_count()} GPU(s)")
        return torch.device(f"cuda:{gpu}")
    return torch.device("cpu")


def make_split_indices(
    sample_count: int,
    test_size: int,
    val_ratio: float,
    seed: int,
    split_mode: str,
    split_gap: int,
) -> Tuple[List[int], List[int], List[int]]:
    if test_size <= 0 or test_size >= sample_count:
        raise ValueError("test_size must be positive and smaller than the sample count")
    if not 0.0 < val_ratio < 1.0:
        raise ValueError("val_ratio must be between 0 and 1")
    test_start = sample_count - test_size
    if split_mode == "chronological":
        if split_gap < 0:
            raise ValueError("split_gap must be non-negative")
        val_pool_end = test_start - split_gap
        val_count = max(1, int(math.floor(val_pool_end * val_ratio)))
        val_start = val_pool_end - val_count
        train_end = val_start - split_gap
        if train_end <= 0:
            raise ValueError(
                "Chronological split leaves no training samples; reduce test_size, "
                "val_ratio or split_gap"
            )
        return (
            list(range(0, train_end)),
            list(range(val_start, val_pool_end)),
            list(range(test_start, sample_count)),
        )

    train_pool_end = test_start
    val_count = max(1, int(math.floor(train_pool_end * val_ratio)))
    if val_count >= train_pool_end:
        raise ValueError("val_ratio leaves no samples for training")
    indices = np.arange(train_pool_end)
    rng = np.random.default_rng(seed)
    rng.shuffle(indices)
    val_indices = indices[:val_count].tolist()
    train_indices = indices[val_count:].tolist()
    test_indices = list(range(train_pool_end, sample_count))
    return train_indices, val_indices, test_indices


def _subset_min_max(values: np.ndarray, indices: Sequence[int]) -> Tuple[float, float]:
    """Find subset limits in bounded chunks instead of copying all windows."""

    minimum = float("inf")
    maximum = float("-inf")
    for start in range(0, len(indices), 128):
        chunk_indices = np.asarray(indices[start:start + 128], dtype=np.int64)
        chunk = values[chunk_indices]
        minimum = min(minimum, float(np.nanmin(chunk)))
        maximum = max(maximum, float(np.nanmax(chunk)))
    return minimum, maximum


def normalize_from_training_windows(
    x: np.ndarray,
    y: np.ndarray,
    train_indices: Sequence[int],
    source_normalizer=None,
) -> Tuple[np.ndarray, np.ndarray, MinMaxNormalizer]:
    """Fit scaling only on training inputs/targets and transform every split."""

    x_min, x_max = _subset_min_max(x, train_indices)
    y_min, y_max = _subset_min_max(y, train_indices)
    source_min = min(x_min, y_min)
    source_max = max(x_max, y_max)
    if source_max <= source_min:
        raise ValueError("Training windows are constant; normalization is undefined")

    if source_normalizer is None:
        raw_minimum, raw_maximum = source_min, source_max
    else:
        old_minimum = getattr(
            source_normalizer, "minimum", getattr(source_normalizer, "min", None)
        )
        old_maximum = getattr(
            source_normalizer, "maximum", getattr(source_normalizer, "max", None)
        )
        if old_minimum is None or old_maximum is None:
            raise TypeError("Legacy normalizer does not expose min/max limits")
        old_range = float(old_maximum) - float(old_minimum)
        raw_minimum = source_min * old_range + float(old_minimum)
        raw_maximum = source_max * old_range + float(old_minimum)

    # The source arrays may already be globally scaled (legacy loader). This
    # affine remapping is equivalent to inverse-transforming and then fitting
    # the new scaler, but avoids two large temporary arrays.
    x = np.ascontiguousarray(x, dtype=np.float32)
    y = np.ascontiguousarray(y, dtype=np.float32)
    x -= source_min
    x /= source_max - source_min
    y -= source_min
    y /= source_max - source_min

    normalizer = MinMaxNormalizer()
    normalizer.set_limits(raw_minimum, raw_maximum)
    return x, y, normalizer


def mobile_dataset_defaults(dataset_name: str) -> Tuple[Path, int, int, int]:
    if dataset_name == "mobile_bj":
        return Path("dataset/MobileBJ.npy"), 28, 24, 24
    if dataset_name == "mobile_nj":
        return Path("dataset/MobileNJ.npy"), 20, 28, 24 * 7
    raise ValueError(f"Not a mobile NumPy dataset: {dataset_name}")


def load_milan_windows_clean(args: argparse.Namespace):
    """Build forward-time Milan windows without the legacy loader's reversal."""

    try:
        import h5py
    except ImportError as exc:
        raise RuntimeError(
            "The clean Milan loader requires h5py. Use the original project environment."
        ) from exc

    data_path = resolve_under_project(args.data or Path("dataset/data_git_version.h5"))
    if not data_path.is_file():
        raise FileNotFoundError(f"Dataset not found: {data_path}")
    traffic_index = {"sms": 0, "call": 1, "internet": 2}[args.traffic]
    with h5py.File(str(data_path), "r") as source:
        if "data" not in source or "idx" not in source:
            raise KeyError("Milan H5 must contain 'data' and 'idx' datasets")
        traffic = np.asarray(source["data"][:, :, traffic_index], dtype=np.float32)
        timestamps = np.asarray(source["idx"])

    expected_cells = args.height * args.width
    if traffic.ndim != 2 or traffic.shape[1] != expected_cells:
        raise ValueError(
            f"Expected Milan traffic [time,{expected_cells}], got {traffic.shape}"
        )
    raw_grid = traffic.reshape(-1, 1, args.height, args.width)
    if args.crop:
        row_start, row_end = args.rows
        col_start, col_end = args.cols
        if not (0 <= row_start < row_end <= args.height):
            raise ValueError(f"Invalid --rows {args.rows} for height {args.height}")
        if not (0 <= col_start < col_end <= args.width):
            raise ValueError(f"Invalid --cols {args.cols} for width {args.width}")
        raw_grid = raw_grid[:, :, row_start:row_end, col_start:col_end]
    actual_height, actual_width = int(raw_grid.shape[-2]), int(raw_grid.shape[-1])

    sample_count = len(raw_grid) - args.close_size - args.predict_size + 1
    if sample_count <= 1:
        raise ValueError(
            f"Not enough Milan steps ({len(raw_grid)}) for input={args.close_size}, "
            f"forecast={args.predict_size}"
        )
    x = np.empty(
        (sample_count, args.close_size, 1, actual_height, actual_width), dtype=np.float32
    )
    y = np.empty(
        (sample_count, args.predict_size, 1, actual_height, actual_width), dtype=np.float32
    )
    # Calendar columns: target hour [0,23], ISO weekday [0,6] (Monday=0).
    calendar = np.empty((sample_count, 2), dtype=np.int64)
    for start in range(sample_count):
        target_start = start + args.close_size
        x[start] = raw_grid[start:target_start]
        y[start] = raw_grid[target_start:target_start + args.predict_size]
        timestamp = timestamps[target_start]
        if isinstance(timestamp, bytes):
            timestamp = timestamp.decode("utf-8")
        timestamp_text = str(timestamp)
        try:
            moment = datetime.fromisoformat(timestamp_text.replace("Z", "+00:00"))
            calendar[start, 0] = moment.hour
            calendar[start, 1] = moment.weekday()
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"Cannot parse Milan timestamp {timestamp_text!r} as an ISO date/time"
            ) from exc

    test_size = args.test_size or 24 * 7
    return x, calendar, y, None, actual_height, actual_width, test_size, data_path


def load_mobile_windows(args: argparse.Namespace):
    default_path, height, width, default_test_size = mobile_dataset_defaults(args.dataset)
    data_path = resolve_under_project(args.data or default_path)
    if not data_path.is_file():
        raise FileNotFoundError(f"Dataset not found: {data_path}")

    raw = np.load(data_path, allow_pickle=False)
    if raw.ndim == 3 and raw.shape[-1] == 1:
        raw = raw[..., 0]
    if raw.ndim != 2:
        raise ValueError(f"Expected a 2-D time-by-grid array, got {raw.shape}")

    grid_cells = height * width
    if raw.shape[1] == grid_cells:
        time_by_grid = raw
    elif raw.shape[0] == grid_cells:
        time_by_grid = raw.T
    else:
        raise ValueError(
            f"{data_path.name} has shape {raw.shape}; neither axis matches "
            f"the paper grid {height}x{width}={grid_cells}."
        )

    if args.dataset == "mobile_bj" and len(time_by_grid) != 28 * 24:
        print(
            "WARNING: MobileBJ.npy contains "
            f"{len(time_by_grid)} hourly steps, while the paper period spans 672 hours. "
            "The run is valid for this file but is not a full paper-data reproduction.",
            flush=True,
        )

    raw_grid = time_by_grid.reshape(-1, 1, height, width).astype(np.float32, copy=False)

    sample_count = len(raw_grid) - args.close_size - args.predict_size + 1
    if sample_count <= 1:
        raise ValueError(
            f"Not enough time steps ({len(raw_grid)}) for input={args.close_size}, "
            f"forecast={args.predict_size}."
        )
    x = np.empty(
        (sample_count, args.close_size, 1, height, width), dtype=np.float32
    )
    y = np.empty(
        (sample_count, args.predict_size, 1, height, width), dtype=np.float32
    )
    calendar = np.empty((sample_count, 2), dtype=np.int64)
    for start in range(sample_count):
        target_start = start + args.close_size
        x[start] = raw_grid[start:target_start]
        y[start] = raw_grid[target_start:target_start + args.predict_size]
        # Mobile files have no timestamps.  These values assume the first stored
        # observation is Monday 00:00; do not mix them with timestamp-backed
        # Milan results without documenting that assumption.
        calendar[start, 0] = target_start % args.cycle_len
        calendar[start, 1] = (target_start // args.cycle_len) % 7

    test_size = args.test_size or default_test_size
    return x, calendar, y, None, height, width, test_size, data_path


def build_loaders(args: argparse.Namespace, device: torch.device):
    if args.dataset.startswith("milan_"):
        traffic_by_dataset = {
            "milan_sms": "sms",
            "milan_call": "call",
            "milan_internet": "internet",
        }
        args.traffic = traffic_by_dataset[args.dataset]
        if args.milan_loader == "clean":
            (
                x,
                calendar,
                y,
                source_normalizer,
                actual_height,
                actual_width,
                test_size,
                data_path,
            ) = load_milan_windows_clean(args)
        else:
            try:
                from utils.dataset import read_data
            except ImportError as exc:
                raise RuntimeError(
                    "Cannot import utils.dataset for --milan-loader legacy."
                ) from exc
            data_path = resolve_under_project(
                args.data or Path("dataset/data_git_version.h5")
            )
            feature_path = resolve_under_project(args.feature_path)
            if not data_path.is_file():
                raise FileNotFoundError(f"Dataset not found: {data_path}")
            if not feature_path.is_file():
                raise FileNotFoundError(f"Feature CSV not found: {feature_path}")
            x, x_meta, _x_cross, y, _labels, source_normalizer = read_data(
                str(data_path), str(feature_path), args
            )
            if x_meta.ndim != 2 or x_meta.shape[1] < 2:
                raise ValueError("X_meta must contain weekday and hour columns")
            calendar = np.stack(
                (
                    x_meta[:, 1].astype(np.int64, copy=False),
                    x_meta[:, 0].astype(np.int64, copy=False),
                ),
                axis=1,
            )
            actual_height, actual_width = int(x.shape[-2]), int(x.shape[-1])
            test_size = args.test_size or 24 * 7
    else:
        (
            x,
            calendar,
            y,
            source_normalizer,
            actual_height,
            actual_width,
            test_size,
            data_path,
        ) = load_mobile_windows(args)

    resolved_gap = (
        args.predict_size - 1 if args.split_gap < 0 else args.split_gap
    ) if args.split_mode == "chronological" else 0
    train_idx, val_idx, test_idx = make_split_indices(
        len(x),
        test_size,
        args.val_ratio,
        args.seed,
        args.split_mode,
        resolved_gap,
    )
    x, y, normalizer = normalize_from_training_windows(
        x, y, train_idx, source_normalizer=source_normalizer
    )
    args.resolved_split_gap = resolved_gap
    args.normalization_min = normalizer.minimum
    args.normalization_max = normalizer.maximum
    dataset = ArrayWindowDataset(x, calendar, y)

    generator = torch.Generator().manual_seed(args.seed)
    common = {
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "pin_memory": device.type == "cuda",
    }
    train_loader = DataLoader(
        Subset(dataset, train_idx), shuffle=True, generator=generator, **common
    )
    val_loader = DataLoader(Subset(dataset, val_idx), shuffle=False, **common)
    test_loader = DataLoader(Subset(dataset, test_idx), shuffle=False, **common)
    print(
        f"dataset={args.dataset}, file={data_path}, samples={len(dataset)}, "
        f"grid={actual_height}x{actual_width}, train={len(train_idx)}, "
        f"val={len(val_idx)}, test={len(test_idx)}, split={args.split_mode}, "
        f"gap={resolved_gap}, milan_loader={args.milan_loader}, "
        f"train_scale=[{normalizer.minimum:.6g},{normalizer.maximum:.6g}]",
        flush=True,
    )
    return train_loader, val_loader, test_loader, normalizer, actual_height, actual_width


def make_model(
    args: argparse.Namespace, variant: str, height: int, width: int
) -> Trainduobu1Ablation:
    return Trainduobu1Ablation(
        t_in=args.close_size,
        t_out=args.predict_size,
        height=height,
        width=width,
        variant=variant,
        d_model=args.d_model,
        seg_len=args.seg_len,
        dropout=args.dropout,
        n_heads=args.n_heads,
        e_layers=args.e_layers,
        d_ff=args.d_ff,
        cycle_len=args.cycle_len,
        embedding_variant=args.embedding_variant,
    )


def batch_loss(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: Optional[torch.optim.Optimizer],
    branch_aux_weight: float = 0.0,
) -> float:
    training = optimizer is not None
    model.train(training)
    total = 0.0
    samples = 0
    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for x, calendar, target in loader:
            x = x.to(device=device, dtype=torch.float32, non_blocking=True)
            calendar = calendar.to(device=device, dtype=torch.long, non_blocking=True)
            target = target.to(device=device, dtype=torch.float32, non_blocking=True)
            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
            use_branch_aux = (
                training
                and branch_aux_weight > 0.0
                and getattr(model, "variant", None) == "full_v2"
            )
            if use_branch_aux:
                prediction, branches = model(x, calendar, return_branches=True)
            else:
                prediction = model(x, calendar)
                branches = None
            if prediction.shape != target.shape:
                raise RuntimeError(
                    f"Prediction/target shape mismatch: {prediction.shape} vs {target.shape}"
                )
            main_loss = criterion(prediction, target)
            loss = main_loss
            if branches is not None:
                spatial = branches["spatial"]
                temporal = branches["temporal"]
                if spatial is None or temporal is None:
                    raise RuntimeError("full_v2 auxiliary loss requires both branches")
                branch_loss = 0.5 * (
                    criterion(spatial, target) + criterion(temporal, target)
                )
                loss = main_loss + branch_aux_weight * branch_loss
            if optimizer is not None:
                loss.backward()
                optimizer.step()
            batch_size = x.shape[0]
            # Training may include an auxiliary regularizer; log the primary
            # fusion objective so train/validation curves remain comparable.
            total += float(main_loss.detach().item()) * batch_size
            samples += batch_size
    return total / max(samples, 1)


def evaluate_predictions(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    normalizer,
) -> Tuple[Dict[str, object], np.ndarray, np.ndarray]:
    model.eval()
    predictions: List[np.ndarray] = []
    truths: List[np.ndarray] = []
    start = time.perf_counter()
    with torch.no_grad():
        for x, calendar, target in loader:
            x = x.to(device=device, dtype=torch.float32, non_blocking=True)
            calendar = calendar.to(device=device, dtype=torch.long, non_blocking=True)
            prediction = model(x, calendar)
            predictions.append(prediction.cpu().numpy())
            truths.append(np.asarray(target))
    elapsed = time.perf_counter() - start

    pred = normalizer.inverse_transform(np.concatenate(predictions, axis=0))
    truth = normalizer.inverse_transform(np.concatenate(truths, axis=0))
    error = pred.astype(np.float64) - truth.astype(np.float64)
    rmse = float(np.sqrt(np.mean(np.square(error))))
    mae = float(np.mean(np.abs(error)))
    horizon_metrics = []
    for horizon in range(pred.shape[1]):
        h_error = error[:, horizon]
        horizon_metrics.append(
            {
                "horizon": horizon + 1,
                "rmse": float(np.sqrt(np.mean(np.square(h_error)))),
                "mae": float(np.mean(np.abs(h_error))),
            }
        )
    return {
        "rmse": rmse,
        "mae": mae,
        "inference_seconds": elapsed,
        "horizons": horizon_metrics,
    }, pred, truth


def train_one(
    args: argparse.Namespace,
    variant: str,
    loaders,
    normalizer,
    height: int,
    width: int,
    device: torch.device,
    output_root: Path,
) -> Dict[str, object]:
    set_seed(args.seed)
    model = make_model(args, variant, height, width).to(device)
    parameters = count_trainable_parameters(model)
    output_root.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_root / "best.pt"

    criterion: nn.Module = nn.L1Loss() if args.loss == "l1" else nn.MSELoss()
    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    milestones = sorted({max(1, int(args.epochs * 0.25)), max(1, int(args.epochs * 0.75))})
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=milestones, gamma=0.5
    )

    train_loader, val_loader, test_loader = loaders
    # ``--variant all`` reuses the loader object. Reset its generator so every
    # variant sees exactly the same epoch-by-epoch shuffle order.
    if getattr(train_loader, "generator", None) is not None:
        train_loader.generator.manual_seed(args.seed)
    best_val = float("inf")
    best_epoch = -1
    wait = 0
    history: List[Dict[str, float]] = []

    print(
        f"\n[{args.dataset}/{args.embedding_variant}/seed_{args.seed}] "
        f"{EMBEDDING_DESCRIPTIONS[args.embedding_variant]}",
        flush=True,
    )
    print(
        f"device={device}, trainable_parameters={parameters:,}, "
        f"embedding_parameters={model.embedding_parameter_count():,}",
        flush=True,
    )
    if variant == "full_v2":
        print(f"branch_aux_weight={args.branch_aux_weight}", flush=True)
    checkpoint_args = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
    }
    if not args.eval_only:
        for epoch in range(1, args.epochs + 1):
            started = time.perf_counter()
            train_loss = batch_loss(
                model,
                train_loader,
                criterion,
                device,
                optimizer,
                branch_aux_weight=(
                    args.branch_aux_weight if variant == "full_v2" else 0.0
                ),
            )
            val_loss = batch_loss(model, val_loader, criterion, device, None)
            scheduler.step()
            epoch_seconds = time.perf_counter() - started
            history.append(
                {
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "seconds": epoch_seconds,
                }
            )
            if val_loss < best_val:
                best_val = val_loss
                best_epoch = epoch
                wait = 0
                torch.save(
                    {
                        "model_state": model.state_dict(),
                        "variant": variant,
                        "embedding_variant": args.embedding_variant,
                        "epoch": epoch,
                        "val_loss": val_loss,
                        "args": checkpoint_args,
                    },
                    checkpoint_path,
                )
            else:
                wait += 1
            if epoch == 1 or epoch % args.log_every == 0:
                print(
                    f"epoch {epoch:03d} | train={train_loss:.6f} | "
                    f"val={val_loss:.6f} | {epoch_seconds:.2f}s",
                    flush=True,
                )
            if args.patience > 0 and wait >= args.patience:
                print(f"early stop at epoch {epoch}; best epoch={best_epoch}", flush=True)
                break
    elif not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found for --eval-only: {checkpoint_path}")

    try:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    except TypeError:  # PyTorch versions before the weights_only argument
        checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state"])
    metrics, pred, truth = evaluate_predictions(model, test_loader, device, normalizer)
    metrics.update(
        {
            "variant": variant,
            "embedding_variant": args.embedding_variant,
            "embedding_description": EMBEDDING_DESCRIPTIONS[args.embedding_variant],
            "dataset": args.dataset,
            "description": VARIANT_DESCRIPTIONS[variant],
            "parameters": parameters,
            "embedding_parameters": model.embedding_parameter_count(),
            "best_epoch": int(checkpoint["epoch"]),
            "best_val_loss": float(checkpoint["val_loss"]),
            "fusion_alpha": model.fusion_alpha(),
        }
    )

    with (output_root / "metrics.json").open("w", encoding="utf-8") as stream:
        json.dump(
            {"metrics": metrics, "config": checkpoint_args, "history": history},
            stream,
            ensure_ascii=False,
            indent=2,
        )
    if args.save_predictions:
        np.savez_compressed(output_root / "predictions.npz", pred=pred, truth=truth)

    print(
        f"[{args.embedding_variant}/seed_{args.seed}] RMSE={metrics['rmse']:.6f}, "
        f"MAE={metrics['mae']:.6f}",
        flush=True,
    )
    return metrics


def write_summary(
    output_root: Path, results: Sequence[Dict[str, object]], filename: str = "summary.csv"
) -> None:
    fields = (
        "dataset",
        "variant",
        "embedding_variant",
        "embedding_description",
        "description",
        "rmse",
        "mae",
        "parameters",
        "embedding_parameters",
        "best_epoch",
        "best_val_loss",
        "fusion_alpha",
        "inference_seconds",
    )
    with (output_root / filename).open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(results)


def smoke_test(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    device = choose_device(args.device, args.gpu)
    # Small valid dimensions keep this check fast while exercising every path.
    t_in, t_out, height, width, seg_len, d_model = 6, 3, 2, 2, 3, 32
    x = torch.randn(2, t_in, 1, height, width, device=device)
    calendar = torch.tensor([[0, 0], [7, 3]], dtype=torch.long, device=device)
    target = torch.randn(2, t_out, 1, height, width, device=device)
    reference_common = None
    embedding_names = ("channel_embedding", "phase_embedding", "joint_embedding")
    for encoding in EMBEDDING_VARIANTS:
        # Re-seeding plus component-local RNG streams makes every shared
        # parameter exactly matched across conditions.
        set_seed(args.seed)
        model = Trainduobu1Ablation(
            t_in=t_in,
            t_out=t_out,
            height=height,
            width=width,
            variant="full_v2",
            d_model=d_model,
            seg_len=seg_len,
            dropout=0.0,
            n_heads=4,
            e_layers=1,
            d_ff=32,
            embedding_variant=encoding,
        )
        common = {
            name: tensor.detach().clone()
            for name, tensor in model.state_dict().items()
            if not any(token in name for token in embedding_names)
        }
        if reference_common is None:
            reference_common = common
        else:
            for name, tensor in common.items():
                if name not in reference_common or not torch.equal(tensor, reference_common[name]):
                    raise RuntimeError(
                        f"Common-parameter initialization mismatch for {encoding}: {name}"
                    )
        model = model.to(device)
        prediction = model(x, calendar)
        loss = torch.mean((prediction - target) ** 2)
        loss.backward()
        if prediction.shape != target.shape or not torch.isfinite(loss):
            raise RuntimeError(f"Smoke test failed for {encoding}")
        print(
            f"PASS {encoding:9s} shape={tuple(prediction.shape)} "
            f"params={count_trainable_parameters(model):,} "
            f"embedding_params={model.embedding_parameter_count():,}"
        )


def main() -> None:
    args = build_parser().parse_args()
    if args.log_every <= 0:
        raise ValueError("--log-every must be positive")
    if args.patience < 0:
        raise ValueError("--patience must be non-negative; use 0 to disable early stopping")
    if args.branch_aux_weight < 0:
        raise ValueError("--branch-aux-weight must be non-negative")
    if args.split_gap < -1:
        raise ValueError("--split-gap must be -1 (auto) or a non-negative integer")
    if args.smoke_test:
        smoke_test(args)
        return

    if args.close_size % args.seg_len != 0 or args.predict_size % args.seg_len != 0:
        raise ValueError("--seg-len must divide both --close-size and --predict-size")
    device = choose_device(args.device, args.gpu)
    set_seed(args.seed)
    train_loader, val_loader, test_loader, normalizer, height, width = build_loaders(
        args, device
    )
    output_root = (
        resolve_under_project(args.output_dir)
        / args.dataset
        / args.embedding_variant
        / f"seed_{args.seed}"
    )
    result = train_one(
        args,
        args.variant,
        (train_loader, val_loader, test_loader),
        normalizer,
        height,
        width,
        device,
        output_root,
    )
    write_summary(output_root, [result], "summary.csv")
    print(f"\nSummary written to: {output_root / 'summary.csv'}", flush=True)


if __name__ == "__main__":
    main()
