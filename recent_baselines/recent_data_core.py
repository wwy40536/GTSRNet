"""Reuse the established benchmark protocol and add weekly slots for STID."""

from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset


SCRIPT_DIR = Path(__file__).resolve().parent
for candidate in (SCRIPT_DIR, SCRIPT_DIR.parent / "GTSRNet_baseline_benchmark"):
    if (candidate / "training_core.py").is_file():
        sys.path.insert(0, str(candidate))
        break
else:
    raise FileNotFoundError(
        "training_core.py was not found. Put these files inside the existing "
        "GTSRNet_baseline_benchmark folder, or beside that folder."
    )

import training_core as base  # noqa: E402


ArrayWindowDataset = base.ArrayWindowDataset
MinMaxNormalizer = base.MinMaxNormalizer
batch_loss = base.batch_loss
choose_device = base.choose_device
evaluate_predictions = base.evaluate_predictions
set_seed = base.set_seed


def resolve_under(root: Path, path: Path) -> Path:
    path = path.expanduser()
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def _parse_week_slot(value: object) -> int:
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    text = str(value).strip()
    try:
        stamp = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        try:
            stamp = datetime.strptime(text[:13], "%Y-%m-%d %H")
        except ValueError as exc:
            raise ValueError(f"Cannot parse Milan timestamp {text!r}") from exc
    return stamp.weekday() * 24 + stamp.hour


def _milan_week_slots(args, sample_count: int) -> np.ndarray:
    import h5py

    data_path = resolve_under(args.project_root, args.data or Path("dataset/data_git_version.h5"))
    with h5py.File(str(data_path), "r") as source:
        timestamps = np.asarray(source["idx"])
    slots = np.empty(sample_count, dtype=np.int64)
    for start in range(sample_count):
        slots[start] = _parse_week_slot(timestamps[start + args.close_size])
    return slots


def build_loaders(args, device: torch.device):
    if args.milan_loader != "clean":
        raise ValueError("The added baselines require --milan-loader clean for reproducible weekly slots")

    if args.dataset.startswith("milan_"):
        args.traffic = {
            "milan_sms": "sms",
            "milan_call": "call",
            "milan_internet": "internet",
        }[args.dataset]
        x, _hour, y, source_normalizer, height, width, test_size, data_path = (
            base.load_milan_windows_clean(args)
        )
        time_slot = _milan_week_slots(args, len(x))
    else:
        x, _hour, y, source_normalizer, height, width, test_size, data_path = (
            base.load_mobile_windows(args)
        )
        # Mobile arrays contain no timestamps. Preserve the original protocol's
        # alignment: stored observation zero is weekly slot zero.
        time_slot = (np.arange(len(x), dtype=np.int64) + args.close_size) % (24 * 7)

    resolved_gap = (
        args.predict_size - 1 if args.split_gap < 0 else args.split_gap
    ) if args.split_mode == "chronological" else 0
    train_idx, val_idx, test_idx = base.make_split_indices(
        len(x), test_size, args.val_ratio, args.seed, args.split_mode, resolved_gap
    )
    x, y, normalizer = base.normalize_from_training_windows(
        x, y, train_idx, source_normalizer=source_normalizer
    )
    args.resolved_split_gap = resolved_gap
    args.normalization_min = normalizer.minimum
    args.normalization_max = normalizer.maximum
    dataset = ArrayWindowDataset(x, time_slot, y)
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
        f"grid={height}x{width}, train={len(train_idx)}, val={len(val_idx)}, "
        f"test={len(test_idx)}, split={args.split_mode}, gap={resolved_gap}, "
        f"time_slot=weekly, train_scale=[{normalizer.minimum:.6g},{normalizer.maximum:.6g}]",
        flush=True,
    )
    return train_loader, val_loader, test_loader, normalizer, height, width
