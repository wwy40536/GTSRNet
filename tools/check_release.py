"""Check release files, optional dependencies and local dataset layouts."""

from __future__ import annotations

import argparse
import csv
import importlib.metadata as metadata
import re
from pathlib import Path


CORE_FILES = (
    "README.md", "requirements.txt", ".gitignore", "LICENSE_NOTICE.md",
    "THIRD_PARTY_NOTICES.md", "data/README.md", "dataset/README.md",
    "tools/export_environment.py",
    "GTSRNet_model/train_ablation.py",
    "GTSRNet_model/gtsrnet_model.py",
    "utils/segrnn_gridB.py", "utils/layers/Embed.py",
    "utils/layers/SelfAttention_Family.py", "utils/layers/Transformer_EncDec.py",
    "utils/masking.py", "utils/dataset.py", "utils/MTGNN.py",
    "utils/layer.py", "utils/timemixer_model.py",
)
EXPERIMENTS = (
    "baselines/run_baselines.py",
    "recent_baselines/run_recent_baselines.py",
    "GTSRNet_gate_ablation/run_ablation.py",
    "GTSRNet_fusion_ablation/run_ablation.py",
    "decoder_comparison/run_decoder_experiments.py",
    "GTSRNet_sensitivity/run_sensitivity.py",
    "GTSRNet_attention/run_attention_analysis.py",
    "GTSRNet_error_analysis/run_mechanism_experiments.py",
    "GTSRNet_scalability/run_windowed_experiments.py",
)


def check_environment(root: Path) -> int:
    failures = 0
    for line in (root / "requirements.txt").read_text(encoding="utf-8-sig").splitlines():
        match = re.match(r"^([A-Za-z0-9_.-]+)==([^\s#]+)", line.strip())
        if not match:
            continue
        name, reference = match.groups()
        try:
            installed = metadata.version(name)
        except metadata.PackageNotFoundError:
            print("FAIL missing package:", name)
            failures += 1
        else:
            # A local CUDA suffix does not change the upstream torch version.
            prefix = "PASS" if installed.split("+")[0] == reference else "INFO"
            print(f"{prefix} {name}: installed={installed}; reference={reference}")
    return failures


def check_data(root: Path, bj_min_hours: int) -> int:
    try:
        import numpy as np
        import h5py
    except ImportError as error:
        print("FAIL data validation needs numpy and h5py:", error.name)
        return 1
    failures = 0
    base = root / "dataset"
    milan = base / "data_git_version.h5"
    if not milan.is_file():
        print("FAIL missing dataset/data_git_version.h5")
        failures += 1
    else:
        try:
            with h5py.File(milan, "r") as handle:
                shape = tuple(handle["data"].shape)
                if len(shape) != 3 or shape[1:] != (10000, 3):
                    raise ValueError("Expected data shape (T, 10000, 3), got " + str(shape))
                if "idx" not in handle or len(handle["idx"]) != shape[0]:
                    raise ValueError("idx must contain one timestamp per data step")
                print("PASS Milan data shape:", shape)
                if shape[0]:
                    print("INFO Milan timestamp endpoints:", repr(handle["idx"][0]), repr(handle["idx"][-1]))
        except (OSError, KeyError, ValueError) as error:
            print("FAIL Milan format:", error)
            failures += 1
    for filename, cells in (("MobileBJ.npy", 672), ("MobileNJ.npy", 560)):
        path = base / filename
        if not path.is_file():
            print("FAIL missing dataset/" + filename)
            failures += 1
            continue
        try:
            values = np.load(path, mmap_mode="r", allow_pickle=False)
            if values.ndim != 3 or tuple(values.shape[1:]) != (cells, 1):
                raise ValueError("Expected (T, " + str(cells) + ", 1), got " + str(values.shape))
            print("PASS", filename, "shape:", tuple(values.shape), "dtype:", values.dtype)
            if filename == "MobileBJ.npy" and values.shape[0] < bj_min_hours:
                print("FAIL MobileBJ has", values.shape[0], "steps; required minimum:", bj_min_hours)
                failures += 1
            if filename == "MobileBJ.npy" and values.shape[0] == 168:
                print("INFO this MobileBJ file has 168 steps (7 days if hourly), not 672 steps.")
            del values
        except (OSError, ValueError) as error:
            print("FAIL", filename, error)
            failures += 1
    auxiliary = base / "crawled_feature.csv"
    if auxiliary.is_file():
        with auxiliary.open(encoding="utf-8-sig", newline="") as stream:
            reader = csv.reader(stream)
            header = next(reader, [])
            count = sum(1 for row in reader if row)
        print("INFO legacy auxiliary CSV:", count, "rows; columns:", header)
    else:
        print("INFO crawled_feature.csv absent; it is needed only by the legacy Milan loader.")
    return failures


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--check-environment", action="store_true")
    parser.add_argument("--check-data", action="store_true")
    parser.add_argument("--bj-min-hours", type=int, default=0)
    args = parser.parse_args()
    if args.bj_min_hours < 0:
        parser.error("--bj-min-hours must be non-negative")
    root = args.project_root.expanduser().resolve()
    failures = 0
    for relative in CORE_FILES:
        if not (root / relative).is_file():
            print("FAIL missing release file:", relative)
            failures += 1
    for relative in EXPERIMENTS:
        print(("FOUND" if (root / relative).is_file() else "OPTIONAL ABSENT"), relative)
    if args.check_environment:
        failures += check_environment(root) if (root / "requirements.txt").is_file() else 1
    if args.check_data:
        failures += check_data(root, args.bj_min_hours)
    print("Structure/data/dependency checks:", "PASS" if not failures else f"FAIL ({failures})")
    print("This check does not execute a model or certify that training results reproduce the paper.")
    raise SystemExit(1 if failures else 0)


if __name__ == "__main__":
    main()
