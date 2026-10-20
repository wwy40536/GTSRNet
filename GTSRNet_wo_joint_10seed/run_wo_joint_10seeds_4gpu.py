"""Run only the w/o Joint embedding condition on ten seeds and five datasets."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_DATASETS = (
    "milan_sms",
    "milan_call",
    "milan_internet",
    "mobile_bj",
    "mobile_nj",
)
DEFAULT_SEEDS = (1, 2, 3, 4, 5, 6, 7, 8, 9, 10)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=SCRIPT_DIR.parent)
    parser.add_argument("--output-dir", type=Path, default=Path("results_wo_joint_10seeds"))
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--datasets", nargs="+", default=DEFAULT_DATASETS)
    parser.add_argument("--seeds", nargs="+", type=int, default=DEFAULT_SEEDS)
    parser.add_argument("--gpus", nargs="+", type=int, default=(0, 1, 2, 3))
    parser.add_argument("--input-len", type=int, default=24)
    parser.add_argument("--horizon", type=int, default=12)
    parser.add_argument("--seg-len", type=int, default=6)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--loss", choices=("l1", "l2"), default="l1")
    parser.add_argument("--branch-aux-weight", type=float, default=0.1)
    parser.add_argument("--patience", type=int, default=0)
    parser.add_argument("--split-gap", type=int, default=-1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--skip-summary", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if len(args.seeds) != 10:
        raise ValueError(f"Exactly 10 seeds are required; received {len(args.seeds)}")
    if len(set(args.seeds)) != 10:
        raise ValueError("The 10 seeds must be unique")

    root = args.project_root.expanduser().resolve()
    output = args.output_dir.expanduser()
    if not output.is_absolute():
        output = (root / output).resolve()

    command = [
        str(args.python.expanduser().resolve()),
        "-u",
        str(SCRIPT_DIR / "run_embedding_ablation_4gpu.py"),
        "--project-root",
        str(root),
        "--output-dir",
        str(output),
        "--datasets",
        *args.datasets,
        "--embeddings",
        "channel_phase",
        "--seeds",
        *[str(seed) for seed in args.seeds],
        "--gpus",
        *[str(gpu) for gpu in args.gpus],
        "--input-len",
        str(args.input_len),
        "--horizon",
        str(args.horizon),
        "--seg-len",
        str(args.seg_len),
        "--d-model",
        str(args.d_model),
        "--epochs",
        str(args.epochs),
        "--batch-size",
        str(args.batch_size),
        "--learning-rate",
        str(args.learning_rate),
        "--loss",
        args.loss,
        "--branch-aux-weight",
        str(args.branch_aux_weight),
        "--patience",
        str(args.patience),
        "--split-gap",
        str(args.split_gap),
        "--num-workers",
        str(args.num_workers),
        "--no-plots",
    ]
    if args.force:
        command.append("--force")
    if args.dry_run:
        command.append("--dry-run")

    environment = os.environ.copy()
    environment.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    subprocess.check_call(command, cwd=root, env=environment)

    if not args.dry_run and not args.skip_summary:
        summary_command = [
            str(args.python.expanduser().resolve()),
            "-u",
            str(SCRIPT_DIR / "summarize_wo_joint_each_seed.py"),
            "--results-dir",
            str(output),
            "--datasets",
            *args.datasets,
            "--seeds",
            *[str(seed) for seed in args.seeds],
        ]
        subprocess.check_call(summary_command, cwd=root, env=environment)


if __name__ == "__main__":
    main()
