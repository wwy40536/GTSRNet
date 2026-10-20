"""Schedule GTSRNet history-window and forecast-horizon sensitivity runs."""

from __future__ import annotations

import argparse
import os
import queue
import shlex
import subprocess
import sys
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_PROJECT_ROOT = SCRIPT_DIR.parent
DATASETS = ("milan_sms", "milan_call", "milan_internet", "mobile_bj", "mobile_nj")
DEFAULT_DATASETS = ("milan_sms", "milan_call", "milan_internet")


@dataclass(frozen=True, order=True)
class Setting:
    history: int
    horizon: int

    @property
    def label(self) -> str:
        return f"hist_{self.history:03d}_pred_{self.horizon:03d}"


@dataclass(frozen=True)
class Task:
    dataset: str
    setting: Setting
    seed: int

    @property
    def label(self) -> str:
        return f"{self.dataset}/{self.setting.label}/seed_{self.seed}"


def positive_int(text: str) -> int:
    value = int(text)
    if value <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return value


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run one-factor-at-a-time sensitivity tests: vary history with the "
            "forecast horizon fixed, then vary horizon with history fixed."
        )
    )
    parser.add_argument("--project-root", type=Path, default=DEFAULT_PROJECT_ROOT)
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--output-dir", type=Path, default=Path("results_GTSRNet_sensitivity"))
    parser.add_argument("--datasets", nargs="+", choices=DATASETS, default=DEFAULT_DATASETS)
    parser.add_argument("--histories", nargs="+", type=positive_int, default=(12, 24, 48, 72))
    parser.add_argument("--horizons", nargs="+", type=positive_int, default=(3, 6, 12, 24))
    parser.add_argument("--base-history", type=positive_int, default=24)
    parser.add_argument("--base-horizon", type=positive_int, default=12)
    parser.add_argument("--seeds", nargs="+", type=int, default=(2,))
    parser.add_argument("--gpus", nargs="+", type=int, default=(0, 1, 2, 3))
    parser.add_argument("--epochs", type=positive_int, default=300)
    parser.add_argument("--batch-size", type=positive_int, default=16)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--loss", choices=("l1", "l2"), default="l1")
    parser.add_argument("--branch-aux-weight", type=float, default=0.1)
    parser.add_argument("--patience", type=int, default=0)
    parser.add_argument("--split-mode", choices=("chronological", "random"), default="chronological")
    parser.add_argument(
        "--split-gap",
        type=int,
        default=-1,
        help="-1 uses one common max_horizon-1 gap for every setting.",
    )
    parser.add_argument("--min-train-samples", type=positive_int, default=100)
    parser.add_argument("--milan-loader", choices=("clean", "legacy"), default="clean")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--log-every", type=positive_int, default=1)
    parser.add_argument("--d-model", type=positive_int, default=128)
    parser.add_argument("--seg-len", type=positive_int, default=6)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--n-heads", type=positive_int, default=4)
    parser.add_argument("--e-layers", type=positive_int, default=2)
    parser.add_argument("--d-ff", type=positive_int, default=256)
    parser.add_argument("--cycle-len", type=positive_int, default=24)
    parser.add_argument("--mase-seasonality", type=positive_int, default=24)
    parser.add_argument("--force", action="store_true", help="Retrain completed settings.")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without training.")
    parser.add_argument("--no-summary", action="store_true")
    return parser


def resolve_under(root: Path, value: Path) -> Path:
    value = value.expanduser()
    return value.resolve() if value.is_absolute() else (root / value).resolve()


def unique_settings(args: argparse.Namespace) -> List[Setting]:
    # A dict removes the duplicated 24->12 baseline shared by the two axes.
    settings = {
        (history, args.base_horizon): Setting(history, args.base_horizon)
        for history in args.histories
    }
    settings.update(
        {
            (args.base_history, horizon): Setting(args.base_history, horizon)
            for horizon in args.horizons
        }
    )
    return sorted(settings.values())


def validate_args(args: argparse.Namespace) -> None:
    if len(set(args.gpus)) != len(args.gpus):
        raise ValueError("--gpus contains duplicate indices")
    if len(set(args.seeds)) != len(args.seeds):
        raise ValueError("--seeds contains duplicates")
    if args.branch_aux_weight < 0:
        raise ValueError("--branch-aux-weight must be non-negative")
    if args.patience < 0:
        raise ValueError("--patience must be non-negative")
    if args.split_gap < -1:
        raise ValueError("--split-gap must be -1 or non-negative")
    if args.learning_rate <= 0:
        raise ValueError("--learning-rate must be positive")
    if not 0 <= args.dropout < 1:
        raise ValueError("--dropout must be in [0,1)")
    invalid_history = [value for value in set(args.histories) | {args.base_history} if value % args.seg_len]
    if invalid_history:
        raise ValueError(
            f"Every history must be divisible by --seg-len={args.seg_len}; invalid: {invalid_history}"
        )


def preflight(args: argparse.Namespace, project_root: Path) -> None:
    required = {
        "milan_sms": project_root / "dataset" / "data_git_version.h5",
        "milan_call": project_root / "dataset" / "data_git_version.h5",
        "milan_internet": project_root / "dataset" / "data_git_version.h5",
        "mobile_bj": project_root / "dataset" / "MobileBJ.npy",
        "mobile_nj": project_root / "dataset" / "MobileNJ.npy",
    }
    missing = sorted({str(required[name]) for name in args.datasets if not required[name].is_file()})
    if args.milan_loader == "legacy" and any(name.startswith("milan_") for name in args.datasets):
        feature = project_root / "dataset" / "crawled_feature.csv"
        if not feature.is_file():
            missing.append(str(feature))
    if missing:
        raise FileNotFoundError("Missing dataset files: " + ", ".join(missing))
    if not args.python.expanduser().resolve().is_file():
        raise FileNotFoundError(f"Python interpreter not found: {args.python}")
    if args.dry_run:
        return
    check = (
        "import torch,h5py,sklearn,pandas,numpy,matplotlib,reformer_pytorch,einops; "
        "assert torch.cuda.is_available(), 'CUDA is unavailable'; "
        f"assert torch.cuda.device_count() > {max(args.gpus)}, "
        f"'Need GPU index {max(args.gpus)}, found %d GPU(s)' % torch.cuda.device_count(); "
        "print('PREFLIGHT OK:', torch.__version__, 'GPUs=', torch.cuda.device_count())"
    )
    subprocess.check_call([str(args.python.resolve()), "-u", "-c", check], cwd=project_root)


def result_root(output_root: Path, task: Task) -> Path:
    return output_root / "configs" / task.setting.label / f"seed_{task.seed}"


def metrics_path(output_root: Path, task: Task) -> Path:
    return result_root(output_root, task) / task.dataset / "GTSRNet" / "metrics.json"


def child_command(
    args: argparse.Namespace,
    task: Task,
    gpu: int,
    project_root: Path,
    output_root: Path,
) -> List[str]:
    alignment_horizon = max([args.base_horizon, *args.horizons])
    origin_end_offset = alignment_horizon - task.setting.horizon
    # A common gap keeps train/validation/test target-origin boundaries equal
    # across all forecast horizons.  -1 means "derive from maximum horizon".
    common_split_gap = alignment_horizon - 1 if args.split_gap < 0 else args.split_gap
    return [
        str(args.python.expanduser().resolve()),
        "-u",
        str(SCRIPT_DIR / "train_sensitivity.py"),
        "--project-root", str(project_root),
        "--output-dir", str(result_root(output_root, task)),
        "--dataset", task.dataset,
        "--variant", "GTSRNet",
        "--close-size", str(task.setting.history),
        "--predict-size", str(task.setting.horizon),
        "--forecast-origin-end-offset", str(origin_end_offset),
        "--min-train-samples", str(args.min_train_samples),
        "--device", "cuda",
        "--gpu", str(gpu),
        "--epochs", str(args.epochs),
        "--batch-size", str(args.batch_size),
        "--learning-rate", str(args.learning_rate),
        "--weight-decay", str(args.weight_decay),
        "--loss", args.loss,
        "--branch-aux-weight", str(args.branch_aux_weight),
        "--patience", str(args.patience),
        "--seed", str(task.seed),
        "--split-mode", args.split_mode,
        "--split-gap", str(common_split_gap),
        "--milan-loader", args.milan_loader,
        "--num-workers", str(args.num_workers),
        "--log-every", str(args.log_every),
        "--d-model", str(args.d_model),
        "--seg-len", str(args.seg_len),
        "--dropout", str(args.dropout),
        "--n-heads", str(args.n_heads),
        "--e-layers", str(args.e_layers),
        "--d-ff", str(args.d_ff),
        "--cycle-len", str(args.cycle_len),
        "--mase-seasonality", str(args.mase_seasonality),
    ]


def printable(command: Sequence[str]) -> str:
    return " ".join(shlex.quote(str(part)) for part in command)


def run_worker(
    gpu: int,
    tasks: "queue.Queue[Task]",
    args: argparse.Namespace,
    project_root: Path,
    output_root: Path,
    print_lock: threading.Lock,
    outcomes: List[Tuple[Task, int]],
) -> None:
    while True:
        try:
            task = tasks.get_nowait()
        except queue.Empty:
            return
        prefix = f"[GPU{gpu} {task.label}]"
        if metrics_path(output_root, task).is_file() and not args.force:
            with print_lock:
                print(f"{prefix} SKIP (metrics.json exists)", flush=True)
            outcomes.append((task, 0))
            tasks.task_done()
            continue

        command = child_command(args, task, gpu, project_root, output_root)
        log_path = output_root / "logs" / task.setting.label / f"seed_{task.seed}" / f"{task.dataset}.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with print_lock:
            print(f"{prefix} START", flush=True)
            print(f"{prefix} CMD {printable(command)}", flush=True)
            print(f"{prefix} LOG {log_path}", flush=True)
        if args.dry_run:
            outcomes.append((task, 0))
            tasks.task_done()
            continue

        environment: Dict[str, str] = os.environ.copy()
        environment["PYTHONUNBUFFERED"] = "1"
        with log_path.open("w", encoding="utf-8") as log_stream:
            process = subprocess.Popen(
                command,
                cwd=project_root,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                bufsize=1,
                env=environment,
            )
            assert process.stdout is not None
            for line in process.stdout:
                log_stream.write(line)
                log_stream.flush()
                with print_lock:
                    print(f"{prefix} {line}", end="", flush=True)
            code = process.wait()
        with print_lock:
            print(f"{prefix} {'DONE' if code == 0 else f'FAILED({code})'}", flush=True)
        outcomes.append((task, code))
        tasks.task_done()


def summarize(args: argparse.Namespace, project_root: Path, output_root: Path) -> int:
    command = [
        str(args.python.expanduser().resolve()),
        "-u",
        str(SCRIPT_DIR / "summarize_sensitivity.py"),
        "--results-dir", str(output_root),
        "--histories", *map(str, args.histories),
        "--horizons", *map(str, args.horizons),
        "--base-history", str(args.base_history),
        "--base-horizon", str(args.base_horizon),
    ]
    return subprocess.call(command, cwd=project_root)


def main() -> None:
    args = build_parser().parse_args()
    validate_args(args)
    project_root = args.project_root.expanduser().resolve()
    output_root = resolve_under(project_root, args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    preflight(args, project_root)

    work: "queue.Queue[Task]" = queue.Queue()
    settings = unique_settings(args)
    for setting in settings:
        for seed in args.seeds:
            for dataset in args.datasets:
                work.put(Task(dataset, setting, seed))
    total = work.qsize()
    print(
        f"Scheduling {total} unique runs ({len(settings)} settings x "
        f"{len(args.seeds)} seeds x {len(args.datasets)} datasets) on GPUs {list(args.gpus)}.",
        flush=True,
    )

    lock = threading.Lock()
    outcomes: List[Tuple[Task, int]] = []
    workers = [
        threading.Thread(
            target=run_worker,
            args=(gpu, work, args, project_root, output_root, lock, outcomes),
            name=f"gpu-{gpu}",
        )
        for gpu in args.gpus
    ]
    for worker in workers:
        worker.start()
    for worker in workers:
        worker.join()

    failures = [(task, code) for task, code in outcomes if code]
    if failures:
        print("Failed runs:", flush=True)
        for task, code in failures:
            print(f"  {task.label}: exit code {code}", flush=True)
        raise SystemExit(1)
    print(f"All {len(outcomes)} runs completed or were already present.", flush=True)
    if not args.dry_run and not args.no_summary:
        raise SystemExit(summarize(args, project_root, output_root))


if __name__ == "__main__":
    main()
