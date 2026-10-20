"""Run the complete baseline benchmark as a fault-tolerant four-GPU queue."""

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
from typing import Dict, List, Sequence, Tuple


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_PROJECT_ROOT = SCRIPT_DIR.parent
DATASETS = ("milan_sms", "milan_call", "milan_internet", "mobile_bj", "mobile_nj")
MODELS = ("seasonal_ha", "gru_seq2seq", "mtgnn", "patchtst", "dlinear", "timemixer", "GTSRNet")


@dataclass(frozen=True)
class Task:
    model: str
    dataset: str
    seed: int

    @property
    def label(self) -> str:
        return f"{self.dataset}/{self.model}/seed_{self.seed}"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=DEFAULT_PROJECT_ROOT)
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--output-dir", type=Path, default=Path("results_GTSRNet_baselines"))
    parser.add_argument("--models", nargs="+", choices=MODELS, default=MODELS)
    parser.add_argument("--datasets", nargs="+", choices=DATASETS, default=DATASETS)
    parser.add_argument("--seeds", nargs="+", type=int, default=(15, 25, 35))
    parser.add_argument("--gpus", nargs="+", type=int, default=(0, 1, 2, 3))
    parser.add_argument("--close-size", type=int, default=24)
    parser.add_argument("--predict-size", type=int, default=12)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--branch-aux-weight", type=float, default=0.1)
    parser.add_argument("--loss", choices=("l1", "l2"), default="l1")
    parser.add_argument("--patience", type=int, default=0)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--split-mode", choices=("chronological", "random"), default="chronological")
    parser.add_argument("--split-gap", type=int, default=-1)
    parser.add_argument("--milan-loader", choices=("clean", "legacy"), default="clean")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--log-every", type=int, default=1)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--seg-len", type=int, default=6)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--e-layers", type=int, default=2)
    parser.add_argument("--d-ff", type=int, default=256)
    parser.add_argument("--cycle-len", type=int, default=24)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--save-predictions", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--no-summary", action="store_true")
    return parser


def resolve_under(root: Path, path: Path) -> Path:
    path = path.expanduser()
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def validate(args: argparse.Namespace) -> None:
    if len(set(args.gpus)) != len(args.gpus):
        raise ValueError("--gpus contains duplicate GPU indices")
    if len(set(args.seeds)) != len(args.seeds):
        raise ValueError("--seeds contains duplicates")
    if not args.seeds or not args.gpus:
        raise ValueError("--seeds and --gpus cannot be empty")
    if args.close_size <= 0 or args.predict_size <= 0:
        raise ValueError("input and prediction lengths must be positive")
    if args.epochs <= 0 or args.batch_size <= 0:
        raise ValueError("epochs and batch size must be positive")
    if args.learning_rate <= 0 or args.branch_aux_weight < 0:
        raise ValueError("invalid learning rate or branch auxiliary weight")
    if args.patience < 0 or args.split_gap < -1:
        raise ValueError("invalid patience or split gap")


def preflight(args: argparse.Namespace, project_root: Path) -> None:
    required = {
        "milan_sms": project_root / "dataset" / "data_git_version.h5",
        "milan_call": project_root / "dataset" / "data_git_version.h5",
        "milan_internet": project_root / "dataset" / "data_git_version.h5",
        "mobile_bj": project_root / "dataset" / "MobileBJ.npy",
        "mobile_nj": project_root / "dataset" / "MobileNJ.npy",
    }
    missing = sorted({str(required[name]) for name in args.datasets if not required[name].is_file()})
    needed_project_files = [project_root / "utils" / "MTGNN.py", project_root / "utils" / "timemixer_model.py"]
    for path in needed_project_files:
        if not path.is_file() and any(model in args.models for model in ("mtgnn", "timemixer")):
            missing.append(str(path))
    if missing:
        raise FileNotFoundError("Missing required files: " + ", ".join(sorted(set(missing))))
    python = args.python.expanduser().resolve()
    if not python.is_file():
        raise FileNotFoundError(f"Python interpreter not found: {python}")
    if args.dry_run:
        return
    check = (
        "import torch,h5py,sklearn,pandas,numpy,matplotlib,reformer_pytorch,einops; "
        "assert torch.cuda.is_available(), 'CUDA is unavailable'; "
        f"assert torch.cuda.device_count() > {max(args.gpus)}, "
        f"'Need GPU index {max(args.gpus)}, found %d GPU(s)' % torch.cuda.device_count(); "
        "print('PREFLIGHT OK: PyTorch=', torch.__version__, 'GPUs=', torch.cuda.device_count())"
    )
    subprocess.check_call([str(python), "-u", "-c", check], cwd=project_root)


def metrics_path(output_root: Path, task: Task) -> Path:
    return output_root / task.dataset / task.model / f"seed_{task.seed}" / "metrics.json"


def completion_path(output_root: Path, task: Task, args: argparse.Namespace) -> Path:
    if args.eval_only and args.save_predictions:
        return output_root / task.dataset / task.model / f"seed_{task.seed}" / "predictions.npz"
    return metrics_path(output_root, task)


def command_for(
    args: argparse.Namespace,
    task: Task,
    gpu: int,
    project_root: Path,
    output_root: Path,
) -> List[str]:
    command = [
        str(args.python.expanduser().resolve()), "-u", str(SCRIPT_DIR / "train_baseline.py"),
        "--project-root", str(project_root),
        "--output-dir", str(output_root),
        "--model", task.model,
        "--dataset", task.dataset,
        "--seed", str(task.seed),
        "--device", "cuda",
        "--gpu", str(gpu),
        "--close-size", str(args.close_size),
        "--predict-size", str(args.predict_size),
        "--epochs", str(args.epochs),
        "--batch-size", str(args.batch_size),
        "--learning-rate", str(args.learning_rate),
        "--weight-decay", str(args.weight_decay),
        "--branch-aux-weight", str(args.branch_aux_weight),
        "--loss", args.loss,
        "--patience", str(args.patience),
        "--val-ratio", str(args.val_ratio),
        "--split-mode", args.split_mode,
        "--split-gap", str(args.split_gap),
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
    ]
    if args.eval_only:
        command.append("--eval-only")
    if args.save_predictions:
        command.append("--save-predictions")
    return command


def printable(command: Sequence[str]) -> str:
    return " ".join(shlex.quote(str(part)) for part in command)


def worker(
    gpu: int,
    tasks: "queue.Queue[Task]",
    args: argparse.Namespace,
    project_root: Path,
    output_root: Path,
    lock: threading.Lock,
    outcomes: List[Tuple[Task, int]],
) -> None:
    while True:
        try:
            task = tasks.get_nowait()
        except queue.Empty:
            return
        prefix = f"[GPU{gpu} {task.label}]"
        if completion_path(output_root, task, args).is_file() and not args.force:
            with lock:
                print(f"{prefix} SKIP: metrics already exists", flush=True)
            outcomes.append((task, 0))
            tasks.task_done()
            continue
        command = command_for(args, task, gpu, project_root, output_root)
        log_suffix = "_prediction_export" if args.eval_only and args.save_predictions else ""
        log_path = output_root / "logs" / task.dataset / task.model / f"seed_{task.seed}{log_suffix}.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with lock:
            print(f"{prefix} START", flush=True)
            print(f"{prefix} CMD {printable(command)}", flush=True)
            print(f"{prefix} LOG {log_path}", flush=True)
        if args.dry_run:
            outcomes.append((task, 0))
            tasks.task_done()
            continue
        environment: Dict[str, str] = os.environ.copy()
        environment["PYTHONUNBUFFERED"] = "1"
        environment.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
        with log_path.open("w", encoding="utf-8") as stream:
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
                stream.write(line)
                stream.flush()
                with lock:
                    print(f"{prefix} {line}", end="", flush=True)
            code = process.wait()
        with lock:
            print(f"{prefix} {'DONE' if code == 0 else f'FAILED({code})'}", flush=True)
        outcomes.append((task, code))
        tasks.task_done()


def summarize(args: argparse.Namespace, project_root: Path, output_root: Path) -> int:
    command = [
        str(args.python.expanduser().resolve()), "-u", str(SCRIPT_DIR / "summarize_baselines.py"),
        "--results-dir", str(output_root),
        "--models", *args.models,
        "--datasets", *args.datasets,
        "--seeds", *map(str, args.seeds),
    ]
    return subprocess.call(command, cwd=project_root)


def main() -> None:
    args = build_parser().parse_args()
    validate(args)
    project_root = args.project_root.expanduser().resolve()
    output_root = resolve_under(project_root, args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    preflight(args, project_root)

    tasks: "queue.Queue[Task]" = queue.Queue()
    for model in args.models:
        model_seeds = (args.seeds[0],) if model == "seasonal_ha" else args.seeds
        for dataset in args.datasets:
            for seed in model_seeds:
                tasks.put(Task(model, dataset, seed))
    total = tasks.qsize()
    print(f"Scheduling {total} runs on GPUs {list(args.gpus)}; results={output_root}", flush=True)
    lock = threading.Lock()
    outcomes: List[Tuple[Task, int]] = []
    threads = [
        threading.Thread(
            target=worker,
            args=(gpu, tasks, args, project_root, output_root, lock, outcomes),
            name=f"gpu-{gpu}",
        )
        for gpu in args.gpus
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    failures = [(task, code) for task, code in outcomes if code]
    if failures:
        print("Failed tasks:", flush=True)
        for task, code in failures:
            print(f"  {task.label}: exit code {code}", flush=True)
        raise SystemExit(1)
    print(f"All {len(outcomes)} runs completed or were skipped as complete.", flush=True)
    if not args.dry_run and not args.no_summary:
        raise SystemExit(summarize(args, project_root, output_root))


if __name__ == "__main__":
    main()
