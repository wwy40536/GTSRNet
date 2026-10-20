"""Fault-tolerant multi-GPU launcher for 5 datasets x 10 SGA-Seq seeds."""

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
DEFAULT_SEEDS = (23, 26, 29, 36, 46, 56, 66, 76, 103, 128)


@dataclass(frozen=True)
class Task:
    dataset: str
    seed: int

    @property
    def label(self) -> str:
        return f"{self.dataset}/sgaseq/seed_{self.seed}"


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--project-root", type=Path, default=DEFAULT_PROJECT_ROOT)
    p.add_argument("--python", type=Path, default=Path(sys.executable))
    p.add_argument("--output-dir", type=Path, default=Path("results_sgaseq"))
    p.add_argument("--datasets", nargs="+", choices=DATASETS, default=DATASETS)
    p.add_argument("--seeds", nargs="+", type=int, default=DEFAULT_SEEDS)
    p.add_argument("--gpus", nargs="+", type=int, default=(0, 1, 2, 3))
    p.add_argument("--close-size", type=int, default=24)
    p.add_argument("--predict-size", type=int, default=12)
    p.add_argument("--epochs", type=int, default=300)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--learning-rate", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--loss", choices=("l1", "l2"), default="l1")
    p.add_argument("--patience", type=int, default=0)
    p.add_argument("--val-ratio", type=float, default=0.1)
    p.add_argument("--split-mode", choices=("chronological", "random"), default="chronological")
    p.add_argument("--split-gap", type=int, default=-1)
    p.add_argument("--milan-loader", choices=("clean", "legacy"), default="clean")
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--log-every", type=int, default=1)
    p.add_argument("--hidden-dim", type=int, default=64)
    p.add_argument("--embed-dim", type=int, default=14)
    p.add_argument("--num-layers", type=int, default=2)
    p.add_argument("--kernel", type=int, default=3)
    p.add_argument("--num-heads", type=int, default=4)
    p.add_argument("--graph-neighbors", type=int, default=8)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--cycle-len", type=int, default=24)
    p.add_argument("--save-predictions", action="store_true")
    p.add_argument("--eval-only", action="store_true")
    p.add_argument("--force", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--no-summary", action="store_true")
    return p


def resolve(root: Path, path: Path) -> Path:
    path = path.expanduser()
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def validate(args: argparse.Namespace) -> None:
    if not args.seeds or not args.gpus or not args.datasets:
        raise ValueError("datasets, seeds and gpus cannot be empty")
    if len(set(args.seeds)) != len(args.seeds):
        raise ValueError("--seeds contains duplicates")
    if len(set(args.gpus)) != len(args.gpus):
        raise ValueError("--gpus contains duplicates")
    if args.epochs <= 0 or args.batch_size <= 0:
        raise ValueError("epochs and batch size must be positive")


def preflight(args: argparse.Namespace, root: Path) -> None:
    paths = {
        "milan_sms": root / "dataset" / "data_git_version.h5",
        "milan_call": root / "dataset" / "data_git_version.h5",
        "milan_internet": root / "dataset" / "data_git_version.h5",
        "mobile_bj": root / "dataset" / "MobileBJ.npy",
        "mobile_nj": root / "dataset" / "MobileNJ.npy",
    }
    missing = sorted({str(paths[name]) for name in args.datasets if not paths[name].is_file()})
    if missing:
        raise FileNotFoundError("Missing dataset files: " + ", ".join(missing))
    python = args.python.expanduser().resolve()
    if not python.is_file():
        raise FileNotFoundError(f"Python interpreter not found: {python}")
    if args.dry_run:
        return
    check = (
        "import torch,h5py,numpy; "
        "assert torch.cuda.is_available(), 'CUDA is unavailable'; "
        f"assert torch.cuda.device_count()>{max(args.gpus)}, "
        f"'Need GPU {max(args.gpus)}, found %d GPU(s)' % torch.cuda.device_count(); "
        "print('PREFLIGHT OK', torch.__version__, torch.cuda.device_count())"
    )
    subprocess.check_call([str(python), "-u", "-c", check], cwd=root)


def completion(output: Path, task: Task, args: argparse.Namespace) -> Path:
    name = "predictions.npz" if args.eval_only and args.save_predictions else "metrics.json"
    return output / task.dataset / "sgaseq" / f"seed_{task.seed}" / name


def command(
    args: argparse.Namespace, task: Task, gpu: int, root: Path, output: Path
) -> List[str]:
    cmd = [
        str(args.python.expanduser().resolve()), "-u", str(SCRIPT_DIR / "train_sgaseq.py"),
        "--project-root", str(root), "--output-dir", str(output),
        "--dataset", task.dataset, "--seed", str(task.seed),
        "--device", "cuda", "--gpu", str(gpu),
        "--close-size", str(args.close_size), "--predict-size", str(args.predict_size),
        "--epochs", str(args.epochs), "--batch-size", str(args.batch_size),
        "--learning-rate", str(args.learning_rate), "--weight-decay", str(args.weight_decay),
        "--loss", args.loss, "--patience", str(args.patience),
        "--val-ratio", str(args.val_ratio), "--split-mode", args.split_mode,
        "--split-gap", str(args.split_gap), "--milan-loader", args.milan_loader,
        "--num-workers", str(args.num_workers), "--log-every", str(args.log_every),
        "--hidden-dim", str(args.hidden_dim), "--embed-dim", str(args.embed_dim),
        "--num-layers", str(args.num_layers), "--kernel", str(args.kernel),
        "--num-heads", str(args.num_heads), "--graph-neighbors", str(args.graph_neighbors),
        "--dropout", str(args.dropout), "--cycle-len", str(args.cycle_len),
    ]
    if args.save_predictions:
        cmd.append("--save-predictions")
    if args.eval_only:
        cmd.append("--eval-only")
    return cmd


def printable(parts: Sequence[str]) -> str:
    return " ".join(shlex.quote(str(part)) for part in parts)


def worker(
    gpu: int,
    pending: "queue.Queue[Task]",
    args: argparse.Namespace,
    root: Path,
    output: Path,
    lock: threading.Lock,
    outcomes: List[Tuple[Task, int]],
) -> None:
    while True:
        try:
            task = pending.get_nowait()
        except queue.Empty:
            return
        prefix = f"[GPU{gpu} {task.label}]"
        if completion(output, task, args).is_file() and not args.force:
            with lock:
                print(f"{prefix} SKIP: result already exists", flush=True)
            outcomes.append((task, 0))
            pending.task_done()
            continue
        cmd = command(args, task, gpu, root, output)
        suffix = "_prediction_export" if args.eval_only and args.save_predictions else ""
        log_path = output / "logs" / task.dataset / "sgaseq" / f"seed_{task.seed}{suffix}.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with lock:
            print(f"{prefix} START\n{prefix} CMD {printable(cmd)}\n{prefix} LOG {log_path}", flush=True)
        if args.dry_run:
            outcomes.append((task, 0))
            pending.task_done()
            continue
        environment: Dict[str, str] = os.environ.copy()
        environment["PYTHONUNBUFFERED"] = "1"
        environment.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
        with log_path.open("w", encoding="utf-8") as stream:
            process = subprocess.Popen(
                cmd, cwd=root, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, encoding="utf-8", errors="replace", bufsize=1, env=environment,
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
        pending.task_done()


def summarize(args: argparse.Namespace, root: Path, output: Path) -> int:
    cmd = [
        str(args.python.expanduser().resolve()), "-u", str(SCRIPT_DIR / "summarize_sgaseq.py"),
        "--results-dir", str(output), "--datasets", *args.datasets,
        "--seeds", *map(str, args.seeds), "--allow-partial",
    ]
    return subprocess.call(cmd, cwd=root)


def main() -> None:
    args = parser().parse_args()
    validate(args)
    root = args.project_root.expanduser().resolve()
    output = resolve(root, args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    preflight(args, root)

    pending: "queue.Queue[Task]" = queue.Queue()
    for dataset in args.datasets:
        for seed in args.seeds:
            pending.put(Task(dataset, seed))
    print(
        f"Scheduling {pending.qsize()} SGA-Seq runs on GPUs {list(args.gpus)}; results={output}",
        flush=True,
    )
    lock = threading.Lock()
    outcomes: List[Tuple[Task, int]] = []
    threads = [
        threading.Thread(
            target=worker, args=(gpu, pending, args, root, output, lock, outcomes), name=f"gpu-{gpu}"
        )
        for gpu in args.gpus
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    if not args.dry_run and not args.no_summary:
        summarize(args, root, output)
    failures = [(task, code) for task, code in outcomes if code]
    if failures:
        print("Failed tasks:", flush=True)
        for task, code in failures:
            print(f"  {task.label}: exit code {code}", flush=True)
        raise SystemExit(1)
    print(f"All {len(outcomes)} runs completed or were skipped.", flush=True)


if __name__ == "__main__":
    main()

