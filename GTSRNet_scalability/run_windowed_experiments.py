#!/usr/bin/env python3
"""Schedule windowed GSTRNet dataset/seed runs over multiple GPUs."""

from __future__ import annotations

import argparse
import os
import queue
import subprocess
import sys
import threading
from dataclasses import dataclass
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent


@dataclass(frozen=True)
class Task:
    dataset: str
    seed: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=SCRIPT_DIR.parent)
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--gpus", nargs="+", type=int, default=(0, 1, 2, 3))
    parser.add_argument(
        "--datasets", nargs="+",
        choices=("milan_sms", "milan_call", "milan_internet"),
        default=("milan_internet",),
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=(23, 29, 103))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--grid-size", type=int, default=100)
    parser.add_argument("--window-size", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--branch-aux-weight", type=float, default=0.1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--seg-len", type=int, default=6)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--e-layers", type=int, default=2)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def resolve(root: Path, value: Path) -> Path:
    value = value.expanduser()
    return value.resolve() if value.is_absolute() else (root / value).resolve()


def child_command(
    args: argparse.Namespace,
    project_root: Path,
    output_dir: Path,
    task: Task,
    gpu: int,
) -> list[str]:
    command = [
        str(args.python.expanduser().resolve()),
        "-u",
        str(SCRIPT_DIR / "train_windowed.py"),
        "--project-root", str(project_root),
        "--output-dir", str(output_dir),
        "--dataset", task.dataset,
        "--seed", str(task.seed),
        "--gpu", str(gpu),
        "--grid-size", str(args.grid_size),
        "--window-size", str(args.window_size),
        "--stride", str(args.window_size),
        "--batch-size", str(args.batch_size),
        "--epochs", str(args.epochs),
        "--patience", str(args.patience),
        "--learning-rate", str(args.learning_rate),
        "--branch-aux-weight", str(args.branch_aux_weight),
        "--num-workers", str(args.num_workers),
        "--d-model", str(args.d_model),
        "--seg-len", str(args.seg_len),
        "--n-heads", str(args.n_heads),
        "--e-layers", str(args.e_layers),
    ]
    if args.amp:
        command.append("--amp")
    if args.force:
        command.append("--force")
    return command


def worker(
    gpu: int,
    tasks: "queue.Queue[Task]",
    args: argparse.Namespace,
    project_root: Path,
    output_dir: Path,
    lock: threading.Lock,
    results: list[tuple[Task, int]],
) -> None:
    while True:
        try:
            task = tasks.get_nowait()
        except queue.Empty:
            return
        metrics_path = output_dir / task.dataset / f"seed_{task.seed}" / "metrics.json"
        log_path = output_dir / "logs" / task.dataset / f"seed_{task.seed}.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        prefix = f"[GPU{gpu} {task.dataset} seed={task.seed}]"
        if metrics_path.is_file() and not args.force:
            with lock:
                print(f"{prefix} SKIP completed", flush=True)
            results.append((task, 0))
            tasks.task_done()
            continue
        command = child_command(args, project_root, output_dir, task, gpu)
        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"
        env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
        with lock:
            print(f"{prefix} START; log={log_path}", flush=True)
        with log_path.open("w", encoding="utf-8") as log:
            process = subprocess.Popen(
                command,
                cwd=project_root,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                bufsize=1,
            )
            assert process.stdout is not None
            for line in process.stdout:
                log.write(line)
                log.flush()
                with lock:
                    print(f"{prefix} {line}", end="", flush=True)
            code = process.wait()
        with lock:
            print(f"{prefix} {'DONE' if code == 0 else f'FAILED({code})'}", flush=True)
        results.append((task, code))
        tasks.task_done()


def main() -> None:
    args = parse_args()
    project_root = args.project_root.expanduser().resolve()
    output_dir = resolve(project_root, args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if len(set(args.gpus)) != len(args.gpus):
        raise ValueError("Duplicate GPU indices")
    if len(set(args.seeds)) != len(args.seeds):
        raise ValueError("Duplicate seeds")
    tasks: "queue.Queue[Task]" = queue.Queue()
    for dataset in args.datasets:
        for seed in args.seeds:
            tasks.put(Task(dataset, seed))
    total = tasks.qsize()
    lock = threading.Lock()
    results: list[tuple[Task, int]] = []
    workers = [
        threading.Thread(
            target=worker,
            args=(gpu, tasks, args, project_root, output_dir, lock, results),
            daemon=False,
        )
        for gpu in args.gpus
    ]
    print(f"Scheduling {total} tasks on GPUs {args.gpus}", flush=True)
    for thread in workers:
        thread.start()
    for thread in workers:
        thread.join()
    failed = [(task, code) for task, code in results if code != 0]
    print(f"COMPLETE {total - len(failed)}/{total}")
    for task, code in failed:
        print(f"FAILED dataset={task.dataset} seed={task.seed} code={code}")
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
