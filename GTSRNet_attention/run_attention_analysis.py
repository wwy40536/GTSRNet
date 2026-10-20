"""Run attention analyses for multiple checkpoint roots on a GPU queue."""

from __future__ import annotations

import argparse
import os
import queue
import re
import subprocess
import sys
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple


SCRIPT_DIR = Path(__file__).resolve().parent
DATASETS = ("milan_sms", "milan_call", "milan_internet", "mobile_bj", "mobile_nj")


@dataclass(frozen=True)
class Task:
    checkpoint: Path
    run_label: str
    dataset: str

    @property
    def label(self):
        return f"{self.run_label}/{self.dataset}"


def parse_args():
    parser = argparse.ArgumentParser(description="Schedule checkpoint attention analyses.")
    parser.add_argument("--project-root", type=Path, required=True)
    checkpoints = parser.add_mutually_exclusive_group(required=True)
    checkpoints.add_argument(
        "--baseline-results-root",
        type=Path,
        help="Root using <dataset>/GTSRNet/seed_<seed>/best.pt layout.",
    )
    checkpoints.add_argument(
        "--checkpoint-roots",
        nargs="+",
        type=Path,
        help="Legacy per-seed roots using <dataset>/GTSRNet/best.pt layout.",
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=(16, 26, 36))
    parser.add_argument("--output-dir", type=Path, default=Path("results_GTSRNet_attention"))
    parser.add_argument("--datasets", nargs="+", choices=DATASETS, default=DATASETS)
    parser.add_argument("--gpus", nargs="+", type=int, default=(0, 1, 2, 3))
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--max-test-samples", type=int, default=32)
    parser.add_argument("--fractions", nargs="+", type=float, default=(0.05, 0.10, 0.20))
    parser.add_argument("--random-repeats", type=int, default=5)
    parser.add_argument("--analysis-seed", type=int, default=2026)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--no-summary", action="store_true")
    return parser.parse_args()


def resolve_under(root: Path, value: Path):
    value = value.expanduser()
    return value.resolve() if value.is_absolute() else (root / value).resolve()


def clean_label(index: int, path: Path):
    name = re.sub(r"[^A-Za-z0-9_.-]+", "_", path.name).strip("_") or "checkpoint"
    return f"run_{index:02d}_{name}"


def child_command(args, task: Task, gpu: int, project_root: Path, output_root: Path):
    return [
        str(args.python.expanduser().resolve()), "-u", str(SCRIPT_DIR / "analyze_attention.py"),
        "--project-root", str(project_root), "--checkpoint", str(task.checkpoint),
        "--dataset", task.dataset, "--output-dir", str(output_root / "runs" / task.run_label / task.dataset),
        "--device", "cuda", "--gpu", str(gpu), "--batch-size", str(args.batch_size),
        "--max-test-samples", str(args.max_test_samples), "--fractions", *map(str, args.fractions),
        "--random-repeats", str(args.random_repeats), "--analysis-seed", str(args.analysis_seed),
        "--num-workers", str(args.num_workers), "--dpi", str(args.dpi),
    ]


def build_tasks(args, project_root: Path) -> List[Task]:
    tasks: List[Task] = []
    if args.baseline_results_root is not None:
        root = resolve_under(project_root, args.baseline_results_root)
        if len(set(args.seeds)) != len(args.seeds):
            raise ValueError("--seeds contains duplicates")
        for seed in args.seeds:
            for dataset in args.datasets:
                checkpoint = root / dataset / "GTSRNet" / f"seed_{seed}" / "best.pt"
                tasks.append(Task(checkpoint, f"seed_{seed}", dataset))
    else:
        roots = [resolve_under(project_root, value) for value in args.checkpoint_roots]
        for index, root in enumerate(roots, start=1):
            label = clean_label(index, root)
            for dataset in args.datasets:
                checkpoint = root / dataset / "GTSRNet" / "best.pt"
                tasks.append(Task(checkpoint, label, dataset))
    return tasks


def preflight(args, project_root: Path, task_list: List[Task]):
    if len(set(args.gpus)) != len(args.gpus):
        raise ValueError("--gpus contains duplicates")
    if args.batch_size <= 0 or args.random_repeats <= 0 or args.max_test_samples < 0:
        raise ValueError("Invalid batch/repeat/sample count")
    missing = [str(task.checkpoint) for task in task_list if not task.checkpoint.is_file()]
    if missing:
        raise FileNotFoundError("Missing checkpoints:\n  " + "\n  ".join(missing))
    if not (project_root / "GTSRNet_model" / "train_ablation.py").is_file():
        raise FileNotFoundError("Original GTSRNet_model folder is missing")
    if args.dry_run:
        return
    check = (
        "import torch,numpy,h5py,matplotlib,sklearn,pandas,reformer_pytorch,einops; "
        "assert torch.cuda.is_available(), 'CUDA unavailable'; "
        f"assert torch.cuda.device_count()>{max(args.gpus)}, 'insufficient GPU count'; "
        "print('PREFLIGHT OK', torch.__version__, torch.cuda.device_count())"
    )
    subprocess.check_call([str(args.python.expanduser().resolve()), "-u", "-c", check], cwd=project_root)


def worker(gpu, tasks, args, project_root, output_root, lock, outcomes):
    while True:
        try:
            task = tasks.get_nowait()
        except queue.Empty:
            return
        task_output = output_root / "runs" / task.run_label / task.dataset
        manifest = task_output / "manifest.json"
        prefix = f"[GPU{gpu} {task.label}]"
        if manifest.is_file() and not args.force:
            with lock:
                print(f"{prefix} SKIP (manifest exists)", flush=True)
            outcomes.append((task, 0))
            tasks.task_done()
            continue
        command = child_command(args, task, gpu, project_root, output_root)
        log_path = output_root / "logs" / task.run_label / f"{task.dataset}.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with lock:
            print(f"{prefix} START\n{prefix} LOG {log_path}", flush=True)
            if args.dry_run:
                print(f"{prefix} CMD {' '.join(command)}", flush=True)
        if args.dry_run:
            outcomes.append((task, 0))
            tasks.task_done()
            continue
        environment: Dict[str, str] = os.environ.copy()
        environment["PYTHONUNBUFFERED"] = "1"
        with log_path.open("w", encoding="utf-8") as log:
            process = subprocess.Popen(
                command, cwd=project_root, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, encoding="utf-8", errors="replace", bufsize=1, env=environment,
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
        outcomes.append((task, code))
        tasks.task_done()


def main():
    args = parse_args()
    project_root = args.project_root.expanduser().resolve()
    output_root = resolve_under(project_root, args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    task_list = build_tasks(args, project_root)
    preflight(args, project_root, task_list)
    tasks = queue.Queue()
    for task in task_list:
        tasks.put(task)
    total = tasks.qsize()
    print(f"Scheduling {total} analyses on GPUs {list(args.gpus)}", flush=True)
    lock = threading.Lock()
    outcomes: List[Tuple[Task, int]] = []
    threads = [threading.Thread(target=worker, args=(gpu, tasks, args, project_root, output_root, lock, outcomes)) for gpu in args.gpus]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    failures = [(task, code) for task, code in outcomes if code]
    if failures:
        for task, code in failures:
            print(f"FAILED {task.label}: {code}", flush=True)
        raise SystemExit(1)
    if not args.dry_run and not args.no_summary:
        command = [
            str(args.python.expanduser().resolve()), "-u", str(SCRIPT_DIR / "summarize_attention.py"),
            "--results-dir", str(output_root), "--dpi", str(args.dpi),
        ]
        raise SystemExit(subprocess.call(command, cwd=project_root))


if __name__ == "__main__":
    main()
