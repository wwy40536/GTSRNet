"""Schedule five full-v2 embedding conditions over datasets and random seeds."""

from __future__ import annotations

import argparse
import os
import queue
import subprocess
import sys
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple


SCRIPT_DIR = Path(__file__).resolve().parent
DATASETS = ("milan_sms", "milan_call", "milan_internet", "mobile_bj", "mobile_nj")
VARIANTS = ("none", "channel", "phase", "channel_phase", "joint")


@dataclass(frozen=True)
class Task:
    dataset: str
    embedding: str
    seed: int

    @property
    def label(self) -> str:
        return f"{self.dataset}/{self.embedding}/seed_{self.seed}"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=SCRIPT_DIR.parent)
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--gpus", nargs="+", type=int, default=(0, 1, 2, 3))
    parser.add_argument("--datasets", nargs="+", choices=DATASETS, default=DATASETS)
    parser.add_argument("--embeddings", nargs="+", choices=VARIANTS, default=VARIANTS)
    parser.add_argument("--seeds", nargs="+", type=int, default=(16, 26, 36))
    parser.add_argument("--output-dir", type=Path, default=Path("results_embedding_reviewer"))
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
    parser.add_argument("--no-plots", action="store_true")
    return parser


def resolve_under(root: Path, path: Path) -> Path:
    path = path.expanduser()
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def preflight(args: argparse.Namespace, root: Path) -> None:
    if len(set(args.gpus)) != len(args.gpus):
        raise ValueError("--gpus contains duplicates")
    if args.input_len % args.seg_len or args.horizon % args.seg_len:
        raise ValueError("--seg-len must divide input length and horizon")
    required = {
        "milan_sms": root / "dataset" / "data_git_version.h5",
        "milan_call": root / "dataset" / "data_git_version.h5",
        "milan_internet": root / "dataset" / "data_git_version.h5",
        "mobile_bj": root / "dataset" / "MobileBJ.npy",
        "mobile_nj": root / "dataset" / "MobileNJ.npy",
    }
    missing = sorted({str(required[name]) for name in args.datasets if not required[name].is_file()})
    if missing:
        raise FileNotFoundError("Missing datasets: " + ", ".join(missing))
    if not args.python.expanduser().resolve().is_file():
        raise FileNotFoundError(f"Python not found: {args.python}")
    if not args.dry_run:
        code = (
            "import torch,numpy,h5py,matplotlib,reformer_pytorch,einops; "
            "assert torch.cuda.is_available(), 'CUDA unavailable'; "
            f"assert torch.cuda.device_count()>{max(args.gpus)}, 'GPU index unavailable'; "
            "print('PREFLIGHT OK',torch.__version__,torch.cuda.device_count())"
        )
        subprocess.check_call([str(args.python.resolve()), "-u", "-c", code], cwd=root)


def command_for(args: argparse.Namespace, task: Task, gpu: int, root: Path, output: Path) -> List[str]:
    return [
        str(args.python.resolve()), "-u", str(SCRIPT_DIR / "train_embedding_ablation.py"),
        "--project-root", str(root), "--dataset", task.dataset,
        "--embedding-variant", task.embedding, "--variant", "full_v2",
        "--close-size", str(args.input_len), "--predict-size", str(args.horizon),
        "--seg-len", str(args.seg_len), "--d-model", str(args.d_model),
        "--epochs", str(args.epochs), "--batch-size", str(args.batch_size),
        "--learning-rate", str(args.learning_rate), "--loss", args.loss,
        "--branch-aux-weight", str(args.branch_aux_weight), "--patience", str(args.patience),
        "--split-mode", "chronological", "--split-gap", str(args.split_gap),
        "--milan-loader", "clean", "--seed", str(task.seed),
        "--num-workers", str(args.num_workers), "--device", "cuda", "--gpu", str(gpu),
        "--output-dir", str(output),
    ]


def worker(gpu, tasks, args, root, output, lock, results) -> None:
    while True:
        try:
            task = tasks.get_nowait()
        except queue.Empty:
            return
        run_dir = output / task.dataset / task.embedding / f"seed_{task.seed}"
        metrics = run_dir / "metrics.json"
        log_dir = output / "logs" / task.dataset
        log_dir.mkdir(parents=True, exist_ok=True)
        log_path = log_dir / f"{task.embedding}_seed_{task.seed}.log"
        prefix = f"[GPU{gpu} {task.label}]"
        if metrics.is_file() and not args.force:
            with lock:
                print(f"{prefix} SKIP complete", flush=True)
            results.append((task, 0)); tasks.task_done(); continue
        command = command_for(args, task, gpu, root, output)
        if args.dry_run:
            with lock:
                print(prefix, " ".join(command), flush=True)
            results.append((task, 0)); tasks.task_done(); continue
        env: Dict[str, str] = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"
        with lock:
            print(f"{prefix} START log={log_path}", flush=True)
        with log_path.open("w", encoding="utf-8") as stream:
            process = subprocess.Popen(
                command, cwd=root, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, encoding="utf-8", errors="replace", bufsize=1, env=env,
            )
            assert process.stdout is not None
            for line in process.stdout:
                stream.write(line); stream.flush()
                with lock:
                    print(f"{prefix} {line}", end="", flush=True)
            code = process.wait()
        with lock:
            print(f"{prefix} {'DONE' if code == 0 else f'FAILED({code})'}", flush=True)
        results.append((task, code)); tasks.task_done()


def main() -> None:
    args = build_parser().parse_args()
    root = args.project_root.expanduser().resolve()
    output = resolve_under(root, args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    preflight(args, root)
    tasks: "queue.Queue[Task]" = queue.Queue()
    for embedding in args.embeddings:
        for dataset in args.datasets:
            for seed in args.seeds:
                tasks.put(Task(dataset, embedding, seed))
    print(f"Scheduling {tasks.qsize()} embedding tasks on GPUs {args.gpus}; output={output}", flush=True)
    lock = threading.Lock(); results: List[Tuple[Task, int]] = []
    threads = [threading.Thread(target=worker, args=(gpu, tasks, args, root, output, lock, results)) for gpu in args.gpus]
    for thread in threads: thread.start()
    for thread in threads: thread.join()
    failures = [(task, code) for task, code in results if code]
    if failures:
        for task, code in failures: print(f"FAILED {task.label}: {code}", flush=True)
        raise SystemExit(1)
    print(f"All {len(results)} embedding tasks completed.", flush=True)
    if not args.no_plots and not args.dry_run:
        subprocess.check_call(
            [str(args.python.resolve()), "-u", str(SCRIPT_DIR / "summarize_embedding_results.py"),
             "--results-dir", str(output), "--seeds", *[str(seed) for seed in args.seeds]],
            cwd=root,
        )


if __name__ == "__main__":
    main()
