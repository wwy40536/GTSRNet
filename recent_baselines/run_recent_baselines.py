"""Run iTransformer, SegRNN and STID on five datasets with a four-GPU queue."""

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
MODELS = ("itransformer", "segrnn", "stid")
DATASETS = ("milan_sms", "milan_call", "milan_internet", "mobile_bj", "mobile_nj")


@dataclass(frozen=True)
class Task:
    model: str
    dataset: str
    seed: int

    @property
    def label(self) -> str:
        return f"{self.dataset}/{self.model}/seed_{self.seed}"


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--project-root", type=Path, default=SCRIPT_DIR.parent)
    p.add_argument("--python", type=Path, default=Path(sys.executable))
    p.add_argument("--output-dir", type=Path, default=Path("results_recent_baselines_4seed"))
    p.add_argument("--models", nargs="+", choices=MODELS, default=MODELS)
    p.add_argument("--datasets", nargs="+", choices=DATASETS, default=DATASETS)
    p.add_argument("--seeds", nargs="+", type=int, default=(23, 29, 103, 128))
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
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--log-every", type=int, default=1)
    p.add_argument("--d-model", type=int, default=128)
    p.add_argument("--seg-len", type=int, default=6)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--n-heads", type=int, default=4)
    p.add_argument("--e-layers", type=int, default=2)
    p.add_argument("--d-ff", type=int, default=256)
    p.add_argument("--force", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    return p


def resolve_under(root: Path, path: Path) -> Path:
    path = path.expanduser()
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def validate(args) -> None:
    if not args.gpus or len(args.gpus) != len(set(args.gpus)):
        raise ValueError("--gpus must contain unique GPU indices")
    if not args.seeds or len(args.seeds) != len(set(args.seeds)):
        raise ValueError("--seeds must contain unique values")
    if min(args.close_size, args.predict_size, args.epochs, args.batch_size) <= 0:
        raise ValueError("lengths, epochs and batch size must be positive")


def preflight(args, project_root: Path) -> None:
    required = {
        "milan_sms": project_root / "dataset" / "data_git_version.h5",
        "milan_call": project_root / "dataset" / "data_git_version.h5",
        "milan_internet": project_root / "dataset" / "data_git_version.h5",
        "mobile_bj": project_root / "dataset" / "MobileBJ.npy",
        "mobile_nj": project_root / "dataset" / "MobileNJ.npy",
    }
    missing = sorted({str(required[d]) for d in args.datasets if not required[d].is_file()})
    base_candidates = (SCRIPT_DIR / "training_core.py", SCRIPT_DIR.parent / "GTSRNet_baseline_benchmark" / "training_core.py")
    if not any(path.is_file() for path in base_candidates):
        missing.append("GTSRNet_baseline_benchmark/training_core.py")
    python = args.python.expanduser().resolve()
    if not python.is_file():
        missing.append(str(python))
    if missing:
        raise FileNotFoundError("Missing required files: " + ", ".join(missing))
    if args.dry_run:
        return
    check = (
        "import torch,h5py,numpy,matplotlib; "
        "assert torch.cuda.is_available(), 'CUDA unavailable'; "
        f"assert torch.cuda.device_count()>{max(args.gpus)}, "
        f"'Need GPU {max(args.gpus)}, found %d GPU(s)' % torch.cuda.device_count(); "
        "print('PREFLIGHT OK', torch.__version__, torch.cuda.device_count())"
    )
    subprocess.check_call([str(python), "-u", "-c", check], cwd=project_root)


def command(args, task: Task, gpu: int, project_root: Path, output_root: Path) -> List[str]:
    return [
        str(args.python.expanduser().resolve()), "-u", str(SCRIPT_DIR / "train_recent_baseline.py"),
        "--project-root", str(project_root), "--output-dir", str(output_root),
        "--model", task.model, "--dataset", task.dataset, "--seed", str(task.seed),
        "--device", "cuda", "--gpu", str(gpu),
        "--close-size", str(args.close_size), "--predict-size", str(args.predict_size),
        "--epochs", str(args.epochs), "--batch-size", str(args.batch_size),
        "--learning-rate", str(args.learning_rate), "--weight-decay", str(args.weight_decay),
        "--loss", args.loss, "--patience", str(args.patience), "--val-ratio", str(args.val_ratio),
        "--split-mode", args.split_mode, "--split-gap", str(args.split_gap),
        "--num-workers", str(args.num_workers), "--log-every", str(args.log_every),
        "--d-model", str(args.d_model), "--seg-len", str(args.seg_len),
        "--dropout", str(args.dropout), "--n-heads", str(args.n_heads),
        "--e-layers", str(args.e_layers), "--d-ff", str(args.d_ff),
    ]


def worker(gpu: int, tasks: "queue.Queue[Task]", args, project_root: Path, output_root: Path, lock: threading.Lock, outcomes: List[Tuple[Task, int]]) -> None:
    while True:
        try:
            task = tasks.get_nowait()
        except queue.Empty:
            return
        prefix = f"[GPU{gpu} {task.label}]"
        metric = output_root / task.dataset / task.model / f"seed_{task.seed}" / "metrics.json"
        if metric.is_file() and not args.force:
            with lock:
                print(f"{prefix} SKIP: metrics already exists", flush=True)
            outcomes.append((task, 0)); tasks.task_done(); continue
        cmd = command(args, task, gpu, project_root, output_root)
        log = output_root / "logs" / task.dataset / task.model / f"seed_{task.seed}.log"
        log.parent.mkdir(parents=True, exist_ok=True)
        with lock:
            print(f"{prefix} START\n{prefix} CMD {' '.join(shlex.quote(x) for x in cmd)}\n{prefix} LOG {log}", flush=True)
        if args.dry_run:
            outcomes.append((task, 0)); tasks.task_done(); continue
        env: Dict[str, str] = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"
        env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
        with log.open("w", encoding="utf-8") as stream:
            process = subprocess.Popen(cmd, cwd=project_root, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace", bufsize=1, env=env)
            assert process.stdout is not None
            for line in process.stdout:
                stream.write(line); stream.flush()
                with lock:
                    print(f"{prefix} {line}", end="", flush=True)
            code = process.wait()
        with lock:
            print(f"{prefix} {'DONE' if code == 0 else f'FAILED({code})'}", flush=True)
        outcomes.append((task, code)); tasks.task_done()


def main() -> None:
    args = parser().parse_args(); validate(args)
    project_root = args.project_root.expanduser().resolve()
    output_root = resolve_under(project_root, args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    preflight(args, project_root)
    tasks: "queue.Queue[Task]" = queue.Queue()
    for model in args.models:
        for dataset in args.datasets:
            for seed in args.seeds:
                tasks.put(Task(model, dataset, seed))
    total = tasks.qsize()
    print(f"Scheduling {total} runs on GPUs {list(args.gpus)}; results={output_root}", flush=True)
    lock = threading.Lock(); outcomes: List[Tuple[Task, int]] = []
    threads = [threading.Thread(target=worker, args=(gpu, tasks, args, project_root, output_root, lock, outcomes), name=f"gpu-{gpu}") for gpu in args.gpus]
    for thread in threads: thread.start()
    for thread in threads: thread.join()
    failures = [(task, code) for task, code in outcomes if code]
    if failures:
        for task, code in failures:
            print(f"FAILED {task.label}: exit={code}", flush=True)
        raise SystemExit(1)
    print(f"All {len(outcomes)} runs completed or were skipped.", flush=True)


if __name__ == "__main__":
    main()
