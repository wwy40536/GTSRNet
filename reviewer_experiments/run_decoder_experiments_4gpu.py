"""Schedule PMF/RMF H=12/H=24 multi-seed experiments across CUDA GPUs."""

from __future__ import annotations

import argparse
import json
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


@dataclass(frozen=True)
class Task:
    dataset: str
    decoder: str
    horizon: int
    seed: int

    @property
    def label(self) -> str:
        return f"{self.dataset}/{self.decoder}/H{self.horizon}/seed_{self.seed}"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=SCRIPT_DIR.parent)
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--gpus", nargs="+", type=int, default=(0, 1, 2, 3))
    parser.add_argument("--datasets", nargs="+", choices=DATASETS, default=("milan_call", "milan_internet"))
    parser.add_argument("--decoders", nargs="+", choices=("pmf", "rmf"), default=("pmf", "rmf"))
    parser.add_argument("--horizons", nargs="+", type=int, default=(12, 24))
    parser.add_argument("--seeds", nargs="+", type=int, default=(45, 46, 47, 48, 49))
    parser.add_argument("--output-dir", type=Path, default=Path("results_decoder_reviewer"))
    parser.add_argument("--input-len", type=int, default=24)
    parser.add_argument("--seg-len", type=int, default=6)
    parser.add_argument("--baseline", choices=("last", "seasonal"), default="last")
    parser.add_argument("--seasonal-period", type=int, default=24)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--loss", choices=("l1", "l2"), default="l1")
    parser.add_argument("--patience", type=int, default=40)
    parser.add_argument("--split-gap", type=int, default=-1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--noise-levels", nargs="+", type=float, default=(0.0, 0.01, 0.03, 0.05, 0.10))
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
    if any(h <= 0 or h % args.seg_len for h in args.horizons):
        raise ValueError("Each horizon must be positive and divisible by --seg-len")
    if args.input_len % args.seg_len:
        raise ValueError("--seg-len must divide --input-len")
    if args.baseline == "seasonal" and not 0 < args.seasonal_period <= args.input_len:
        raise ValueError("--baseline seasonal requires 0 < --seasonal-period <= --input-len")
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
            "import torch,numpy,h5py,matplotlib; "
            "assert torch.cuda.is_available(), 'CUDA unavailable'; "
            f"assert torch.cuda.device_count()>{max(args.gpus)}, 'GPU index unavailable'; "
            "print('PREFLIGHT OK',torch.__version__,torch.cuda.device_count())"
        )
        subprocess.check_call([str(args.python.resolve()), "-u", "-c", code], cwd=root)


def command_for(args: argparse.Namespace, task: Task, gpu: int, root: Path, output: Path) -> List[str]:
    command = [
        str(args.python.resolve()), "-u", str(SCRIPT_DIR / "train_decoder_experiment.py"),
        "--project-root", str(root), "--dataset", task.dataset,
        "--decoder", task.decoder, "--input-len", str(args.input_len),
        "--horizon", str(task.horizon), "--seg-len", str(args.seg_len),
        "--baseline", args.baseline, "--seasonal-period", str(args.seasonal_period),
        "--d-model", str(args.d_model), "--dropout", str(args.dropout),
        "--epochs", str(args.epochs), "--batch-size", str(args.batch_size),
        "--learning-rate", str(args.learning_rate), "--loss", args.loss,
        "--patience", str(args.patience), "--split-mode", "chronological",
        "--split-gap", str(args.split_gap), "--milan-loader", "clean",
        "--seed", str(task.seed), "--num-workers", str(args.num_workers),
        "--device", "cuda", "--gpu", str(gpu), "--output-dir", str(output),
        "--noise-levels", *[str(level) for level in args.noise_levels],
    ]
    return command


def worker(
    gpu: int,
    tasks: "queue.Queue[Task]",
    args: argparse.Namespace,
    root: Path,
    output: Path,
    print_lock: threading.Lock,
    results: List[Tuple[Task, int]],
) -> None:
    while True:
        try:
            task = tasks.get_nowait()
        except queue.Empty:
            return
        run_dir = output / task.dataset / task.decoder / f"H{task.horizon}" / f"seed_{task.seed}"
        summary = run_dir / "summary.json"
        noise = run_dir / "noise_sweep.json"
        log_dir = output / "logs" / task.dataset / f"H{task.horizon}"
        log_dir.mkdir(parents=True, exist_ok=True)
        log_path = log_dir / f"{task.decoder}_seed_{task.seed}.log"
        prefix = f"[GPU{gpu} {task.label}]"
        if summary.is_file() and not args.force:
            # Never silently mix baselines inside one output directory.
            config = json.loads(summary.read_text(encoding="utf-8")).get("config", {})
            old_baseline = config.get("baseline", "last")
            old_period = int(config.get("seasonal_period", 24))
            if old_baseline != args.baseline or (
                args.baseline == "seasonal" and old_period != args.seasonal_period
            ):
                with print_lock:
                    print(
                        f"{prefix} FAILED: existing run uses baseline={old_baseline} "
                        f"(period={old_period}); requested {args.baseline} "
                        f"(period={args.seasonal_period}). Use a different --output-dir.",
                        flush=True,
                    )
                results.append((task, 2))
                tasks.task_done()
                continue
        if summary.is_file() and noise.is_file() and not args.force:
            with print_lock:
                print(f"{prefix} SKIP complete", flush=True)
            results.append((task, 0))
            tasks.task_done()
            continue
        command = command_for(args, task, gpu, root, output)
        if args.dry_run:
            with print_lock:
                print(prefix, " ".join(command), flush=True)
            results.append((task, 0))
            tasks.task_done()
            continue
        environment: Dict[str, str] = os.environ.copy()
        environment["PYTHONUNBUFFERED"] = "1"
        with print_lock:
            print(f"{prefix} START log={log_path}", flush=True)
        with log_path.open("w", encoding="utf-8") as log_stream:
            process = subprocess.Popen(
                command, cwd=root, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, encoding="utf-8", errors="replace", bufsize=1, env=environment,
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
        results.append((task, code))
        tasks.task_done()


def main() -> None:
    args = build_parser().parse_args()
    root = args.project_root.expanduser().resolve()
    output = resolve_under(root, args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    preflight(args, root)
    task_queue: "queue.Queue[Task]" = queue.Queue()
    for horizon in args.horizons:
        for dataset in args.datasets:
            for seed in args.seeds:
                for decoder in args.decoders:
                    task_queue.put(Task(dataset, decoder, horizon, seed))
    print(f"Scheduling {task_queue.qsize()} tasks on GPUs {args.gpus}; output={output}", flush=True)
    lock = threading.Lock()
    results: List[Tuple[Task, int]] = []
    workers = [
        threading.Thread(target=worker, args=(gpu, task_queue, args, root, output, lock, results))
        for gpu in args.gpus
    ]
    for thread in workers:
        thread.start()
    for thread in workers:
        thread.join()
    failures = [(task, code) for task, code in results if code]
    if failures:
        for task, code in failures:
            print(f"FAILED {task.label}: {code}", flush=True)
        raise SystemExit(1)
    print(f"All {len(results)} decoder tasks completed.", flush=True)
    if not args.no_plots and not args.dry_run:
        subprocess.check_call(
            [str(args.python.resolve()), "-u", str(SCRIPT_DIR / "summarize_decoder_results.py"),
             "--results-dir", str(output), "--seeds", *[str(seed) for seed in args.seeds]],
            cwd=root,
        )


if __name__ == "__main__":
    main()
