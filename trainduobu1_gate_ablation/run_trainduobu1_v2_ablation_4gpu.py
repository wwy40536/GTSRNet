"""Run corrected trainduobu1 variants, including full_v2, on a GPU work queue.

Every child process is unbuffered. Its output is printed live with a task prefix
and simultaneously saved to a per-task log file.
"""

from __future__ import annotations

import argparse
import os
import queue
import subprocess
import sys
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_PROJECT_ROOT = SCRIPT_DIR.parent
DATASETS = ("milan_sms", "milan_call", "milan_internet", "mobile_bj", "mobile_nj")
VARIANTS = (
    "full",
    "full_v2",
    "full_v2_aux0",
    "fixed_avg",
    "horizon_static",
    "wo_g",
    "wo_seg",
    "wo_sthe",
    "wo_s",
    "wo_t",
)


@dataclass(frozen=True)
class Task:
    dataset: str
    variant: str

    @property
    def label(self) -> str:
        return f"{self.dataset}/{self.variant}"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Dynamically schedule corrected trainduobu1/full_v2 ablations."
    )
    parser.add_argument("--project-root", type=Path, default=DEFAULT_PROJECT_ROOT)
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--gpus", nargs="+", type=int, default=(0, 1, 2, 3))
    parser.add_argument("--datasets", nargs="+", choices=DATASETS, default=DATASETS)
    parser.add_argument("--variants", nargs="+", choices=VARIANTS, default=VARIANTS)
    parser.add_argument("--output-dir", type=Path, default=Path("results_ablation"))
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--loss", choices=("l1", "l2"), default="l1")
    parser.add_argument(
        "--branch-aux-weight",
        type=float,
        default=0.1,
        help="Training-only branch auxiliary-loss weight shared by two-branch variants.",
    )
    parser.add_argument(
        "--patience", type=int, default=0,
        help="Early-stop patience; 0 runs the paper's full epoch count."
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--split-mode", choices=("chronological", "random"), default="chronological"
    )
    parser.add_argument(
        "--split-gap", type=int, default=-1,
        help="-1 automatically purges predict_size-1 windows at split boundaries."
    )
    parser.add_argument(
        "--milan-loader", choices=("clean", "legacy"), default="clean"
    )
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--log-every", type=int, default=1)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--seg-len", type=int, default=6)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--e-layers", type=int, default=2)
    parser.add_argument("--force", action="store_true", help="Retrain completed tasks.")
    parser.add_argument(
        "--no-plot", action="store_true", help="Do not generate figures after all jobs pass."
    )
    return parser


def resolve_under_project(project_root: Path, path: Path) -> Path:
    path = path.expanduser()
    return path.resolve() if path.is_absolute() else (project_root / path).resolve()


def preflight(args: argparse.Namespace, project_root: Path) -> None:
    required_files = {
        "milan_sms": project_root / "dataset" / "data_git_version.h5",
        "milan_call": project_root / "dataset" / "data_git_version.h5",
        "milan_internet": project_root / "dataset" / "data_git_version.h5",
        "mobile_bj": project_root / "dataset" / "MobileBJ.npy",
        "mobile_nj": project_root / "dataset" / "MobileNJ.npy",
    }
    missing = sorted(
        {str(required_files[name]) for name in args.datasets if not required_files[name].is_file()}
    )
    feature_path = project_root / "dataset" / "crawled_feature.csv"
    if (
        args.milan_loader == "legacy"
        and any(name.startswith("milan_") for name in args.datasets)
        and not feature_path.is_file()
    ):
        missing.append(str(feature_path))
    if missing:
        raise FileNotFoundError("Missing required dataset files: " + ", ".join(missing))

    check_code = (
        "import torch,h5py,sklearn,pandas,numpy,matplotlib,reformer_pytorch,einops; "
        "assert torch.cuda.is_available(), 'CUDA is unavailable'; "
        f"assert torch.cuda.device_count() > {max(args.gpus)}, "
        f"'Need GPU index {max(args.gpus)}, found %d GPU(s)' % torch.cuda.device_count(); "
        "print('PREFLIGHT OK: CUDA GPUs=', torch.cuda.device_count(), "
        "'PyTorch=', torch.__version__)"
    )
    subprocess.check_call(
        [str(args.python.resolve()), "-u", "-c", check_code], cwd=project_root
    )


def child_command(
    args: argparse.Namespace, task: Task, gpu: int, project_root: Path, output_root: Path
) -> List[str]:
    return [
        str(args.python.resolve()),
        "-u",
        str(SCRIPT_DIR / "train_trainduobu1_v2_ablation.py"),
        "--project-root",
        str(project_root),
        "--dataset",
        task.dataset,
        "--variant",
        task.variant,
        "--output-dir",
        str(output_root),
        "--device",
        "cuda",
        "--gpu",
        str(gpu),
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
        "--seed",
        str(args.seed),
        "--split-mode",
        args.split_mode,
        "--split-gap",
        str(args.split_gap),
        "--milan-loader",
        args.milan_loader,
        "--num-workers",
        str(args.num_workers),
        "--log-every",
        str(args.log_every),
        "--e-layers",
        str(args.e_layers),
        "--seg-len",
        str(args.seg_len),
        "--d-model",
        str(args.d_model),
        "--n-heads",
        str(args.n_heads),
    ]


def run_worker(
    gpu: int,
    tasks: "queue.Queue[Task]",
    args: argparse.Namespace,
    project_root: Path,
    output_root: Path,
    print_lock: threading.Lock,
    results: List[Tuple[Task, int]],
) -> None:
    while True:
        try:
            task = tasks.get_nowait()
        except queue.Empty:
            return

        metrics_path = output_root / task.dataset / task.variant / "metrics.json"
        log_dir = output_root / "logs" / task.dataset
        log_dir.mkdir(parents=True, exist_ok=True)
        log_path = log_dir / f"{task.variant}.log"
        prefix = f"[GPU{gpu} {task.label}]"

        if metrics_path.is_file() and not args.force:
            with print_lock:
                print(f"{prefix} SKIP: metrics.json already exists", flush=True)
            results.append((task, 0))
            tasks.task_done()
            continue

        command = child_command(args, task, gpu, project_root, output_root)
        environment: Dict[str, str] = os.environ.copy()
        environment["PYTHONUNBUFFERED"] = "1"
        with print_lock:
            print(f"{prefix} START", flush=True)
            print(f"{prefix} LOG  {log_path}", flush=True)

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
            return_code = process.wait()

        with print_lock:
            state = "DONE" if return_code == 0 else f"FAILED({return_code})"
            print(f"{prefix} {state}", flush=True)
        results.append((task, return_code))
        tasks.task_done()


def generate_plots(args: argparse.Namespace, project_root: Path, output_root: Path) -> int:
    plot_script = SCRIPT_DIR / "plot_trainduobu1_v2_ablation.py"
    if not plot_script.is_file():
        print("Plot script is not included in the gate-ablation package; skipping plots.")
        return 0
    command = [
        str(args.python.resolve()),
        "-u",
        str(plot_script),
        "--results-dir",
        str(output_root),
    ]
    # A one-variant append (for example only full_v2) is valid even when this
    # directory does not yet contain the six reference variants.
    if set(args.variants) != set(VARIANTS):
        command.append("--allow-partial")
    return subprocess.call(command, cwd=project_root)


def main() -> None:
    args = build_parser().parse_args()
    project_root = args.project_root.expanduser().resolve()
    output_root = resolve_under_project(project_root, args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    if not args.python.expanduser().resolve().is_file():
        raise FileNotFoundError(f"Python interpreter not found: {args.python}")
    if len(set(args.gpus)) != len(args.gpus):
        raise ValueError("--gpus contains duplicate GPU indices")
    if args.branch_aux_weight < 0:
        raise ValueError("--branch-aux-weight must be non-negative")
    preflight(args, project_root)

    task_queue: "queue.Queue[Task]" = queue.Queue()
    # Run the same model variant across datasets first. This produces all full
    # baselines early, then advances through the five ablations.
    for variant in args.variants:
        for dataset in args.datasets:
            task_queue.put(Task(dataset, variant))

    print_lock = threading.Lock()
    results: List[Tuple[Task, int]] = []
    workers = [
        threading.Thread(
            target=run_worker,
            args=(gpu, task_queue, args, project_root, output_root, print_lock, results),
            name=f"gpu-{gpu}",
        )
        for gpu in args.gpus
    ]
    print(
        f"Scheduling {task_queue.qsize()} tasks on GPUs {list(args.gpus)}. "
        f"Results: {output_root}",
        flush=True,
    )
    for worker in workers:
        worker.start()
    for worker in workers:
        worker.join()

    failures = [(task, code) for task, code in results if code != 0]
    if failures:
        print("\nFailed tasks:", flush=True)
        for task, code in failures:
            print(f"  {task.label}: exit code {code}", flush=True)
        raise SystemExit(1)

    print(f"\nAll {len(results)} tasks completed successfully.", flush=True)
    if not args.no_plot:
        plot_code = generate_plots(args, project_root, output_root)
        if plot_code != 0:
            raise SystemExit(plot_code)


if __name__ == "__main__":
    main()
