"""Run the reviewer-requested paired mechanism experiments on a GPU queue."""

from __future__ import annotations

import argparse
import json
import os
import queue
import shlex
import shutil
import subprocess
import sys
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
DATASETS = ("milan_sms", "milan_call", "milan_internet", "mobile_bj", "mobile_nj")
MILAN_DATASETS = ("milan_sms", "milan_call", "milan_internet")
PHASE_MODES = ("original_phase", "no_periodic")


@dataclass(frozen=True)
class Task:
    dataset: str
    seed: int
    variant: str
    phase_mode: str

    @property
    def label(self) -> str:
        return f"{self.phase_mode}/seed_{self.seed}/{self.dataset}/{self.variant}"


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Train GTSRNet, wo_s, and no_periodic with paired seeds, save test "
            "predictions, then make per-seed grid/hour error-difference figures."
        )
    )
    p.add_argument("--project-root", type=Path, default=SCRIPT_DIR.parent)
    p.add_argument("--python", type=Path, default=Path(sys.executable))
    p.add_argument("--results-dir", type=Path, default=Path("results_mechanism_error_maps"))
    p.add_argument(
        "--baseline-results",
        type=Path,
        default=Path("results_GTSRNet_baselines"),
        help=(
            "Optional existing baseline root laid out as "
            "<dataset>/GTSRNet/seed_<seed>/best.pt. Compatible GTSRNet checkpoints "
            "are reused for evaluation instead of retrained."
        ),
    )
    p.add_argument("--spatial-datasets", nargs="+", choices=DATASETS, default=DATASETS)
    p.add_argument("--phase-datasets", nargs="+", choices=MILAN_DATASETS, default=MILAN_DATASETS)
    p.add_argument("--seeds", nargs="+", type=int, default=(16, 26, 36))
    p.add_argument("--gpus", nargs="+", type=int, default=(0, 1, 2, 3))
    p.add_argument("--epochs", type=int, default=300)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--learning-rate", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--branch-aux-weight", type=float, default=0.1)
    p.add_argument("--patience", type=int, default=0)
    p.add_argument("--loss", choices=("l1", "l2"), default="l1")
    p.add_argument("--split-mode", choices=("chronological", "random"), default="chronological")
    p.add_argument("--split-gap", type=int, default=-1)
    p.add_argument("--milan-loader", choices=("clean", "legacy"), default="clean")
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--log-every", type=int, default=1)
    p.add_argument("--close-size", type=int, default=24)
    p.add_argument("--predict-size", type=int, default=12)
    p.add_argument("--d-model", type=int, default=128)
    p.add_argument("--seg-len", type=int, default=6)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--n-heads", type=int, default=4)
    p.add_argument("--e-layers", type=int, default=2)
    p.add_argument("--d-ff", type=int, default=256)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--force", action="store_true", help="Retrain even if predictions exist.")
    p.add_argument("--no-analysis", action="store_true")
    return p


def resolve(root: Path, value: Path) -> Path:
    value = value.expanduser()
    return value.resolve() if value.is_absolute() else (root / value).resolve()


def prediction_path(results: Path, task: Task) -> Path:
    return (
        results
        / task.phase_mode
        / f"seed_{task.seed}"
        / task.dataset
        / task.variant
        / "predictions.npz"
    )


def checkpoint_path(results: Path, task: Task) -> Path:
    return prediction_path(results, task).with_name("best.pt")


def output_root(results: Path, task: Task) -> Path:
    return results / task.phase_mode / f"seed_{task.seed}"


def make_tasks(
    spatial_datasets: Sequence[str], phase_datasets: Sequence[str], seeds: Sequence[int]
) -> List[Task]:
    tasks: List[Task] = []
    for seed in seeds:
        for dataset in spatial_datasets:
            tasks.append(Task(dataset, seed, "GTSRNet", "original_phase"))
            tasks.append(Task(dataset, seed, "wo_s", "original_phase"))
        for dataset in phase_datasets:
            if dataset not in spatial_datasets:
                tasks.append(Task(dataset, seed, "GTSRNet", "original_phase"))
            tasks.append(Task(dataset, seed, "GTSRNet", "no_periodic"))
    return tasks


def child_command(args: argparse.Namespace, task: Task, gpu: int, root: Path, results: Path) -> List[str]:
    command = [
        str(args.python.expanduser().resolve()),
        "-u",
        str(SCRIPT_DIR / "train_mechanism_pair.py"),
        "--project-root", str(root),
        "--output-dir", str(output_root(results, task)),
        "--dataset", task.dataset,
        "--variant", task.variant,
        "--phase-mode", task.phase_mode,
        "--device", "cuda",
        "--gpu", str(gpu),
        "--seed", str(task.seed),
        "--epochs", str(args.epochs),
        "--batch-size", str(args.batch_size),
        "--learning-rate", str(args.learning_rate),
        "--weight-decay", str(args.weight_decay),
        "--branch-aux-weight", str(args.branch_aux_weight),
        "--patience", str(args.patience),
        "--loss", args.loss,
        "--split-mode", args.split_mode,
        "--split-gap", str(args.split_gap),
        "--milan-loader", args.milan_loader,
        "--num-workers", str(args.num_workers),
        "--log-every", str(args.log_every),
        "--close-size", str(args.close_size),
        "--predict-size", str(args.predict_size),
        "--d-model", str(args.d_model),
        "--seg-len", str(args.seg_len),
        "--dropout", str(args.dropout),
        "--n-heads", str(args.n_heads),
        "--e-layers", str(args.e_layers),
        "--d-ff", str(args.d_ff),
        "--save-predictions",
    ]
    if checkpoint_path(results, task).is_file() and not args.force:
        command.append("--eval-only")
    return command


def preflight(args: argparse.Namespace, root: Path) -> None:
    if len(set(args.gpus)) != len(args.gpus) or len(set(args.seeds)) != len(args.seeds):
        raise ValueError("--gpus and --seeds must not contain duplicates")
    if not args.gpus or not args.seeds or not args.spatial_datasets or not args.phase_datasets:
        raise ValueError("At least one GPU, seed, spatial dataset, and phase dataset are required")
    if args.epochs <= 0 or args.batch_size <= 0 or args.learning_rate <= 0:
        raise ValueError("epochs, batch size, and learning rate must be positive")
    if args.close_size % args.seg_len or args.predict_size % args.seg_len:
        raise ValueError("--seg-len must divide close-size and predict-size")
    required = {
        "milan_sms": root / "dataset" / "data_git_version.h5",
        "milan_call": root / "dataset" / "data_git_version.h5",
        "milan_internet": root / "dataset" / "data_git_version.h5",
        "mobile_bj": root / "dataset" / "MobileBJ.npy",
        "mobile_nj": root / "dataset" / "MobileNJ.npy",
    }
    requested = set(args.spatial_datasets) | set(args.phase_datasets)
    missing = sorted({str(required[name]) for name in requested if not required[name].is_file()})
    if missing:
        raise FileNotFoundError("Missing dataset files: " + ", ".join(missing))
    if not (root / "utils").is_dir():
        raise FileNotFoundError(f"Missing original model utilities: {root / 'utils'}")
    python = args.python.expanduser().resolve()
    if not python.is_file():
        raise FileNotFoundError(f"Python not found: {python}")
    if args.dry_run:
        return
    check = (
        "import torch,h5py,numpy,matplotlib; "
        "assert torch.cuda.is_available(), 'CUDA is unavailable'; "
        f"assert torch.cuda.device_count()>{max(args.gpus)}, "
        f"'GPU index {max(args.gpus)} unavailable'; "
        "print('PREFLIGHT OK', torch.__version__, torch.cuda.device_count())"
    )
    subprocess.check_call([str(python), "-u", "-c", check], cwd=root)


def same_number(left: object, right: object) -> bool:
    try:
        return bool(np.isclose(float(left), float(right), rtol=1e-12, atol=1e-12))
    except (TypeError, ValueError):
        return False


def stage_baseline_checkpoints(
    args: argparse.Namespace, root: Path, results: Path, tasks: Sequence[Task]
) -> int:
    """Copy protocol-compatible GTSRNet checkpoints into the evaluation layout."""
    if args.force:
        return 0
    baseline = resolve(root, args.baseline_results)
    if not baseline.is_dir():
        print(f"Baseline reuse disabled: directory not found: {baseline}", flush=True)
        return 0
    expected_exact = {
        "close_size": args.close_size, "predict_size": args.predict_size,
        "batch_size": args.batch_size, "epochs": args.epochs, "loss": args.loss,
        "split_mode": args.split_mode, "split_gap": args.split_gap,
        "milan_loader": args.milan_loader, "d_model": args.d_model,
        "seg_len": args.seg_len, "n_heads": args.n_heads,
        "e_layers": args.e_layers, "d_ff": args.d_ff, "cycle_len": 24,
    }
    expected_numeric = {
        "learning_rate": args.learning_rate, "weight_decay": args.weight_decay,
        "branch_aux_weight": args.branch_aux_weight, "dropout": args.dropout,
    }
    staged = 0
    seen = set()
    for task in tasks:
        if task.variant != "GTSRNet" or task.phase_mode != "original_phase":
            continue
        key = (task.dataset, task.seed)
        if key in seen or prediction_path(results, task).is_file() or checkpoint_path(results, task).is_file():
            continue
        seen.add(key)
        source_dir = baseline / task.dataset / "GTSRNet" / f"seed_{task.seed}"
        source_checkpoint = source_dir / "best.pt"
        source_metrics = source_dir / "metrics.json"
        if not source_checkpoint.is_file() or not source_metrics.is_file():
            continue
        payload = json.loads(source_metrics.read_text(encoding="utf-8"))
        config = payload.get("config", {})
        mismatches = [
            key_name for key_name, value in expected_exact.items()
            if config.get(key_name) != value
        ]
        mismatches += [
            key_name for key_name, value in expected_numeric.items()
            if not same_number(config.get(key_name), value)
        ]
        if config.get("seed") != task.seed or config.get("dataset") != task.dataset:
            mismatches += ["seed/dataset"]
        if mismatches:
            raise ValueError(
                f"Refusing incompatible baseline reuse for {task.dataset}/seed_{task.seed}: "
                + ", ".join(mismatches)
            )
        destination = checkpoint_path(results, task)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_checkpoint, destination)
        (destination.parent / "REUSED_BASELINE.txt").write_text(
            f"Checkpoint copied from {source_checkpoint}\nConfiguration validated by runner.\n",
            encoding="utf-8",
        )
        print(f"REUSE GTSRNet checkpoint: {source_checkpoint} -> {destination}", flush=True)
        staged += 1
    return staged


def worker(
    gpu: int,
    jobs: "queue.Queue[Task]",
    args: argparse.Namespace,
    root: Path,
    results: Path,
    lock: threading.Lock,
    outcomes: List[Tuple[Task, int]],
) -> None:
    while True:
        try:
            task = jobs.get_nowait()
        except queue.Empty:
            return
        target = prediction_path(results, task)
        prefix = f"[GPU{gpu} {task.label}]"
        if target.is_file() and not args.force:
            with lock:
                print(f"{prefix} SKIP: predictions.npz exists", flush=True)
            outcomes.append((task, 0))
            jobs.task_done()
            continue
        command = child_command(args, task, gpu, root, results)
        log = results / "logs" / task.phase_mode / f"seed_{task.seed}" / f"{task.dataset}_{task.variant}.log"
        log.parent.mkdir(parents=True, exist_ok=True)
        with lock:
            print(f"{prefix} START", flush=True)
            print(f"{prefix} CMD {' '.join(shlex.quote(x) for x in command)}", flush=True)
            print(f"{prefix} LOG {log}", flush=True)
        if args.dry_run:
            outcomes.append((task, 0))
            jobs.task_done()
            continue
        env: Dict[str, str] = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"
        with log.open("w", encoding="utf-8") as stream:
            process = subprocess.Popen(
                command,
                cwd=root,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                bufsize=1,
                env=env,
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
        jobs.task_done()


def main() -> None:
    args = parser().parse_args()
    root = args.project_root.expanduser().resolve()
    results = resolve(root, args.results_dir)
    results.mkdir(parents=True, exist_ok=True)
    preflight(args, root)
    tasks = make_tasks(args.spatial_datasets, args.phase_datasets, args.seeds)
    if not args.dry_run:
        staged = stage_baseline_checkpoints(args, root, results, tasks)
        print(f"Staged {staged} compatible GTSRNet checkpoint(s) for eval-only reuse.", flush=True)
    jobs: "queue.Queue[Task]" = queue.Queue()
    for task in tasks:
        jobs.put(task)
    lock = threading.Lock()
    outcomes: List[Tuple[Task, int]] = []
    threads = [
        threading.Thread(
            target=worker,
            args=(gpu, jobs, args, root, results, lock, outcomes),
            name=f"gpu-{gpu}",
        )
        for gpu in args.gpus
    ]
    print(f"Scheduling {len(tasks)} tasks on GPUs {list(args.gpus)}; results={results}")
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    failed = [(task, code) for task, code in outcomes if code]
    if failed:
        for task, code in failed:
            print(f"FAILED {task.label}: exit code {code}", file=sys.stderr)
        raise SystemExit(1)
    if not args.dry_run and not args.no_analysis:
        command = [
            str(args.python.expanduser().resolve()),
            "-u",
            str(SCRIPT_DIR / "analyze_mechanism_errors.py"),
            "--results-dir", str(results),
            "--spatial-datasets", *args.spatial_datasets,
            "--phase-datasets", *args.phase_datasets,
            "--seeds", *(str(x) for x in args.seeds),
        ]
        subprocess.check_call(command, cwd=root)
    print(f"All {len(tasks)} tasks completed. Results: {results}", flush=True)


if __name__ == "__main__":
    main()
