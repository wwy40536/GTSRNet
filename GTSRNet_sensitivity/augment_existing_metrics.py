"""Re-evaluate existing checkpoints and add R2/sMAPE/MASE/PSNR metrics."""

from __future__ import annotations

import argparse
import csv
import json
import queue
import shutil
import subprocess
import sys
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Mapping, Sequence, Tuple


SCRIPT_DIR = Path(__file__).resolve().parent
VARIANTS = ("GTSRNet", "wo_g", "wo_seg", "wo_sthe", "wo_s", "wo_t", "full")


@dataclass(frozen=True)
class Task:
    metrics_path: Path
    dataset: str
    variant: str
    config: Mapping[str, object]

    @property
    def label(self) -> str:
        return f"{self.dataset}/{self.variant}"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Add interpretable metrics to existing results without retraining."
    )
    parser.add_argument("--project-root", type=Path, default=SCRIPT_DIR.parent)
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--variants", nargs="+", choices=VARIANTS, default=("GTSRNet",))
    parser.add_argument("--datasets", nargs="+", default=None)
    parser.add_argument("--gpus", nargs="+", type=int, default=(0, 1, 2, 3))
    parser.add_argument("--mase-seasonality", type=int, default=24)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def value(config: Mapping[str, object], key: str, default: object) -> object:
    candidate = config.get(key, default)
    return default if candidate is None else candidate


def discover(args: argparse.Namespace, results_dir: Path) -> List[Task]:
    tasks: List[Task] = []
    for path in sorted(results_dir.glob("*/*/metrics.json")):
        dataset, variant = path.parents[1].name, path.parent.name
        if variant not in args.variants:
            continue
        if args.datasets and dataset not in args.datasets:
            continue
        checkpoint = path.parent / "best.pt"
        if not checkpoint.is_file():
            print(f"SKIP {dataset}/{variant}: missing {checkpoint}", flush=True)
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        tasks.append(Task(path, dataset, variant, payload.get("config", {})))
    if not tasks:
        raise FileNotFoundError("No matching metrics.json + best.pt pairs found")
    return tasks


def command(
    args: argparse.Namespace,
    task: Task,
    gpu: int,
    project_root: Path,
    results_dir: Path,
) -> List[str]:
    config = task.config
    cmd = [
        str(args.python.expanduser().resolve()), "-u",
        str(SCRIPT_DIR / "train_sensitivity.py"),
        "--project-root", str(project_root),
        "--output-dir", str(results_dir),
        "--dataset", task.dataset,
        "--variant", task.variant,
        "--eval-only",
        "--device", "cuda",
        "--gpu", str(gpu),
        "--close-size", str(value(config, "close_size", 24)),
        "--predict-size", str(value(config, "predict_size", 12)),
        "--batch-size", str(value(config, "batch_size", 16)),
        "--epochs", str(value(config, "epochs", 300)),
        "--learning-rate", str(value(config, "learning_rate", 0.001)),
        "--weight-decay", str(value(config, "weight_decay", 0.0)),
        "--loss", str(value(config, "loss", "l1")),
        "--branch-aux-weight", str(value(config, "branch_aux_weight", 0.1)),
        "--val-ratio", str(value(config, "val_ratio", 0.1)),
        "--split-mode", str(value(config, "split_mode", "chronological")),
        "--split-gap", str(value(config, "split_gap", -1)),
        "--milan-loader", str(value(config, "milan_loader", "clean")),
        "--seed", str(value(config, "seed", 2)),
        "--num-workers", str(args.num_workers),
        "--d-model", str(value(config, "d_model", 128)),
        "--seg-len", str(value(config, "seg_len", 6)),
        "--dropout", str(value(config, "dropout", 0.1)),
        "--n-heads", str(value(config, "n_heads", 4)),
        "--e-layers", str(value(config, "e_layers", 2)),
        "--d-ff", str(value(config, "d_ff", 256)),
        "--cycle-len", str(value(config, "cycle_len", 24)),
        "--mase-seasonality", str(args.mase_seasonality),
    ]
    if bool(value(config, "crop", True)):
        cmd.append("--crop")
    else:
        cmd.append("--no-crop")
    rows = value(config, "rows", [40, 60])
    cols = value(config, "cols", [40, 60])
    if isinstance(rows, Sequence) and not isinstance(rows, str) and len(rows) == 2:
        cmd.extend(("--rows", str(rows[0]), str(rows[1])))
    if isinstance(cols, Sequence) and not isinstance(cols, str) and len(cols) == 2:
        cmd.extend(("--cols", str(cols[0]), str(cols[1])))
    test_size = int(value(config, "test_size", 0))
    if test_size:
        cmd.extend(("--test-size", str(test_size)))
    return cmd


def worker(
    gpu: int,
    pending: "queue.Queue[Task]",
    args: argparse.Namespace,
    project_root: Path,
    results_dir: Path,
    lock: threading.Lock,
    outcomes: List[Tuple[Task, int]],
) -> None:
    while True:
        try:
            task = pending.get_nowait()
        except queue.Empty:
            return
        cmd = command(args, task, gpu, project_root, results_dir)
        with lock:
            print(f"[GPU{gpu} {task.label}] {' '.join(cmd)}", flush=True)
        if args.dry_run:
            outcomes.append((task, 0))
            pending.task_done()
            continue
        original_payload = json.loads(task.metrics_path.read_text(encoding="utf-8"))
        original_rmse = float(original_payload["metrics"]["rmse"])
        original_mae = float(original_payload["metrics"]["mae"])
        backup = task.metrics_path.with_name("metrics_before_extra_metrics.json")
        if not backup.exists():
            shutil.copy2(task.metrics_path, backup)
        log = task.metrics_path.with_name("augment_metrics.log")
        with log.open("w", encoding="utf-8") as stream:
            process = subprocess.Popen(
                cmd,
                cwd=project_root,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
            )
            assert process.stdout is not None
            for line in process.stdout:
                stream.write(line)
                stream.flush()
                with lock:
                    print(f"[GPU{gpu} {task.label}] {line}", end="", flush=True)
            code = process.wait()
        if code == 0:
            updated = json.loads(task.metrics_path.read_text(encoding="utf-8"))
            updated_rmse = float(updated["metrics"]["rmse"])
            updated_mae = float(updated["metrics"]["mae"])
            tolerance = max(1e-6, 1e-6 * max(abs(original_rmse), 1.0))
            if (
                abs(updated_rmse - original_rmse) > tolerance
                or abs(updated_mae - original_mae) > tolerance
            ):
                shutil.copy2(backup, task.metrics_path)
                code = 2
                with lock:
                    print(
                        f"[GPU{gpu} {task.label}] REJECT: re-evaluated MAE/RMSE "
                        "do not match the saved result; restored original metrics.json",
                        flush=True,
                    )
        outcomes.append((task, code))
        pending.task_done()


def write_reports(tasks: Sequence[Task], results_dir: Path) -> None:
    metrics = ("rmse", "mae", "r2", "smape_percent", "mase", "psnr_db")
    overall_rows: List[Dict[str, object]] = []
    horizon_rows: List[Dict[str, object]] = []
    grid_rows: List[Dict[str, object]] = []
    for task in tasks:
        payload = json.loads(task.metrics_path.read_text(encoding="utf-8"))
        values = payload["metrics"]
        overall_rows.append(
            {
                "dataset": task.dataset,
                "variant": task.variant,
                **{name: values.get(name) for name in metrics},
            }
        )
        for row in values.get("horizons", []):
            horizon_rows.append(
                {
                    "dataset": task.dataset,
                    "variant": task.variant,
                    "forecast_step": row["horizon"],
                    **{name: row.get(name) for name in metrics},
                }
            )
        for row in values.get("grids", []):
            grid_rows.append(
                {
                    "dataset": task.dataset,
                    "variant": task.variant,
                    "grid_index": row["grid_index"],
                    "row": row["row"],
                    "col": row["col"],
                    **{name: row.get(name) for name in metrics},
                }
            )
    report_dir = results_dir / "extra_metrics_summary"
    report_dir.mkdir(parents=True, exist_ok=True)
    specifications = (
        ("overall_metrics.csv", overall_rows, ("dataset", "variant", *metrics)),
        ("per_horizon_metrics.csv", horizon_rows, ("dataset", "variant", "forecast_step", *metrics)),
        ("per_grid_metrics.csv", grid_rows, ("dataset", "variant", "grid_index", "row", "col", *metrics)),
    )
    for filename, rows, fields in specifications:
        path = report_dir / filename
        with path.open("w", newline="", encoding="utf-8-sig") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)
        print(f"WROTE {path}", flush=True)


def main() -> None:
    args = build_parser().parse_args()
    if args.mase_seasonality <= 0:
        raise ValueError("--mase-seasonality must be positive")
    if len(set(args.gpus)) != len(args.gpus):
        raise ValueError("--gpus contains duplicates")
    project_root = args.project_root.expanduser().resolve()
    results_dir = (
        args.results_dir.expanduser().resolve()
        if args.results_dir.is_absolute()
        else (project_root / args.results_dir).resolve()
    )
    tasks = discover(args, results_dir)
    pending: "queue.Queue[Task]" = queue.Queue()
    for task in tasks:
        pending.put(task)
    lock = threading.Lock()
    outcomes: List[Tuple[Task, int]] = []
    threads = [
        threading.Thread(
            target=worker,
            args=(gpu, pending, args, project_root, results_dir, lock, outcomes),
        )
        for gpu in args.gpus
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    failures = [(task, code) for task, code in outcomes if code]
    if failures:
        for task, code in failures:
            print(f"FAILED {task.label}: {code}", flush=True)
        raise SystemExit(1)
    if not args.dry_run:
        write_reports(tasks, results_dir)
    print(f"Updated {len(outcomes)} checkpoints without retraining.", flush=True)


if __name__ == "__main__":
    main()
