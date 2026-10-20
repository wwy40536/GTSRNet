"""Validate and export every w/o Joint run without averaging seeds."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--datasets", nargs="+", required=True)
    parser.add_argument("--seeds", nargs="+", type=int, required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    root = args.results_dir.expanduser().resolve()
    rows = []
    missing = []

    for seed in args.seeds:
        for dataset in args.datasets:
            path = root / dataset / "channel_phase" / f"seed_{seed}" / "metrics.json"
            if not path.is_file():
                missing.append(str(path))
                continue
            payload = json.loads(path.read_text(encoding="utf-8"))
            metrics = payload["metrics"]
            rows.append(
                {
                    "seed": seed,
                    "dataset": dataset,
                    "variant": "w/o Joint",
                    "embedding_variant": "channel_phase",
                    "rmse": float(metrics["rmse"]),
                    "mae": float(metrics["mae"]),
                    "best_epoch": int(metrics["best_epoch"]),
                    "parameters": int(metrics["parameters"]),
                    "embedding_parameters": int(metrics["embedding_parameters"]),
                    "inference_seconds": float(metrics["inference_seconds"]),
                }
            )

    if missing:
        preview = "\n".join(missing[:10])
        raise RuntimeError(f"Missing {len(missing)} runs. First missing paths:\n{preview}")

    summary_dir = root / "summary"
    summary_dir.mkdir(parents=True, exist_ok=True)
    output = summary_dir / "wo_joint_each_seed.csv"
    with output.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=tuple(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    for row in rows:
        print(
            f"seed={row['seed']:>3} dataset={row['dataset']:<14} "
            f"RMSE={row['rmse']:.9f} MAE={row['mae']:.9f}",
            flush=True,
        )
    print(f"COMPLETE: {len(rows)} runs; wrote {output}", flush=True)


if __name__ == "__main__":
    main()
