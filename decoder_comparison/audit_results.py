"""Fail-fast audit for completeness, finite metrics, and perturbation behavior."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def finite(value, label: str) -> None:
    if not math.isfinite(float(value)):
        raise RuntimeError(f"Non-finite metric: {label}={value}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--decoder-results", type=Path, default=Path("results_decoder_reviewer"))
    parser.add_argument("--embedding-results", type=Path, default=Path("results_embedding_reviewer"))
    parser.add_argument("--decoder-datasets", nargs="+", default=("milan_call", "milan_internet"))
    parser.add_argument("--horizons", nargs="+", type=int, default=(12, 24))
    parser.add_argument("--decoder-seeds", nargs="+", type=int, default=(45, 46, 47, 48, 49))
    parser.add_argument("--embedding-datasets", nargs="+", default=("milan_sms", "milan_call", "milan_internet", "mobile_bj", "mobile_nj"))
    parser.add_argument("--embedding-seeds", nargs="+", type=int, default=(16, 26, 36))
    parser.add_argument("--noise-levels", nargs="+", type=float, default=(0.0, 0.01, 0.03, 0.05, 0.10))
    parser.add_argument(
        "--decoder-baseline", choices=("last", "seasonal"), default=None,
        help="If given, require every decoder run to use this output baseline.",
    )
    parser.add_argument(
        "--skip-decoder",
        action="store_true",
        help="Skip decoder-result checks and audit only embedding results.",
    )
    parser.add_argument(
        "--skip-embedding",
        action="store_true",
        help="Skip embedding-result checks and audit only decoder results.",
    )
    args = parser.parse_args()
    if args.skip_decoder and args.skip_embedding:
        raise ValueError("--skip-decoder and --skip-embedding cannot both be used")
    decoder_root = args.decoder_results.expanduser().resolve()
    embedding_root = args.embedding_results.expanduser().resolve()
    decoder_count = 0
    if not args.skip_decoder:
        for dataset in args.decoder_datasets:
            for horizon in args.horizons:
                for decoder in ("pmf", "rmf"):
                    for seed in args.decoder_seeds:
                        run = decoder_root / dataset / decoder / f"H{horizon}" / f"seed_{seed}"
                        summary_path = run / "summary.json"; noise_path = run / "noise_sweep.json"
                        if not summary_path.is_file() or not noise_path.is_file():
                            raise FileNotFoundError(f"Incomplete decoder run: {run}")
                        summary = json.loads(summary_path.read_text(encoding="utf-8"))
                        config = summary["config"]
                        if config.get("split_mode") != "chronological" or config.get("milan_loader") != "clean":
                            raise RuntimeError(f"Unsafe split/loader in {summary_path}")
                        if args.decoder_baseline and config.get("baseline", "last") != args.decoder_baseline:
                            raise RuntimeError(f"Baseline mismatch in {summary_path}: {config.get('baseline', 'last')}")
                        for metric in ("rmse", "mae", "first_rmse", "last_rmse"):
                            finite(summary["metrics"][metric], f"{run}/{metric}")
                        noise = json.loads(noise_path.read_text(encoding="utf-8"))
                        levels = sorted(float(item["noise_std"]) for item in noise)
                        if levels != sorted(set(args.noise_levels)):
                            raise RuntimeError(f"Noise levels mismatch in {noise_path}: {levels}")
                        for item in noise:
                            level = float(item["noise_std"])
                            finite(item["delta_rmse"], f"{run}/noise={level}")
                            if decoder == "pmf" and abs(float(item["delta_rmse"])) > 1e-9:
                                raise RuntimeError(f"PMF negative control changed under feedback noise: {run}")
                            if decoder == "rmf" and level > 0:
                                if max(abs(float(v)) for v in item["delta_rmse_h"][:6]) > 1e-8:
                                    raise RuntimeError(f"RMF noise affected the first segment: {run}")
                                if max(abs(float(v)) for v in item["delta_rmse_h"][6:]) <= 1e-8:
                                    raise RuntimeError(f"RMF noise failed to affect later segments: {run}")
                        decoder_count += 1
    embeddings = ("none", "channel", "phase", "channel_phase", "joint")
    embedding_count = 0
    if not args.skip_embedding:
        for dataset in args.embedding_datasets:
            for embedding in embeddings:
                for seed in args.embedding_seeds:
                    path = embedding_root / dataset / embedding / f"seed_{seed}" / "metrics.json"
                    if not path.is_file(): raise FileNotFoundError(f"Missing embedding run: {path}")
                    payload = json.loads(path.read_text(encoding="utf-8")); config = payload["config"]
                    if config.get("split_mode") != "chronological" or config.get("milan_loader") != "clean":
                        raise RuntimeError(f"Unsafe split/loader in {path}")
                    for metric in ("rmse", "mae"):
                        finite(payload["metrics"][metric], f"{path}/{metric}")
                    embedding_count += 1
    print(f"AUDIT PASS: decoder_runs={decoder_count}, embedding_runs={embedding_count}")


if __name__ == "__main__":
    main()
