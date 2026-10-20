"""Profile parameter counts, CUDA memory, and latency on one fixed GPU setup."""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path
from typing import Callable, Dict, List

import torch


SCRIPT_DIR = Path(__file__).resolve().parent


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=SCRIPT_DIR.parent)
    parser.add_argument("--output", type=Path, default=Path("results_reviewer_profile/complexity_profile.csv"))
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--input-len", type=int, default=24)
    parser.add_argument("--horizons", nargs="+", type=int, default=(12, 24))
    parser.add_argument("--height", type=int, default=20)
    parser.add_argument("--width", type=int, default=20)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--seg-len", type=int, default=6)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=100)
    args = parser.parse_args()
    root = args.project_root.expanduser().resolve()
    for path in (SCRIPT_DIR, root):
        if str(path) not in sys.path: sys.path.insert(0, str(path))
    from decoder_model import SegRNNGridDecoderAblation
    from embedding_ablation_model import EMBEDDING_VARIANTS, Trainduobu1Ablation
    if not torch.cuda.is_available(): raise RuntimeError("CUDA is required for comparable profiling")
    device = torch.device(f"cuda:{args.gpu}"); torch.cuda.set_device(device)
    if args.warmup < 0 or args.repeats <= 0: raise ValueError("Invalid warmup/repeats")

    def count(module) -> int:
        return sum(p.numel() for p in module.parameters() if p.requires_grad) if module is not None else 0

    @torch.inference_mode()
    def profile(model, forward: Callable[[], torch.Tensor]) -> Dict[str, float]:
        model.eval()
        for _ in range(args.warmup): forward()
        torch.cuda.synchronize(device); torch.cuda.reset_peak_memory_stats(device)
        baseline = torch.cuda.memory_allocated(device) / (1024.0 ** 2)
        started = time.perf_counter()
        for _ in range(args.repeats): forward()
        torch.cuda.synchronize(device)
        elapsed = (time.perf_counter() - started) * 1000.0 / args.repeats
        peak = torch.cuda.max_memory_allocated(device) / (1024.0 ** 2)
        return {"latency_ms_per_batch": elapsed, "peak_allocated_mb": peak, "incremental_peak_mb": max(0.0, peak - baseline)}

    rows: List[Dict[str, object]] = []
    for horizon in args.horizons:
        x = torch.randn(args.batch_size, args.input_len, 1, args.height, args.width, device=device)
        calendar = torch.zeros(args.batch_size, 2, dtype=torch.long, device=device)
        for embedding in EMBEDDING_VARIANTS:
            model = Trainduobu1Ablation(
                t_in=args.input_len, t_out=horizon, height=args.height, width=args.width,
                variant="full_v2", d_model=args.d_model, seg_len=args.seg_len,
                dropout=0.0, n_heads=4, e_layers=2, d_ff=256,
                embedding_variant=embedding,
            ).to(device)
            measured = profile(model, lambda m=model: m(x, calendar))
            rows.append(
                {
                    "family": "full_v2_embedding", "condition": embedding, "horizon": horizon,
                    "total_parameters": count(model), "spatial_parameters": count(model.spatial_branch),
                    "temporal_parameters": count(model.temporal_branch), "fusion_parameters": count(model.fusion),
                    "embedding_parameters": model.embedding_parameter_count(),
                    "total_work": "O(B*L*(N^2*d+N*d*d_ff) + B*N*(n+m)*d^2 + B*N*T*d)",
                    "future_serial_depth": "O(1) parallel future segments",
                    **measured,
                }
            )
            del model; torch.cuda.empty_cache()
        for decoder in ("pmf", "rmf"):
            model = SegRNNGridDecoderAblation(
                Tin=args.input_len, Tout=horizon, H=args.height, W=args.width,
                d_model=args.d_model, seg_len=args.seg_len, dropout=0.0,
                decoder_mode=decoder,
            ).to(device)
            measured = profile(model, lambda m=model: m(x))
            rows.append(
                {
                    "family": "temporal_decoder_only", "condition": decoder, "horizon": horizon,
                    "total_parameters": count(model), "spatial_parameters": 0,
                    "temporal_parameters": count(model), "fusion_parameters": 0,
                    "embedding_parameters": 0,
                    "total_work": "O(B*N*m*d^2)",
                    "future_serial_depth": "O(1)" if decoder == "pmf" else "O(m)",
                    **measured,
                }
            )
            del model; torch.cuda.empty_cache()
    output = args.output.expanduser()
    if not output.is_absolute(): output = (root / output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    fields = tuple(rows[0].keys())
    with output.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields); writer.writeheader(); writer.writerows(rows)
    print(f"WROTE {output}")


if __name__ == "__main__":
    main()
