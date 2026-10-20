"""Verify that attention extraction leaves GTSRNet predictions unchanged."""

from __future__ import annotations

import argparse
import importlib
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--gpu", type=int, default=0)
    args = parser.parse_args()
    root = args.project_root.expanduser().resolve()
    training_dir = root / "GTSRNet_model"
    # Put the project ahead of site-packages. Some environments install an
    # unrelated top-level package named ``utils``; appending these paths would
    # let that package shadow this project's ``utils.layers`` package.
    priority_paths = (str(training_dir), str(root))
    sys.path[:] = list(priority_paths) + [
        path for path in sys.path if path not in priority_paths
    ]
    importlib.invalidate_caches()

    import torch
    from attention_model import GTSRNet as AnalysisModel

    original_module = importlib.import_module("gtsrnet_model")
    OriginalModel = original_module.GTSRNet
    if args.device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA unavailable")
        device = torch.device(f"cuda:{args.gpu}")
    else:
        device = torch.device("cpu")
    options = dict(
        t_in=12, t_out=6, height=2, width=2, variant="GTSRNet",
        d_model=16, seg_len=3, dropout=0.0, n_heads=4, e_layers=2,
        d_ff=32, cycle_len=24,
    )
    torch.manual_seed(123)
    original = OriginalModel(**options).to(device).eval()
    analysis = AnalysisModel(**options).to(device).eval()
    if set(original.state_dict()) != set(analysis.state_dict()):
        raise RuntimeError("State-dict keys differ; checkpoint compatibility failed")
    analysis.load_state_dict(original.state_dict(), strict=True)
    x = torch.randn(2, 12, 1, 2, 2, device=device)
    phase = torch.tensor([3, 17], device=device)
    with torch.no_grad():
        reference = original(x, phase)
        ordinary = analysis(x, phase)
        extracted, branches, attentions = analysis(
            x, phase, return_branches=True, return_attention=True
        )
    if not torch.allclose(reference, ordinary, atol=1e-7, rtol=1e-6):
        raise RuntimeError("Ordinary analysis-model output differs from original")
    if not torch.allclose(reference, extracted, atol=1e-7, rtol=1e-6):
        raise RuntimeError("Returning attention changed predictions")
    expected = (2, 4, 4, 4)
    if len(attentions) != 2 or any(tuple(value.shape) != expected for value in attentions):
        raise RuntimeError(f"Unexpected attention shapes: {[tuple(v.shape) for v in attentions]}")
    if branches["spatial"] is None or branches["temporal"] is None:
        raise RuntimeError("Branch outputs missing")
    print("PASS: state dict is compatible and attention extraction preserves predictions")
    print("attention shapes:", [tuple(value.shape) for value in attentions])


if __name__ == "__main__":
    main()
