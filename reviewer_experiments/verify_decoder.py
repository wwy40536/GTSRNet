"""Fast structural sanity checks for the PMF/RMF decoder implementation.

Checks both output baselines (last value and seasonal) with identical weights.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from decoder_model import SegRNNGridDecoderAblation

TOL = 1e-5


def check(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)
    print("PASS:", message)


def max_abs(a: torch.Tensor, b: torch.Tensor) -> float:
    return float(torch.max(torch.abs(a - b)))


def main() -> None:
    torch.manual_seed(123)
    common = dict(Tin=24, Tout=24, H=4, W=5, d_model=32, seg_len=6, dropout=0.0)
    reference = SegRNNGridDecoderAblation(decoder_mode="pmf", **common).eval()
    state = reference.state_dict()
    x = torch.randn(2, 24, 1, 4, 5)
    x_flat = x.squeeze(2).reshape(2, 24, 20)
    counts = set()

    for baseline in ("last", "seasonal"):
        pmf = SegRNNGridDecoderAblation(decoder_mode="pmf", baseline_mode=baseline, **common).eval()
        rmf = SegRNNGridDecoderAblation(decoder_mode="rmf", baseline_mode=baseline, **common).eval()
        pmf.load_state_dict(state)
        rmf.load_state_dict(state)
        counts.update(sum(p.numel() for p in m.parameters()) for m in (pmf, rmf))

        with torch.no_grad():
            pmf_out, rmf_out = pmf(x), rmf(x)
            torch.manual_seed(999)
            pmf_noisy = pmf(x, feedback_noise_std=0.05)
            torch.manual_seed(999)
            rmf_noisy = rmf(x, feedback_noise_std=0.05)

        tag = f"[{baseline}]"
        check(max_abs(pmf_out[:, :6], rmf_out[:, :6]) < TOL, f"{tag} same-weight PMF/RMF first segment identical")
        check(max_abs(pmf_out, pmf_noisy) < TOL, f"{tag} feedback noise does not change PMF")
        check(max_abs(rmf_out[:, :6], rmf_noisy[:, :6]) < TOL, f"{tag} RMF noise does not change steps 1-6")
        check(max_abs(rmf_out[:, 6:], rmf_noisy[:, 6:]) > TOL, f"{tag} RMF noise changes steps 7-24")

        # With a zero output layer the residual is zero, so the forecast must be
        # exactly the baseline: persistence or seasonal naive.
        for model in (pmf, rmf):
            zeroed = SegRNNGridDecoderAblation(
                decoder_mode=model.decoder_mode, baseline_mode=baseline, **common
            ).eval()
            zeroed.load_state_dict(state)
            with torch.no_grad():
                zeroed.predict[1].weight.zero_()
                zeroed.predict[1].bias.zero_()
                out = zeroed(x)
            expected = x_flat[:, -1:, :].expand(-1, 24, -1) if baseline == "last" else x_flat[:, 0:24, :]
            check(max_abs(out, expected) < TOL,
                  f"{tag} {model.decoder_mode.upper()} with zero residual equals the {baseline} baseline")

    # Last-baseline RMF must behave exactly like the original implementation:
    # feedback is the raw residual prediction (offset is zero).
    rmf_last = SegRNNGridDecoderAblation(decoder_mode="rmf", baseline_mode="last", **common).eval()
    rmf_last.load_state_dict(state)
    with torch.no_grad():
        seq_last = x_flat[:, -1:, :]
        base = rmf_last._build_baseline(x_flat, seq_last)
        offset = rmf_last._to_segments(base - seq_last, 2)
    check(float(offset.abs().max()) == 0.0, "[last] RMF feedback offset is exactly zero (backward compatible)")
    check(len(counts) == 1, f"all PMF/RMF x baseline models have equal parameter counts: {counts.pop()}")


if __name__ == "__main__":
    main()
