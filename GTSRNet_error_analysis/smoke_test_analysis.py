"""Small synthetic check for the analysis pipeline; no project data or GPU needed."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np

from analyze_mechanism_errors import phase_analysis, spatial_analysis


def main() -> None:
    rng = np.random.default_rng(123)
    truth = rng.normal(10.0, 2.0, size=(8, 12, 1, 3, 4))
    full = truth + rng.normal(0.0, 0.5, size=truth.shape)
    wo_s = truth + rng.normal(0.0, 0.9, size=truth.shape)
    no_phase = truth + rng.normal(0.0, 0.7, size=truth.shape)
    start = np.arange(8, dtype=np.int64) * 3
    hours = (start[:, None] + np.arange(12)[None, :]) % 24
    payload = {"config": {"crop": True, "rows": [40, 43], "cols": [40, 44]}}
    with tempfile.TemporaryDirectory(prefix="mechanism_analysis_") as temp:
        out = Path(temp)
        spatial_analysis(full, wo_s, truth, payload, "synthetic", 16, out, 80)
        phase_analysis(full, no_phase, truth, hours, "synthetic", 16, out, 80)
        required = (
            "spatial_grid_difference.png", "spatial_grid_difference.pdf",
            "spatial_grid_differences.csv", "spatial_top_bottom_grids.csv",
            "spatial_by_traffic_quartile.csv", "phase_hour_difference.png",
            "phase_hour_difference.pdf", "phase_hour_differences.csv",
        )
        missing = [name for name in required if not (out / name).is_file()]
        if missing:
            raise RuntimeError(f"Smoke test missing outputs: {missing}")
    print("PASS analysis smoke test")


if __name__ == "__main__":
    main()
