"""Visualize learned Channel, Phase, Channel+Phase, or Joint embeddings."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch


SHAPES = {
    "milan_sms": (20, 20), "milan_call": (20, 20), "milan_internet": (20, 20),
    "mobile_bj": (28, 24), "mobile_nj": (20, 28),
}


def pca2(matrix: np.ndarray) -> np.ndarray:
    matrix = matrix.astype(np.float64)
    matrix -= matrix.mean(axis=0, keepdims=True)
    _u, _s, vt = np.linalg.svd(matrix, full_matrices=False)
    return matrix @ vt[:2].T


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dataset", choices=tuple(SHAPES), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dpi", type=int, default=300)
    args = parser.parse_args()
    try:
        checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    except TypeError:
        checkpoint = torch.load(args.checkpoint, map_location="cpu")
    state = checkpoint.get("model_state", checkpoint)
    channel = state.get("spatial_branch.channel_embedding")
    phase = state.get("spatial_branch.phase_embedding.weight")
    joint = state.get("spatial_branch.joint_embedding.weight")
    if channel is None and phase is None and joint is None:
        raise ValueError("Checkpoint contains no reviewer embedding parameters (none condition?)")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    panels = int(channel is not None) + int(phase is not None) + (2 if joint is not None else 0)
    fig, axes = plt.subplots(1, panels, figsize=(5.0 * panels, 4.3), constrained_layout=True)
    axes = np.atleast_1d(axes); panel = 0
    height, width = SHAPES[args.dataset]
    if channel is not None:
        values = channel.numpy(); coords = pca2(values)
        colors = np.arange(len(values)) // width
        scatter = axes[panel].scatter(coords[:, 0], coords[:, 1], c=colors, cmap="viridis", s=14)
        axes[panel].set_title("Channel embedding PCA"); axes[panel].set_xlabel("PC1"); axes[panel].set_ylabel("PC2")
        fig.colorbar(scatter, ax=axes[panel], label="Grid row"); panel += 1
    if phase is not None:
        values = phase.numpy(); coords = pca2(values)
        axes[panel].plot(coords[:, 0], coords[:, 1], "o-", color="#C44E52")
        for hour, point in enumerate(coords): axes[panel].text(point[0], point[1], str(hour), fontsize=7)
        axes[panel].set_title("Phase embedding PCA"); axes[panel].set_xlabel("PC1"); axes[panel].set_ylabel("PC2"); panel += 1
    if joint is not None:
        values = joint.numpy().reshape(joint.shape[0], height * width, -1)
        phase_mean = values.mean(axis=1); coords = pca2(phase_mean)
        axes[panel].plot(coords[:, 0], coords[:, 1], "o-", color="#8172B3")
        for hour, point in enumerate(coords): axes[panel].text(point[0], point[1], str(hour), fontsize=7)
        axes[panel].set_title("Joint embedding phase PCA"); axes[panel].set_xlabel("PC1"); axes[panel].set_ylabel("PC2"); panel += 1
        norm_map = np.linalg.norm(values, axis=2)
        image = axes[panel].imshow(norm_map, aspect="auto", cmap="magma")
        axes[panel].set_title("Joint embedding norm"); axes[panel].set_xlabel("Grid-cell index"); axes[panel].set_ylabel("Hour phase")
        fig.colorbar(image, ax=axes[panel]); panel += 1
    fig.suptitle(f"Learned embedding representation — {args.dataset}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=args.dpi, bbox_inches="tight"); plt.close(fig)
    print(f"WROTE {args.output}")


if __name__ == "__main__":
    main()
