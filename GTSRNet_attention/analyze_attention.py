"""Extract GTSRNet spatial attention and run matched-budget perturbation tests."""

from __future__ import annotations

import argparse
import csv
import importlib
import json
import math
import sys
from pathlib import Path
from typing import Dict, List, Mapping, Sequence, Tuple

import numpy as np


DATASETS = ("milan_sms", "milan_call", "milan_internet", "mobile_bj", "mobile_nj")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Attention visualization, degeneration diagnostics, and causal occlusion."
    )
    parser.add_argument("--project-root", type=Path, required=True)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--checkpoint", type=Path)
    source.add_argument("--checkpoint-root", type=Path)
    parser.add_argument("--dataset", choices=DATASETS, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--max-test-samples", type=int, default=32, help="0 uses all test windows.")
    parser.add_argument("--fractions", nargs="+", type=float, default=(0.05, 0.10, 0.20))
    parser.add_argument("--random-repeats", type=int, default=5)
    parser.add_argument("--top-mass-fraction", type=float, default=0.10)
    parser.add_argument("--analysis-seed", type=int, default=2026)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--dpi", type=int, default=300)
    return parser


def resolve_under(root: Path, value: Path) -> Path:
    value = value.expanduser()
    return value.resolve() if value.is_absolute() else (root / value).resolve()


def write_csv(path: Path, rows: Sequence[Mapping[str, object]], fields: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    print(f"WROTE {path}", flush=True)


def load_training_api(project_root: Path):
    training_dir = project_root / "GTSRNet_model"
    training_file = training_dir / "train_ablation.py"
    if not training_file.is_file():
        raise FileNotFoundError(f"Required original training entry not found: {training_file}")
    for path in (str(training_dir), str(project_root)):
        if path not in sys.path:
            sys.path.insert(0, path)
    return importlib.import_module("train_ablation")


def torch_load(path: Path, map_location):
    import torch

    try:
        return torch.load(path, map_location=map_location, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=map_location)


def checkpoint_path(args: argparse.Namespace, project_root: Path) -> Path:
    if args.checkpoint is not None:
        path = resolve_under(project_root, args.checkpoint)
    else:
        root = resolve_under(project_root, args.checkpoint_root)
        path = root / args.dataset / "GTSRNet" / "best.pt"
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    return path


def choose_device(name: str, gpu: int):
    import torch

    if name == "cpu":
        return torch.device("cpu")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    if gpu < 0 or gpu >= torch.cuda.device_count():
        raise ValueError(f"Invalid GPU {gpu}; found {torch.cuda.device_count()} GPU(s)")
    return torch.device(f"cuda:{gpu}")


def reconstruct_data_args(training_api, config, project_root, dataset, batch_size, num_workers):
    defaults = vars(training_api.build_parser().parse_args([])).copy()
    defaults.update(dict(config))
    defaults.update(
        project_root=project_root,
        dataset=dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        device="cpu",
        gpu=0,
    )
    for key in ("project_root", "data", "feature_path", "output_dir"):
        value = defaults.get(key)
        if value is not None and not isinstance(value, Path):
            defaults[key] = Path(str(value))
    return argparse.Namespace(**defaults)


def build_model(config, height: int, width: int, device):
    from attention_model import GTSRNet

    return GTSRNet(
        t_in=int(config.get("close_size", 24)),
        t_out=int(config.get("predict_size", 12)),
        height=height,
        width=width,
        variant="GTSRNet",
        d_model=int(config.get("d_model", 128)),
        seg_len=int(config.get("seg_len", 6)),
        dropout=float(config.get("dropout", 0.1)),
        n_heads=int(config.get("n_heads", 4)),
        e_layers=int(config.get("e_layers", 2)),
        d_ff=int(config.get("d_ff", 256)),
        cycle_len=int(config.get("cycle_len", 24)),
    ).to(device)


def subset_parts(loader):
    subset = loader.dataset
    if not hasattr(subset, "indices") or not hasattr(subset, "dataset"):
        raise TypeError("Expected torch.utils.data.Subset from the original loader")
    return subset.dataset, np.asarray(subset.indices, dtype=np.int64)


def select_events(dataset, train_indices: np.ndarray, test_indices: np.ndarray):
    x = dataset.x
    phases = np.asarray(dataset.phase, dtype=np.int64)
    train_score = np.asarray(x[train_indices, -1, 0]).mean(axis=(1, 2))
    test_score = np.asarray(x[test_indices, -1, 0]).mean(axis=(1, 2))
    q05, q95 = np.quantile(train_score, (0.05, 0.95))
    cycle = max(int(phases.max()) + 1, 24)
    phase_mean = np.full(cycle, float(train_score.mean()), dtype=np.float64)
    phase_std = np.full(cycle, max(float(train_score.std()), 1e-8), dtype=np.float64)
    train_phase = phases[train_indices]
    for phase in np.unique(train_phase):
        values = train_score[train_phase == phase]
        phase_mean[phase] = float(values.mean())
        phase_std[phase] = max(float(values.std()), 1e-8)
    test_phase = phases[test_indices]
    zscore = np.abs((test_score - phase_mean[test_phase]) / phase_std[test_phase])
    if len(test_indices) < 3:
        raise ValueError("At least three test windows are required for distinct events")

    # Keep the three representative cases distinct. Without this guard, a
    # short or nearly constant test period can silently overwrite one event in
    # event_by_index and produce fewer than the promised three figures.
    used = set()

    def first_unused(order):
        for position in order:
            candidate = int(position)
            if candidate not in used:
                used.add(candidate)
                return candidate
        raise RuntimeError("Could not select three distinct event windows")

    peak_pos = first_unused(np.argsort(np.abs(test_score - q95)))
    trough_pos = first_unused(np.argsort(np.abs(test_score - q05)))
    anomaly_pos = first_unused(np.argsort(-zscore))
    events = {
        "peak": int(test_indices[peak_pos]),
        "trough": int(test_indices[trough_pos]),
        "anomaly": int(test_indices[anomaly_pos]),
    }
    details = {
        "train_q05": float(q05),
        "train_q95": float(q95),
        "peak_score": float(test_score[peak_pos]),
        "trough_score": float(test_score[trough_pos]),
        "anomaly_score": float(test_score[anomaly_pos]),
        "anomaly_abs_z": float(zscore[anomaly_pos]),
    }
    return events, details


def select_analysis_indices(test_indices: np.ndarray, events: Mapping[str, int], maximum: int):
    if maximum <= 0 or maximum >= len(test_indices):
        selected = list(map(int, test_indices))
    else:
        positions = np.linspace(0, len(test_indices) - 1, maximum, dtype=np.int64)
        selected = [int(test_indices[position]) for position in positions]
    selected.extend(int(value) for value in events.values())
    return np.asarray(sorted(set(selected)), dtype=np.int64)


def phase_conditioned_replacement(dataset, train_indices: np.ndarray, cycle_len: int):
    x = dataset.x
    phases = np.asarray(dataset.phase, dtype=np.int64)
    t_in = int(x.shape[1])
    cells = int(x.shape[-2] * x.shape[-1])
    sums = np.zeros((cycle_len, cells), dtype=np.float64)
    counts = np.zeros(cycle_len, dtype=np.int64)
    for offset in range(t_in):
        historical_phase = (phases[train_indices] - t_in + offset) % cycle_len
        values = np.asarray(x[train_indices, offset, 0]).reshape(len(train_indices), cells)
        for phase in np.unique(historical_phase):
            mask = historical_phase == phase
            sums[phase] += values[mask].sum(axis=0)
            counts[phase] += int(mask.sum())
    global_mean = np.asarray(x[train_indices, :, 0]).mean(axis=(0, 1)).reshape(cells)
    replacement = np.empty_like(sums, dtype=np.float32)
    for phase in range(cycle_len):
        replacement[phase] = sums[phase] / counts[phase] if counts[phase] else global_mean
    return replacement


def grid_geometry(height: int, width: int):
    coordinates = np.stack(np.unravel_index(np.arange(height * width), (height, width)), axis=1)
    delta = coordinates[:, None, :] - coordinates[None, :, :]
    distance = np.sqrt(np.square(delta).sum(axis=-1)).astype(np.float32)
    local = (np.abs(delta).max(axis=-1) <= 1).astype(np.float32)
    return coordinates, distance, local


def attention_rollout(attentions):
    import torch

    batch, _, cells, _ = attentions[0].shape
    identity = torch.eye(cells, device=attentions[0].device).unsqueeze(0).expand(batch, -1, -1)
    rollout = identity
    for attention in attentions:
        matrix = attention.mean(dim=1)
        matrix = matrix + identity
        matrix = matrix / matrix.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        rollout = torch.bmm(matrix, rollout)
    return rollout


def attention_statistics(attentions, sample_indices, event_by_index, distance, local, top_fraction):
    rows: List[Dict[str, object]] = []
    eps = 1e-12
    cells = attentions[0].shape[-1]
    top_k = max(1, int(round(cells * top_fraction)))
    log_cells = math.log(cells)
    distance_t = None
    local_t = None
    for layer_index, tensor in enumerate(attentions, start=1):
        if distance_t is None:
            import torch

            distance_t = torch.as_tensor(distance, device=tensor.device, dtype=tensor.dtype)
            local_t = torch.as_tensor(local, device=tensor.device, dtype=tensor.dtype)
        entropy = -(tensor.clamp_min(eps) * tensor.clamp_min(eps).log()).sum(dim=-1) / log_cells
        top_mass = tensor.topk(top_k, dim=-1).values.sum(dim=-1)
        diagonal = tensor.diagonal(dim1=-2, dim2=-1).mean(dim=-1)
        local_mass = (tensor * local_t).sum(dim=-1).mean(dim=-1)
        mean_distance = (tensor * distance_t).sum(dim=-1).mean(dim=-1)
        for batch_index, sample_index in enumerate(sample_indices):
            head_js = []
            for first in range(tensor.shape[1]):
                for second in range(first + 1, tensor.shape[1]):
                    p = tensor[batch_index, first].clamp_min(eps)
                    q = tensor[batch_index, second].clamp_min(eps)
                    midpoint = 0.5 * (p + q)
                    js = 0.5 * (
                        (p * (p / midpoint).log()).sum(dim=-1)
                        + (q * (q / midpoint).log()).sum(dim=-1)
                    )
                    head_js.append(float(js.mean().item()))
            rows.append(
                {
                    "sample_index": int(sample_index),
                    "event": event_by_index.get(int(sample_index), "sampled_test"),
                    "layer": layer_index,
                    "normalized_entropy": float(entropy[batch_index].mean().item()),
                    "top_mass_fraction": top_fraction,
                    "top_mass": float(top_mass[batch_index].mean().item()),
                    "self_mass": float(diagonal[batch_index].mean().item()),
                    "local_3x3_mass": float(local_mass[batch_index].mean().item()),
                    "mean_grid_distance": float(mean_distance[batch_index].mean().item()),
                    "mean_pairwise_head_js": float(np.mean(head_js)) if head_js else 0.0,
                }
            )
    return rows


def prediction_errors(prediction, target, normalizer):
    pred = normalizer.inverse_transform(prediction.detach().cpu().numpy())
    truth = normalizer.inverse_transform(target.detach().cpu().numpy())
    error = pred.astype(np.float64) - truth.astype(np.float64)
    axes = tuple(range(1, error.ndim))
    rmse = np.sqrt(np.mean(np.square(error), axis=axes))
    mae = np.mean(np.abs(error), axis=axes)
    return rmse, mae


def replacement_for_batch(x, phase, replacement: np.ndarray, cycle_len: int):
    batch, t_in, _, height, width = x.shape
    values = np.empty((batch, t_in, height * width), dtype=np.float32)
    phases = phase.detach().cpu().numpy().astype(np.int64)
    for item in range(batch):
        for offset in range(t_in):
            historical_phase = (phases[item] - t_in + offset) % cycle_len
            values[item, offset] = replacement[historical_phase]
    return values


def mask_cells(x, cell_indices: np.ndarray, replacements: np.ndarray):
    import torch

    modified = x.clone()
    flat = modified[:, :, 0].reshape(modified.shape[0], modified.shape[1], -1)
    for item, indices in enumerate(cell_indices):
        index_tensor = torch.as_tensor(indices, device=x.device, dtype=torch.long)
        source = torch.as_tensor(
            replacements[item][:, indices], device=x.device, dtype=x.dtype
        )
        flat[item][:, index_tensor] = source
    return modified


def shuffle_cells(x, cell_indices: np.ndarray, rng: np.random.Generator):
    import torch

    modified = x.clone()
    flat = modified[:, :, 0].reshape(modified.shape[0], modified.shape[1], -1)
    for item, indices in enumerate(cell_indices):
        permutation = np.asarray(indices).copy()
        rng.shuffle(permutation)
        if len(permutation) > 1 and np.array_equal(permutation, indices):
            permutation = np.roll(permutation, 1)
        source_tensor = torch.as_tensor(permutation, device=x.device, dtype=torch.long)
        target_tensor = torch.as_tensor(indices, device=x.device, dtype=torch.long)
        flat[item][:, target_tensor] = flat[item][:, source_tensor].clone()
    return modified


def index_sets(importance: np.ndarray, fraction: float, distance: np.ndarray, rng):
    batch, cells = importance.shape
    count = max(2, min(cells, int(round(cells * fraction))))
    order = np.argsort(importance, axis=1)
    high = order[:, -count:]
    low = order[:, :count]
    local = np.empty_like(high)
    for item in range(batch):
        center = int(order[item, -1])
        local[item] = np.argsort(distance[center])[:count]
    random_indices = np.stack(
        [rng.choice(cells, size=count, replace=False) for _ in range(batch)], axis=0
    )
    return high, low, local, random_indices


def append_metric_rows(
    rows,
    dataset,
    training_seed,
    sample_indices,
    event_by_index,
    intervention,
    fraction,
    repeat,
    full_prediction,
    spatial_prediction,
    target,
    normalizer,
    baseline,
):
    for branch, prediction in (("GTSRNet", full_prediction), ("spatial_branch", spatial_prediction)):
        rmse, mae = prediction_errors(prediction, target, normalizer)
        for item, sample_index in enumerate(sample_indices):
            base_rmse, base_mae = baseline[(int(sample_index), branch)]
            rows.append(
                {
                    "dataset": dataset,
                    "training_seed": training_seed,
                    "sample_index": int(sample_index),
                    "event": event_by_index.get(int(sample_index), "sampled_test"),
                    "intervention": intervention,
                    "fraction": fraction,
                    "repeat": repeat,
                    "branch": branch,
                    "rmse": float(rmse[item]),
                    "mae": float(mae[item]),
                    "delta_rmse": float(rmse[item] - base_rmse),
                    "delta_mae": float(mae[item] - base_mae),
                }
            )


def plot_event(output_dir, name, sample_index, x, target, prediction, attentions, normalizer, height, width, dpi):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    event_dir = output_dir / "events" / name
    event_dir.mkdir(parents=True, exist_ok=True)
    x_raw = normalizer.inverse_transform(x.detach().cpu().numpy())[0]
    target_raw = normalizer.inverse_transform(target.detach().cpu().numpy())[0]
    prediction_raw = normalizer.inverse_transform(prediction.detach().cpu().numpy())[0]
    query = int(np.argmax(x_raw[-1, 0]))
    query_row, query_col = divmod(query, width)
    arrays = {f"layer_{layer + 1}": value[0].detach().cpu().numpy() for layer, value in enumerate(attentions)}
    np.savez_compressed(
        event_dir / "attention_raw.npz",
        sample_index=np.asarray(sample_index),
        query_cell=np.asarray(query),
        input=x_raw,
        target=target_raw,
        prediction=prediction_raw,
        **arrays,
    )

    layer_count = len(attentions)
    head_count = attentions[0].shape[1]
    fig, axes = plt.subplots(layer_count, head_count, figsize=(3 * head_count, 2.8 * layer_count), squeeze=False)
    values = [attention[0, :, query].detach().cpu().numpy() for attention in attentions]
    maximum = max(float(item.max()) for item in values)
    for layer in range(layer_count):
        for head in range(head_count):
            axis = axes[layer, head]
            image = axis.imshow(values[layer][head].reshape(height, width), cmap="magma", vmin=0, vmax=maximum)
            axis.scatter([query_col], [query_row], marker="x", color="cyan", s=30)
            axis.set_title(f"Layer {layer + 1}, head {head + 1}")
            axis.set_xticks([])
            axis.set_yticks([])
    fig.colorbar(image, ax=axes, shrink=0.75, label="Attention weight")
    fig.suptitle(f"{name}: query cell ({query_row},{query_col}), sample {sample_index}")
    fig.savefig(event_dir / "attention_heads.png", dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    # The spatial maps above show one interpretable query cell. Also render
    # every complete [query,key] matrix because the reviewer explicitly asks
    # for the attention matrices of every layer and head.
    matrix_maximum = max(float(attention[0].max().item()) for attention in attentions)
    fig, axes = plt.subplots(
        layer_count,
        head_count,
        figsize=(3.2 * head_count, 3.0 * layer_count),
        squeeze=False,
        constrained_layout=True,
    )
    for layer in range(layer_count):
        matrix_values = attentions[layer][0].detach().cpu().numpy()
        for head in range(head_count):
            axis = axes[layer, head]
            matrix_image = axis.imshow(
                matrix_values[head], cmap="magma", vmin=0, vmax=matrix_maximum,
                interpolation="nearest", aspect="auto"
            )
            axis.set_title(f"Layer {layer + 1}, head {head + 1}")
            axis.set_xlabel("Key cell index")
            axis.set_ylabel("Query cell index")
    fig.colorbar(matrix_image, ax=axes, shrink=0.75, label="Attention weight")
    fig.suptitle(f"{name}: complete attention matrices, sample {sample_index}")
    fig.savefig(
        event_dir / "attention_matrices.png", dpi=dpi,
        bbox_inches="tight", facecolor="white"
    )
    plt.close(fig)

    rollout = attention_rollout(attentions)[0].detach().cpu().numpy()
    importance = rollout.mean(axis=0)
    top_k = max(1, int(round(len(importance) * 0.10)))
    order = np.argsort(importance)
    low_indices = order[:top_k]
    high_indices = order[-top_k:]
    importance_map = importance.reshape(height, width)
    fig, axis = plt.subplots(figsize=(5.2, 4.4), constrained_layout=True)
    rollout_image = axis.imshow(importance_map, cmap="viridis")
    high_rows, high_cols = np.unravel_index(high_indices, (height, width))
    low_rows, low_cols = np.unravel_index(low_indices, (height, width))
    axis.scatter(high_cols, high_rows, facecolors="none", edgecolors="red", s=24, label="Top 10%")
    axis.scatter(low_cols, low_rows, marker="x", color="cyan", s=18, label="Bottom 10%")
    axis.set_title(f"{name}: attention-rollout cell importance")
    axis.set_xticks([])
    axis.set_yticks([])
    axis.legend(frameon=True, fontsize=8)
    fig.colorbar(rollout_image, ax=axis, shrink=0.8, label="Mean received rollout attention")
    fig.savefig(
        event_dir / "attention_rollout_importance.png", dpi=dpi,
        bbox_inches="tight", facecolor="white"
    )
    plt.close(fig)

    fig, axes = plt.subplots(1, 4, figsize=(13, 3.1), constrained_layout=True)
    panels = (
        (x_raw[-1, 0], "Last observed"),
        (target_raw[0, 0], "Target t+1"),
        (prediction_raw[0, 0], "Prediction t+1"),
        (np.abs(prediction_raw[0, 0] - target_raw[0, 0]), "Absolute error"),
    )
    for axis, (values, title) in zip(axes, panels):
        image = axis.imshow(values, cmap="viridis")
        axis.set_title(title)
        axis.set_xticks([])
        axis.set_yticks([])
        fig.colorbar(image, ax=axis, shrink=0.75)
    fig.savefig(event_dir / "traffic_context.png", dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> None:
    args = build_parser().parse_args()
    if args.batch_size <= 0 or args.random_repeats <= 0 or args.max_test_samples < 0:
        raise ValueError("batch size/repeats must be positive and max samples non-negative")
    if not all(0 < value <= 1 for value in args.fractions):
        raise ValueError("Every perturbation fraction must be in (0,1]")
    if not 0 < args.top_mass_fraction <= 1:
        raise ValueError("--top-mass-fraction must be in (0,1]")

    import torch

    project_root = args.project_root.expanduser().resolve()
    output_dir = resolve_under(project_root, args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    training_api = load_training_api(project_root)
    device = choose_device(args.device, args.gpu)
    checkpoint = checkpoint_path(args, project_root)
    saved = torch_load(checkpoint, "cpu")
    config = dict(saved.get("args", {}))
    checkpoint_model = saved.get("variant", config.get("model", config.get("variant")))
    if checkpoint_model != "GTSRNet":
        raise ValueError(f"Expected a GTSRNet checkpoint, found {checkpoint_model!r}")
    if config.get("dataset", args.dataset) != args.dataset:
        raise ValueError(
            f"Checkpoint dataset is {config.get('dataset')!r}, requested {args.dataset!r}"
        )
    data_args = reconstruct_data_args(
        training_api, config, project_root, args.dataset, args.batch_size, args.num_workers
    )
    training_api.set_seed(int(config.get("seed", 42)))
    train_loader, _val_loader, test_loader, normalizer, height, width = training_api.build_loaders(
        data_args, device
    )
    dataset, train_indices = subset_parts(train_loader)
    test_dataset, test_indices = subset_parts(test_loader)
    if test_dataset is not dataset:
        raise RuntimeError("Training and test subsets do not share the same dataset")

    model = build_model(config, height, width, device)
    model.load_state_dict(saved["model_state"], strict=True)
    model.eval()
    training_seed = int(config.get("seed", 42))
    cycle_len = int(config.get("cycle_len", 24))
    events, event_details = select_events(dataset, train_indices, test_indices)
    event_by_index = {sample_index: name for name, sample_index in events.items()}
    analysis_indices = select_analysis_indices(test_indices, events, args.max_test_samples)
    replacement = phase_conditioned_replacement(dataset, train_indices, cycle_len)
    _coordinates, distance, local = grid_geometry(height, width)
    rng = np.random.default_rng(args.analysis_seed + training_seed)

    attention_rows: List[Dict[str, object]] = []
    perturbation_rows: List[Dict[str, object]] = []
    plotted = set()
    print(
        f"dataset={args.dataset}, checkpoint={checkpoint}, seed={training_seed}, "
        f"test={len(test_indices)}, analyzed={len(analysis_indices)}, grid={height}x{width}",
        flush=True,
    )

    with torch.no_grad():
        for start in range(0, len(analysis_indices), args.batch_size):
            batch_indices = analysis_indices[start : start + args.batch_size]
            x = torch.as_tensor(dataset.x[batch_indices], device=device, dtype=torch.float32)
            phase = torch.as_tensor(dataset.phase[batch_indices], device=device, dtype=torch.long)
            target = torch.as_tensor(dataset.y[batch_indices], device=device, dtype=torch.float32)
            prediction, branches, attentions = model(
                x, phase, return_branches=True, return_attention=True
            )
            spatial = branches["spatial"]
            if spatial is None:
                raise RuntimeError("GTSRNet did not return its spatial branch")

            baseline: Dict[Tuple[int, str], Tuple[float, float]] = {}
            for branch_name, branch_prediction in (("GTSRNet", prediction), ("spatial_branch", spatial)):
                rmse, mae = prediction_errors(branch_prediction, target, normalizer)
                for item, sample_index in enumerate(batch_indices):
                    baseline[(int(sample_index), branch_name)] = (float(rmse[item]), float(mae[item]))
            append_metric_rows(
                perturbation_rows, args.dataset, training_seed, batch_indices, event_by_index,
                "baseline", 0.0, 0, prediction, spatial, target, normalizer, baseline,
            )

            rows = attention_statistics(
                attentions, batch_indices, event_by_index, distance, local, args.top_mass_fraction
            )
            for row in rows:
                row.update(dataset=args.dataset, training_seed=training_seed)
            attention_rows.extend(rows)

            rollout = attention_rollout(attentions)
            importance = rollout.mean(dim=1).detach().cpu().numpy()
            batch_replacement = replacement_for_batch(x, phase, replacement, cycle_len)

            for item, sample_index in enumerate(batch_indices):
                event_name = event_by_index.get(int(sample_index))
                if event_name and event_name not in plotted:
                    event_attentions = [value[item : item + 1] for value in attentions]
                    plot_event(
                        output_dir, event_name, int(sample_index), x[item : item + 1],
                        target[item : item + 1], prediction[item : item + 1], event_attentions,
                        normalizer, height, width, args.dpi,
                    )
                    plotted.add(event_name)

            for fraction in args.fractions:
                high, low, local_indices, _ = index_sets(importance, fraction, distance, rng)
                for intervention, indices in (
                    ("mask_high_attention", high),
                    ("mask_low_attention", low),
                    ("mask_local_around_top", local_indices),
                ):
                    modified = mask_cells(x, indices, batch_replacement)
                    changed, changed_branches = model(modified, phase, return_branches=True)
                    append_metric_rows(
                        perturbation_rows, args.dataset, training_seed, batch_indices,
                        event_by_index, intervention, fraction, 0, changed,
                        changed_branches["spatial"], target, normalizer, baseline,
                    )
                for repeat in range(args.random_repeats):
                    _, _, _, random_indices = index_sets(importance, fraction, distance, rng)
                    modified = mask_cells(x, random_indices, batch_replacement)
                    changed, changed_branches = model(modified, phase, return_branches=True)
                    append_metric_rows(
                        perturbation_rows, args.dataset, training_seed, batch_indices,
                        event_by_index, "mask_random", fraction, repeat, changed,
                        changed_branches["spatial"], target, normalizer, baseline,
                    )
                    shuffled = shuffle_cells(x, high, rng)
                    changed, changed_branches = model(shuffled, phase, return_branches=True)
                    append_metric_rows(
                        perturbation_rows, args.dataset, training_seed, batch_indices,
                        event_by_index, "shuffle_high_positions", fraction, repeat, changed,
                        changed_branches["spatial"], target, normalizer, baseline,
                    )
            print(f"processed {min(start + args.batch_size, len(analysis_indices))}/{len(analysis_indices)}", flush=True)

    attention_fields = (
        "dataset", "training_seed", "sample_index", "event", "layer",
        "normalized_entropy", "top_mass_fraction", "top_mass", "self_mass",
        "local_3x3_mass", "mean_grid_distance", "mean_pairwise_head_js",
    )
    perturbation_fields = (
        "dataset", "training_seed", "sample_index", "event", "intervention",
        "fraction", "repeat", "branch", "rmse", "mae", "delta_rmse", "delta_mae",
    )
    write_csv(output_dir / "attention_statistics.csv", attention_rows, attention_fields)
    write_csv(output_dir / "perturbation_results.csv", perturbation_rows, perturbation_fields)
    manifest = {
        "dataset": args.dataset,
        "training_seed": training_seed,
        "checkpoint": str(checkpoint),
        "height": height,
        "width": width,
        "test_windows": len(test_indices),
        "analyzed_windows": len(analysis_indices),
        "events": events,
        "event_selection": event_details,
        "fractions": list(args.fractions),
        "random_repeats": args.random_repeats,
        "analysis_seed": args.analysis_seed,
        "replacement": "training-only phase-conditioned per-cell mean",
        "checkpoint_config": config,
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )
    print(f"WROTE {output_dir / 'manifest.json'}", flush=True)


if __name__ == "__main__":
    main()
