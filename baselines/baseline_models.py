"""Baseline adapters for the GTSRNet five-dataset benchmark.

Every model consumes ``[B, Tin, 1, H, W]`` and returns
``[B, Tout, 1, H, W]``.  The compact adapters preserve the defining mechanism
of each cited method while making their I/O and evaluation protocol identical.
"""

from __future__ import annotations

import math
from typing import Dict, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F
from torch import nn

from gtsrnet_model import GTSRNet


MODEL_NAMES: Tuple[str, ...] = (
    "seasonal_ha",
    "gru_seq2seq",
    "mtgnn",
    "patchtst",
    "dlinear",
    "timemixer",
    "GTSRNet",
)

MODEL_DESCRIPTIONS: Dict[str, str] = {
    "seasonal_ha": "Hourly seasonal historical average (statistical)",
    "gru_seq2seq": "Shared per-cell GRU encoder-decoder (RNN)",
    "mtgnn": "MTGNN with learned adaptive graph (GNN)",
    "patchtst": "Channel-independent patched Transformer (PatchTST adapter)",
    "dlinear": "Trend/seasonal decomposition with direct linear heads (DLinear)",
    "timemixer": "Decomposable multiscale mixing model (TimeMixer adapter)",
    "GTSRNet": "Complete horizon-calibrated dual-branch model",
}


class SeasonalHistoricalAverage(nn.Module):
    """Per-cell mean conditioned on the target hour-of-day.

    ``fit`` only receives the training loader.  Repeated observations caused by
    sliding windows receive equal weight; no validation or test targets enter
    the table.
    """

    def __init__(self, t_out: int, height: int, width: int, cycle_len: int = 24) -> None:
        super().__init__()
        self.t_out = t_out
        self.height = height
        self.width = width
        self.cycle_len = cycle_len
        cells = height * width
        self.register_buffer("hourly_mean", torch.zeros(cycle_len, cells))
        self.register_buffer("global_mean", torch.zeros(cells))
        self.fitted = False

    def fit(self, loader) -> None:
        cells = self.height * self.width
        sums = torch.zeros(self.cycle_len, cells, dtype=torch.float64)
        counts = torch.zeros(self.cycle_len, dtype=torch.float64)
        global_sum = torch.zeros(cells, dtype=torch.float64)
        global_count = 0
        for _x, phase, target in loader:
            phase = phase.to(dtype=torch.long).reshape(-1)
            values = target.to(dtype=torch.float64).reshape(target.shape[0], self.t_out, cells)
            for horizon in range(self.t_out):
                target_phase = (phase + horizon) % self.cycle_len
                horizon_values = values[:, horizon]
                sums.index_add_(0, target_phase, horizon_values)
                counts.index_add_(0, target_phase, torch.ones_like(target_phase, dtype=torch.float64))
                global_sum += horizon_values.sum(dim=0)
                global_count += horizon_values.shape[0]
        if global_count == 0:
            raise RuntimeError("Cannot fit seasonal historical average on an empty loader")
        global_mean = global_sum / global_count
        table = global_mean.unsqueeze(0).repeat(self.cycle_len, 1)
        observed = counts > 0
        table[observed] = sums[observed] / counts[observed, None]
        self.hourly_mean.copy_(table.to(dtype=self.hourly_mean.dtype))
        self.global_mean.copy_(global_mean.to(dtype=self.global_mean.dtype))
        self.fitted = True

    def forward(self, x: torch.Tensor, phase: Optional[torch.Tensor] = None) -> torch.Tensor:
        if not self.fitted:
            raise RuntimeError("Call SeasonalHistoricalAverage.fit(train_loader) first")
        if phase is None:
            raise ValueError("phase is required by seasonal_ha")
        batch = x.shape[0]
        phase = phase.to(device=x.device, dtype=torch.long).reshape(batch)
        offsets = torch.arange(self.t_out, device=x.device)
        target_phase = (phase[:, None] + offsets[None, :]) % self.cycle_len
        prediction = self.hourly_mean[target_phase]
        return prediction.reshape(batch, self.t_out, 1, self.height, self.width)


class GRUSeq2Seq(nn.Module):
    """A shared recurrent encoder-decoder applied to every grid cell."""

    def __init__(self, t_out: int, height: int, width: int, hidden_dim: int = 64, layers: int = 1) -> None:
        super().__init__()
        self.t_out = t_out
        self.height = height
        self.width = width
        self.encoder = nn.GRU(1, hidden_dim, num_layers=layers, batch_first=True)
        self.decoder = nn.GRUCell(1, hidden_dim)
        self.projection = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor, phase: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch, steps, channels, height, width = x.shape
        if channels != 1 or (height, width) != (self.height, self.width):
            raise ValueError(f"Unexpected GRU input shape: {tuple(x.shape)}")
        cells = height * width
        series = x.squeeze(2).reshape(batch, steps, cells).permute(0, 2, 1)
        last = series[:, :, -1:].detach()
        centered = series - last
        encoded = centered.reshape(batch * cells, steps, 1)
        _, hidden = self.encoder(encoded)
        state = hidden[-1]
        previous = torch.zeros(batch * cells, 1, device=x.device, dtype=x.dtype)
        outputs = []
        for _ in range(self.t_out):
            state = self.decoder(previous, state)
            previous = self.projection(state)
            outputs.append(previous)
        prediction = torch.stack(outputs, dim=1).reshape(batch, cells, self.t_out)
        prediction = prediction + last
        return prediction.permute(0, 2, 1).reshape(batch, self.t_out, 1, height, width)


class MovingAverage(nn.Module):
    def __init__(self, kernel_size: int) -> None:
        super().__init__()
        self.kernel_size = kernel_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        left = (self.kernel_size - 1) // 2
        right = self.kernel_size - 1 - left
        padded = F.pad(x, (left, right), mode="replicate")
        return F.avg_pool1d(padded, kernel_size=self.kernel_size, stride=1)


class DLinear(nn.Module):
    """DLinear with channel-shared direct seasonal and trend projections."""

    def __init__(self, t_in: int, t_out: int, height: int, width: int, moving_avg: int = 25) -> None:
        super().__init__()
        self.t_in = t_in
        self.t_out = t_out
        self.height = height
        self.width = width
        self.decomposition = MovingAverage(moving_avg)
        self.seasonal = nn.Linear(t_in, t_out)
        self.trend = nn.Linear(t_in, t_out)
        nn.init.constant_(self.seasonal.weight, 1.0 / t_in)
        nn.init.constant_(self.trend.weight, 1.0 / t_in)
        nn.init.zeros_(self.seasonal.bias)
        nn.init.zeros_(self.trend.bias)

    def forward(self, x: torch.Tensor, phase: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch, steps, channels, height, width = x.shape
        if steps != self.t_in or channels != 1:
            raise ValueError(f"Unexpected DLinear input shape: {tuple(x.shape)}")
        series = x.squeeze(2).reshape(batch, steps, height * width).permute(0, 2, 1)
        trend = self.decomposition(series)
        seasonal = series - trend
        prediction = self.seasonal(seasonal) + self.trend(trend)
        return prediction.permute(0, 2, 1).reshape(batch, self.t_out, 1, height, width)


class PatchTST(nn.Module):
    """Compact channel-independent PatchTST adapter with RevIN-style scaling."""

    def __init__(
        self,
        t_in: int,
        t_out: int,
        height: int,
        width: int,
        patch_len: int = 4,
        stride: int = 4,
        d_model: int = 128,
        n_heads: int = 4,
        layers: int = 3,
        d_ff: int = 256,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.t_in = t_in
        self.t_out = t_out
        self.height = height
        self.width = width
        self.patch_len = min(patch_len, t_in)
        self.stride = min(stride, self.patch_len)
        self.patch_count = max(1, math.ceil((t_in - self.patch_len) / self.stride) + 1)
        self.padded_length = (self.patch_count - 1) * self.stride + self.patch_len
        self.patch_projection = nn.Linear(self.patch_len, d_model)
        self.position = nn.Parameter(torch.zeros(1, self.patch_count, d_model))
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=layers, norm=nn.LayerNorm(d_model))
        self.head = nn.Sequential(nn.Flatten(start_dim=1), nn.Dropout(dropout), nn.Linear(self.patch_count * d_model, t_out))

    def forward(self, x: torch.Tensor, phase: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch, steps, channels, height, width = x.shape
        if steps != self.t_in or channels != 1:
            raise ValueError(f"Unexpected PatchTST input shape: {tuple(x.shape)}")
        cells = height * width
        series = x.squeeze(2).reshape(batch, steps, cells).permute(0, 2, 1)
        mean = series.mean(dim=-1, keepdim=True).detach()
        stdev = torch.sqrt(series.var(dim=-1, keepdim=True, unbiased=False) + 1e-5)
        normalized = (series - mean) / stdev
        if self.padded_length > self.t_in:
            normalized = F.pad(normalized, (0, self.padded_length - self.t_in), mode="replicate")
        patches = normalized.unfold(-1, self.patch_len, self.stride)
        tokens = self.patch_projection(patches.reshape(batch * cells, self.patch_count, self.patch_len))
        encoded = self.encoder(tokens + self.position)
        prediction = self.head(encoded).reshape(batch, cells, self.t_out)
        prediction = prediction * stdev + mean
        return prediction.permute(0, 2, 1).reshape(batch, self.t_out, 1, height, width)


class MTGNNAdapter(nn.Module):
    """Shape adapter around the project's MTGNN port of the official model."""

    def __init__(self, t_in: int, t_out: int, height: int, width: int, device: torch.device) -> None:
        super().__init__()
        from utils.MTGNN import gtnet

        self.t_out = t_out
        self.height = height
        self.width = width
        nodes = height * width
        self.model = gtnet(
            gcn_true=True,
            buildA_true=True,
            gcn_depth=2,
            num_nodes=nodes,
            device=device,
            predefined_A=None,
            static_feat=None,
            dropout=0.3,
            subgraph_size=min(20, nodes),
            node_dim=40,
            dilation_exponential=1,
            conv_channels=32,
            residual_channels=32,
            skip_channels=64,
            end_channels=128,
            seq_length=t_in,
            in_dim=1,
            out_dim=t_out,
            layers=3,
            propalpha=0.05,
            tanhalpha=3,
            layer_norm_affline=True,
        )

    def forward(self, x: torch.Tensor, phase: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch = x.shape[0]
        values = x.squeeze(2).reshape(batch, x.shape[1], -1).permute(0, 2, 1).unsqueeze(1)
        prediction = self.model(values)
        if prediction.ndim != 4 or prediction.shape[-1] != 1:
            raise RuntimeError(f"Unexpected MTGNN output shape: {tuple(prediction.shape)}")
        return prediction.reshape(batch, self.t_out, 1, self.height, self.width)


class TimeMixerAdapter(nn.Module):
    def __init__(self, t_in: int, t_out: int, height: int, width: int) -> None:
        super().__init__()
        from utils.timemixer_model import TimeMixer

        self.model = TimeMixer(
            T_in=t_in,
            T_out=t_out,
            H=height,
            W=width,
            d_model=16,
            d_ff=32,
            n_heads=4,
            e_layers=2,
            dropout=0.1,
        )

    def forward(self, x: torch.Tensor, phase: Optional[torch.Tensor] = None) -> torch.Tensor:
        return self.model(x)


def build_model(
    name: str,
    t_in: int,
    t_out: int,
    height: int,
    width: int,
    device: torch.device,
    d_model: int = 128,
    seg_len: int = 6,
    dropout: float = 0.1,
    n_heads: int = 4,
    e_layers: int = 2,
    d_ff: int = 256,
    cycle_len: int = 24,
) -> nn.Module:
    if name == "seasonal_ha":
        return SeasonalHistoricalAverage(t_out, height, width, cycle_len)
    if name == "gru_seq2seq":
        return GRUSeq2Seq(t_out, height, width, hidden_dim=64)
    if name == "mtgnn":
        return MTGNNAdapter(t_in, t_out, height, width, device)
    if name == "patchtst":
        return PatchTST(t_in, t_out, height, width)
    if name == "dlinear":
        return DLinear(t_in, t_out, height, width)
    if name == "timemixer":
        return TimeMixerAdapter(t_in, t_out, height, width)
    if name == "GTSRNet":
        return GTSRNet(
            t_in=t_in,
            t_out=t_out,
            height=height,
            width=width,
            variant="GTSRNet",
            d_model=d_model,
            seg_len=seg_len,
            dropout=dropout,
            n_heads=n_heads,
            e_layers=e_layers,
            d_ff=d_ff,
            cycle_len=cycle_len,
        )
    raise ValueError(f"Unknown model {name!r}; choices are {MODEL_NAMES}")


def count_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
