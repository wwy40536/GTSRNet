"""Paper-oriented adapters for iTransformer, SegRNN and STID.

All models accept traffic grids shaped [B, Tin, 1, H, W] and return
[B, Tout, 1, H, W].  The implementations preserve the defining mechanism of
each paper while sharing the exact same input/output contract as the existing
baseline benchmark.
"""

from __future__ import annotations

from typing import Dict

import torch
from torch import nn


MODEL_NAMES = ("itransformer", "segrnn", "stid")
MODEL_DESCRIPTIONS: Dict[str, str] = {
    "itransformer": "iTransformer: grid cells are variate tokens; attention is across variables",
    "segrnn": "SegRNN: segment-wise recurrent encoder with parallel multi-step decoding",
    "stid": "STID: spatial identity and time-of-day/day-of-week identity embeddings with MLP",
}


def _flatten_grid(x: torch.Tensor) -> torch.Tensor:
    if x.ndim != 5 or x.shape[2] != 1:
        raise ValueError(f"Expected [B,T,1,H,W], got {tuple(x.shape)}")
    return x[:, :, 0].flatten(2)  # [B,T,N]


def _restore_grid(y: torch.Tensor, height: int, width: int) -> torch.Tensor:
    # y: [B,Tout,N]
    return y.reshape(y.shape[0], y.shape[1], 1, height, width)


class ITransformer(nn.Module):
    """Inverted Transformer: one token per traffic grid variable."""

    def __init__(
        self,
        t_in: int,
        t_out: int,
        height: int,
        width: int,
        d_model: int = 128,
        n_heads: int = 4,
        e_layers: int = 2,
        d_ff: int = 256,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.height, self.width = height, width
        self.history_embedding = nn.Linear(t_in, d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=e_layers)
        self.projector = nn.Linear(d_model, t_out)

    def forward(self, x: torch.Tensor, _time_slot: torch.Tensor) -> torch.Tensor:
        series = _flatten_grid(x)  # [B,T,N]
        mean = series.mean(dim=1, keepdim=True).detach()
        stdev = torch.sqrt(series.var(dim=1, keepdim=True, unbiased=False) + 1e-5).detach()
        normalized = (series - mean) / stdev
        tokens = self.history_embedding(normalized.transpose(1, 2))  # [B,N,D]
        encoded = self.encoder(tokens)
        forecast = self.projector(encoded).transpose(1, 2)  # [B,Tout,N]
        forecast = forecast * stdev + mean
        return _restore_grid(forecast, self.height, self.width)


class SegRNN(nn.Module):
    """Segment-wise RNN with parallel segment decoding."""

    def __init__(
        self,
        t_in: int,
        t_out: int,
        height: int,
        width: int,
        d_model: int = 128,
        seg_len: int = 6,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if t_in % seg_len or t_out % seg_len:
            raise ValueError("SegRNN requires seg_len to divide both input and output lengths")
        if d_model % 2:
            raise ValueError("SegRNN d_model must be even")
        self.t_in, self.t_out = t_in, t_out
        self.height, self.width = height, width
        self.nodes = height * width
        self.seg_len = seg_len
        self.in_segments = t_in // seg_len
        self.out_segments = t_out // seg_len
        self.segment_embedding = nn.Sequential(nn.Linear(seg_len, d_model), nn.ReLU())
        self.gru = nn.GRU(d_model, d_model, batch_first=True)
        half = d_model // 2
        self.position_embedding = nn.Parameter(torch.randn(self.out_segments, half) * 0.02)
        self.channel_embedding = nn.Parameter(torch.randn(self.nodes, half) * 0.02)
        self.predictor = nn.Sequential(nn.Dropout(dropout), nn.Linear(d_model, seg_len))

    def forward(self, x: torch.Tensor, _time_slot: torch.Tensor) -> torch.Tensor:
        series = _flatten_grid(x)  # [B,T,N]
        batch, _, nodes = series.shape
        if nodes != self.nodes:
            raise ValueError(f"Configured for {self.nodes} cells, got {nodes}")
        last = series[:, -1:, :].detach()
        centered = (series - last).transpose(1, 2).reshape(
            batch * nodes, self.in_segments, self.seg_len
        )
        encoded = self.segment_embedding(centered)
        _, hidden = self.gru(encoded)  # [1,B*N,D]

        pos = self.position_embedding[None, :, None, :].expand(
            batch, self.out_segments, nodes, -1
        )
        channel = self.channel_embedding[None, None, :, :].expand(
            batch, self.out_segments, nodes, -1
        )
        decoder_input = torch.cat((pos, channel), dim=-1).reshape(
            batch * self.out_segments * nodes, 1, -1
        )
        decoder_hidden = hidden.reshape(1, batch, 1, nodes, -1).expand(
            -1, -1, self.out_segments, -1, -1
        ).reshape(1, batch * self.out_segments * nodes, -1)
        _, decoded = self.gru(decoder_input, decoder_hidden.contiguous())
        segments = self.predictor(decoded.squeeze(0)).reshape(
            batch, self.out_segments, nodes, self.seg_len
        )
        forecast = segments.permute(0, 1, 3, 2).reshape(batch, self.t_out, nodes)
        forecast = forecast + last
        return _restore_grid(forecast, self.height, self.width)


class ResidualMLP(nn.Module):
    def __init__(self, channels: int, dropout: float) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(channels, channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(channels, channels),
        )
        self.activation = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.activation(x + self.block(x))


class STID(nn.Module):
    """Spatial-Temporal Identity MLP.

    time_slot is a weekly slot [0,167].  For timestamp-free mobile arrays, slot
    zero is aligned with the first observation, matching the existing protocol.
    """

    def __init__(
        self,
        t_in: int,
        t_out: int,
        height: int,
        width: int,
        d_model: int = 128,
        e_layers: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.height, self.width = height, width
        self.nodes = height * width
        self.history_embedding = nn.Linear(t_in, d_model)
        self.node_embedding = nn.Parameter(torch.randn(self.nodes, d_model) * 0.02)
        self.tod_embedding = nn.Embedding(24, d_model)
        self.dow_embedding = nn.Embedding(7, d_model)
        hidden = 4 * d_model
        self.encoder = nn.Sequential(*[ResidualMLP(hidden, dropout) for _ in range(e_layers)])
        self.regression = nn.Linear(hidden, t_out)

    def forward(self, x: torch.Tensor, time_slot: torch.Tensor) -> torch.Tensor:
        series = _flatten_grid(x)
        batch, _, nodes = series.shape
        if nodes != self.nodes:
            raise ValueError(f"Configured for {self.nodes} cells, got {nodes}")
        history = self.history_embedding(series.transpose(1, 2))
        node = self.node_embedding[None].expand(batch, -1, -1)
        tod = self.tod_embedding(torch.remainder(time_slot, 24))[:, None, :].expand(-1, nodes, -1)
        dow = self.dow_embedding(torch.remainder(time_slot // 24, 7))[:, None, :].expand(-1, nodes, -1)
        encoded = self.encoder(torch.cat((history, node, tod, dow), dim=-1))
        forecast = self.regression(encoded).transpose(1, 2)
        return _restore_grid(forecast, self.height, self.width)


def build_model(
    name: str,
    t_in: int,
    t_out: int,
    height: int,
    width: int,
    *,
    d_model: int,
    seg_len: int,
    dropout: float,
    n_heads: int,
    e_layers: int,
    d_ff: int,
) -> nn.Module:
    if name == "itransformer":
        return ITransformer(t_in, t_out, height, width, d_model, n_heads, e_layers, d_ff, dropout)
    if name == "segrnn":
        return SegRNN(t_in, t_out, height, width, d_model, seg_len, dropout)
    if name == "stid":
        return STID(t_in, t_out, height, width, d_model, e_layers, dropout)
    raise ValueError(f"Unknown model: {name}")


def count_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
