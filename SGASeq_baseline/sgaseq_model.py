"""SGA-Seq adapted to grid traffic and a 24-input/12-output protocol.

The implementation follows the authors' public Apache-2.0 code at
https://github.com/OvOYu/SGA-Seq.  The public model assumes equal input and
output lengths.  Here its layer-wise residual path is generalized so that a
24-step history can directly produce 12 future steps.  The station-aware
adaptive graph convolution, local graph attention, GRU gates and additive
multi-layer forecasts are retained.

The original dense all-station attention is prohibitively expensive on the
400/560-node grids used here.  We retain the learned adaptive graph but keep
the strongest ``graph_neighbors`` connections for each station.  This makes
the five-dataset experiment practical and records the setting in metrics.json.
"""

from __future__ import annotations

import math
from typing import Tuple

import torch
from torch import nn
import torch.nn.functional as F


def adaptive_neighbors(
    node_embeddings: torch.Tensor, neighbors: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return learned top-k neighbor indices and normalized edge weights."""

    nodes = node_embeddings.shape[0]
    k = min(max(1, int(neighbors)), nodes)
    similarity = F.relu(node_embeddings @ node_embeddings.transpose(0, 1))
    values, indices = torch.topk(similarity, k=k, dim=-1)
    return indices, torch.softmax(values, dim=-1)


class LocalGraphAttention(nn.Module):
    """Multi-head attention restricted to learned adaptive graph neighbors."""

    def __init__(self, model_dim: int, num_heads: int) -> None:
        super().__init__()
        if model_dim % num_heads:
            raise ValueError("model_dim must be divisible by num_heads")
        self.model_dim = model_dim
        self.num_heads = num_heads
        self.head_dim = model_dim // num_heads

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        neighbor_index: torch.Tensor,
    ) -> torch.Tensor:
        # query/key/value: [B,N,D], neighbor_index: [N,K]
        batch, nodes, _ = query.shape
        count = neighbor_index.shape[1]
        q = query.reshape(batch, nodes, self.num_heads, self.head_dim)
        k = key[:, neighbor_index, :].reshape(
            batch, nodes, count, self.num_heads, self.head_dim
        )
        v = value[:, neighbor_index, :].reshape(
            batch, nodes, count, self.num_heads, self.head_dim
        )
        score = (q.unsqueeze(2) * k).sum(dim=-1) / math.sqrt(self.head_dim)
        attention = torch.softmax(score, dim=2)
        output = (attention.unsqueeze(-1) * v).sum(dim=2)
        return output.reshape(batch, nodes, self.model_dim)


class StationSpecificGraphConv(nn.Module):
    """Node-adaptive two-support graph convolution from the public SGA-Seq."""

    def __init__(self, in_dim: int, out_dim: int, embed_dim: int) -> None:
        super().__init__()
        self.weights_pool = nn.Parameter(torch.empty(embed_dim, 2, in_dim, out_dim))
        self.bias_pool = nn.Parameter(torch.empty(embed_dim, out_dim))
        nn.init.xavier_uniform_(self.weights_pool)
        nn.init.xavier_uniform_(self.bias_pool)

    def forward(
        self,
        x: torch.Tensor,
        node_embeddings: torch.Tensor,
        neighbor_index: torch.Tensor,
        neighbor_weight: torch.Tensor,
    ) -> torch.Tensor:
        # Adaptive graph aggregation without constructing [B,N,N,D].
        neighbor_x = x[:, neighbor_index, :]
        aggregated = (neighbor_x * neighbor_weight[None, :, :, None]).sum(dim=2)
        supports = torch.stack((x, aggregated), dim=2)  # [B,N,2,Cin]
        weights = torch.einsum(
            "nd,dkio->nkio", node_embeddings, self.weights_pool
        )
        bias = node_embeddings @ self.bias_pool
        return torch.einsum("bnki,nkio->bno", supports, weights) + bias


class StationAwareGraphNet(nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        embed_dim: int,
        num_heads: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.gcn = StationSpecificGraphConv(in_dim, out_dim, embed_dim)
        self.attention = LocalGraphAttention(out_dim, num_heads)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(out_dim)

    def forward(
        self,
        x: torch.Tensor,
        state: torch.Tensor,
        node_embeddings: torch.Tensor,
        neighbor_index: torch.Tensor,
        neighbor_weight: torch.Tensor,
    ) -> torch.Tensor:
        graph_x = self.gcn(
            x, node_embeddings, neighbor_index, neighbor_weight
        )
        attended = self.attention(
            self.norm(graph_x), state, state, neighbor_index
        )
        return graph_x + self.dropout(attended)


class SGAGRUCell(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        embed_dim: int,
        num_heads: int,
        dropout: float,
    ) -> None:
        super().__init__()
        joint_dim = input_dim + hidden_dim
        self.hidden_dim = hidden_dim
        self.update_gate = StationAwareGraphNet(
            joint_dim, hidden_dim, embed_dim, num_heads, dropout
        )
        self.reset_gate = StationAwareGraphNet(
            joint_dim, hidden_dim, embed_dim, num_heads, dropout
        )
        self.candidate = StationAwareGraphNet(
            joint_dim, hidden_dim, embed_dim, num_heads, dropout
        )

    def forward(
        self,
        x: torch.Tensor,
        state: torch.Tensor,
        node_embeddings: torch.Tensor,
        neighbor_index: torch.Tensor,
        neighbor_weight: torch.Tensor,
    ) -> torch.Tensor:
        joined = torch.cat((x, state), dim=-1)
        z = torch.sigmoid(
            self.update_gate(
                joined, state, node_embeddings, neighbor_index, neighbor_weight
            )
        )
        r = torch.sigmoid(
            self.reset_gate(
                joined, state, node_embeddings, neighbor_index, neighbor_weight
            )
        )
        candidate_input = torch.cat((x, z * state), dim=-1)
        candidate = torch.tanh(
            self.candidate(
                candidate_input,
                state,
                node_embeddings,
                neighbor_index,
                neighbor_weight,
            )
        )
        return r * state + (1.0 - r) * candidate


class SGASequencePredictor(nn.Module):
    """Layer-wise residual SGA-GRU with additive direct forecasts."""

    def __init__(
        self,
        hidden_dim: int,
        embed_dim: int,
        layers: int,
        out_steps: int,
        kernel: int,
        num_heads: int,
        dropout: float,
    ) -> None:
        super().__init__()
        if layers < 1:
            raise ValueError("SGA-Seq requires at least one recurrent layer")
        if kernel < 1:
            raise ValueError("kernel must be positive")
        self.hidden_dim = hidden_dim
        self.layers = layers
        self.out_steps = out_steps
        self.kernel = kernel
        self.cells = nn.ModuleList(
            SGAGRUCell(
                hidden_dim, hidden_dim, embed_dim, num_heads, dropout
            )
            for _ in range(layers)
        )
        self.predictors = nn.ModuleList(
            nn.Conv2d(kernel, out_steps, kernel_size=(1, hidden_dim))
            for _ in range(layers)
        )
        # This residual projection is the 24->12 compatibility adaptation.
        self.residual_projections = nn.ModuleList(
            nn.Linear(hidden_dim, hidden_dim) for _ in range(max(0, layers - 1))
        )
        self.dropouts = nn.ModuleList(nn.Dropout(dropout) for _ in range(layers))

    def forward(
        self,
        source: torch.Tensor,
        node_embeddings: torch.Tensor,
        neighbor_index: torch.Tensor,
        neighbor_weight: torch.Tensor,
    ) -> torch.Tensor:
        batch, _, nodes, _ = source.shape
        current = source
        forecasts = []
        for layer, cell in enumerate(self.cells):
            skip = current
            state = source.new_zeros(batch, nodes, self.hidden_dim)
            sequence = []
            for step in range(current.shape[1]):
                state = cell(
                    current[:, step],
                    state,
                    node_embeddings,
                    neighbor_index,
                    neighbor_weight,
                )
                sequence.append(state)
            hidden_sequence = torch.stack(sequence, dim=1)
            tail = hidden_sequence[:, -self.kernel :]
            if tail.shape[1] < self.kernel:
                raise ValueError("kernel cannot exceed the input history length")
            forecasts.append(self.predictors[layer](self.dropouts[layer](tail)))
            if layer < self.layers - 1:
                correction = self.residual_projections[layer](hidden_sequence)
                current = skip - correction
        return torch.stack(forecasts, dim=0).sum(dim=0)  # [B,Tout,N,1]


class SGASeq(nn.Module):
    """Grid adapter exposing ``forward(x, phase) -> [B,Tout,1,H,W]``."""

    def __init__(
        self,
        in_steps: int,
        out_steps: int,
        height: int,
        width: int,
        hidden_dim: int = 64,
        embed_dim: int = 14,
        layers: int = 2,
        kernel: int = 3,
        num_heads: int = 4,
        graph_neighbors: int = 8,
        cycle_len: int = 24,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if hidden_dim % 2:
            raise ValueError("hidden_dim must be even")
        if hidden_dim % num_heads:
            raise ValueError("hidden_dim must be divisible by num_heads")
        self.in_steps = in_steps
        self.out_steps = out_steps
        self.height = height
        self.width = width
        self.nodes = height * width
        self.cycle_len = cycle_len
        self.graph_neighbors = graph_neighbors
        half = hidden_dim // 2
        self.value_projection = nn.Linear(1, half)
        self.time_embedding = nn.Embedding(cycle_len, half)
        self.node_embeddings = nn.Parameter(torch.empty(self.nodes, embed_dim))
        nn.init.xavier_uniform_(self.node_embeddings)
        self.predictor = SGASequencePredictor(
            hidden_dim=hidden_dim,
            embed_dim=embed_dim,
            layers=layers,
            out_steps=out_steps,
            kernel=kernel,
            num_heads=num_heads,
            dropout=dropout,
        )

    def forward(self, x: torch.Tensor, phase: torch.Tensor) -> torch.Tensor:
        if x.ndim != 5 or x.shape[1] != self.in_steps:
            raise ValueError(
                f"expected [B,{self.in_steps},1,H,W], got {tuple(x.shape)}"
            )
        batch = x.shape[0]
        traffic = x[:, :, 0].reshape(batch, self.in_steps, self.nodes, 1)
        value_features = self.value_projection(traffic)
        offsets = torch.arange(self.in_steps, device=x.device)
        history_phase = (
            phase[:, None] - self.in_steps + offsets[None, :]
        ) % self.cycle_len
        time_features = self.time_embedding(history_phase)
        time_features = time_features[:, :, None, :].expand(
            -1, -1, self.nodes, -1
        )
        source = torch.cat((value_features, time_features), dim=-1)
        neighbor_index, neighbor_weight = adaptive_neighbors(
            self.node_embeddings, self.graph_neighbors
        )
        prediction = self.predictor(
            source,
            self.node_embeddings,
            neighbor_index,
            neighbor_weight,
        )
        return prediction.permute(0, 1, 3, 2).reshape(
            batch, self.out_steps, 1, self.height, self.width
        )


def count_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)

