"""Full-table, low-rank, and no-joint variants of corrected ``GTSRNet``.

This is deliberately a *modified model*, not a paper reproduction.  It keeps
the original design intent (spatial Transformer branch + SegRNN temporal
branch + MLP-controlled fusion), while fixing the incompatible branch shapes
and making each ablation remove a real component.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
from torch import nn

try:
    from utils.layers.Embed import DataEmbedding_inverted
    from utils.layers.SelfAttention_Family import AttentionLayer, FullAttention
    from utils.layers.Transformer_EncDec import Encoder, EncoderLayer
    from utils.segrnn_gridB import SegRNNGrid
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "Cannot import the original project's utils package. Pass the project "
        "root to the training script with --project-root."
    ) from exc


VARIANTS: Tuple[str, ...] = ("GTSRNet", "lowrank_joint", "wo_joint")

VARIANT_DESCRIPTIONS: Dict[str, str] = {
    "GTSRNet": "Original Full V2 with the dense cell-phase lookup table",
    "lowrank_joint": "Full V2 with a factorized low-rank cell-phase interaction",
    "wo_joint": "Remove only the joint cell-phase interaction embedding",
}


class LowRankJointEmbedding(nn.Module):
    """Factorized cell-phase interaction with output shape ``[B, N, D]``.

    The dense lookup contains ``P*N*D`` parameters.  This CP-style
    factorization uses ``N*r + P*r + r*D + 1`` parameters while retaining a
    multiplicative interaction between cell identity and periodic phase.
    """

    def __init__(
        self,
        n_cells: int,
        cycle_len: int,
        d_model: int,
        rank: int = 16,
    ) -> None:
        super().__init__()
        if rank <= 0:
            raise ValueError("joint rank must be positive")
        self.n_cells = n_cells
        self.cycle_len = cycle_len
        self.d_model = d_model
        self.rank = rank
        self.cell_factor = nn.Parameter(torch.empty(n_cells, rank))
        self.phase_factor = nn.Embedding(cycle_len, rank)
        self.output_projection = nn.Linear(rank, d_model, bias=False)
        self.joint_scale = nn.Parameter(torch.tensor(1.0))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.xavier_normal_(self.cell_factor)
        nn.init.xavier_normal_(self.phase_factor.weight)
        nn.init.xavier_normal_(self.output_projection.weight)

    def forward(self, phase: torch.Tensor) -> torch.Tensor:
        if phase.ndim != 1:
            raise ValueError(f"Expected phase [B], got {tuple(phase.shape)}")
        phase_factor = self.phase_factor(phase)  # [B,r]
        interaction = self.cell_factor.unsqueeze(0) * phase_factor.unsqueeze(1)
        return self.joint_scale * self.output_projection(interaction)


class SpatialPredictor(nn.Module):
    """The GTSRNet spatial branch, with phase explicitly supplied."""

    def __init__(
        self,
        t_in: int,
        t_out: int,
        height: int,
        width: int,
        d_model: int,
        n_heads: int,
        e_layers: int,
        d_ff: int,
        dropout: float,
        cycle_len: int,
        use_sthe: bool,
        joint_mode: str = "full",
        joint_rank: int = 16,
    ) -> None:
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        self.t_in = t_in
        self.t_out = t_out
        self.height = height
        self.width = width
        self.n_cells = height * width
        self.d_model = d_model
        self.cycle_len = cycle_len
        self.use_sthe = use_sthe
        if joint_mode not in ("full", "lowrank", "none"):
            raise ValueError("joint_mode must be one of: full, lowrank, none")
        self.joint_mode = joint_mode
        self.joint_rank = joint_rank

        self.embedding = DataEmbedding_inverted(t_in, d_model, "timeF", "h", dropout)
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(
                            False,
                            factor=1,
                            attention_dropout=dropout,
                            output_attention=False,
                        ),
                        d_model,
                        n_heads,
                    ),
                    d_model,
                    d_ff,
                    dropout=dropout,
                    activation="gelu",
                )
                for _ in range(e_layers)
            ],
            norm_layer=nn.LayerNorm(d_model),
        )
        self.projector = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, t_out),
        )

        if use_sthe:
            self.channel_embedding = nn.Parameter(torch.empty(self.n_cells, d_model))
            self.phase_embedding = nn.Embedding(cycle_len, d_model)
            if joint_mode == "full":
                self.joint_embedding: Optional[nn.Module] = nn.Embedding(
                    cycle_len, self.n_cells * d_model
                )
            elif joint_mode == "lowrank":
                self.joint_embedding = LowRankJointEmbedding(
                    n_cells=self.n_cells,
                    cycle_len=cycle_len,
                    d_model=d_model,
                    rank=joint_rank,
                )
            else:
                self.joint_embedding = None
            nn.init.xavier_normal_(self.channel_embedding)
            nn.init.xavier_normal_(self.phase_embedding.weight)
            if joint_mode == "full" and self.joint_embedding is not None:
                assert isinstance(self.joint_embedding, nn.Embedding)
                nn.init.xavier_normal_(self.joint_embedding.weight)

    def forward(self, x: torch.Tensor, phase: Optional[torch.Tensor]) -> torch.Tensor:
        if x.ndim != 5:
            raise ValueError(f"Expected x [B,T,1,H,W], got {tuple(x.shape)}")
        batch, t_in, channels, height, width = x.shape
        if (t_in, channels, height, width) != (
            self.t_in,
            1,
            self.height,
            self.width,
        ):
            raise ValueError(
                f"Unexpected input shape {tuple(x.shape)} for "
                f"[B,{self.t_in},1,{self.height},{self.width}]"
            )

        values = x.squeeze(2).reshape(batch, self.t_in, self.n_cells)
        mean = values.mean(dim=1, keepdim=True).detach()
        centered = values - mean
        stdev = torch.sqrt(torch.var(centered, dim=1, keepdim=True, unbiased=False) + 1e-5)
        tokens = self.embedding(centered / stdev, None)

        if self.use_sthe:
            if phase is None:
                raise ValueError("phase is required while STHE is enabled")
            phase = phase.to(device=x.device, dtype=torch.long).reshape(batch)
            if torch.any((phase < 0) | (phase >= self.cycle_len)):
                raise ValueError(f"phase must be in [0,{self.cycle_len - 1}]")
            channel = self.channel_embedding.unsqueeze(0).expand(batch, -1, -1)
            phase_embedding = self.phase_embedding(phase).unsqueeze(1).expand(
                -1, self.n_cells, -1
            )
            tokens = tokens + channel + phase_embedding
            if self.joint_embedding is not None:
                joint = self.joint_embedding(phase)
                if self.joint_mode == "full":
                    joint = joint.reshape(batch, self.n_cells, self.d_model)
                tokens = tokens + joint

        residual = tokens
        encoded, _ = self.encoder(tokens, attn_mask=None)
        prediction = self.projector(encoded + residual).permute(0, 2, 1)
        prediction = prediction * stdev[:, :1, :] + mean[:, :1, :]
        return prediction.reshape(batch, self.t_out, self.height, self.width)


class DynamicGatedMLPFusion(nn.Module):
    """Stable MLP gate used for the corrected ``GTSRNet`` full model.

    The archived implementation stacked an image tensor with a flattened tensor,
    which is invalid.  Here both are flattened to [B,T,H*W] before fusion.  The
    final layer is zero-initialized, so training starts at an equal convex fusion
    rather than a random, unbounded prediction scale.
    """

    def __init__(self, hidden_dim: int = 32) -> None:
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )
        nn.init.zeros_(self.gate[-1].weight)
        nn.init.zeros_(self.gate[-1].bias)

    def forward(self, spatial: torch.Tensor, temporal: torch.Tensor) -> torch.Tensor:
        if spatial.shape != temporal.shape:
            raise ValueError(
                f"Fusion inputs must match, got {tuple(spatial.shape)} and {tuple(temporal.shape)}"
            )
        pair = torch.stack((spatial, temporal), dim=-1)
        alpha = torch.sigmoid(self.gate(pair)).squeeze(-1)
        return alpha * spatial + (1.0 - alpha) * temporal


class HorizonCalibratedGatedFusion(nn.Module):
    """V2 gate with independently removable calibration and horizon embedding."""

    def __init__(
        self,
        t_out: int,
        hidden_dim: int = 32,
        horizon_dim: int = 8,
        use_calibration: bool = True,
        use_horizon_embedding: bool = True,
    ) -> None:
        super().__init__()
        self.t_out = t_out
        self.use_horizon_embedding = use_horizon_embedding
        if use_calibration:
            self.spatial_scale = nn.Parameter(torch.ones(t_out))
            self.spatial_bias = nn.Parameter(torch.zeros(t_out))
            self.temporal_scale = nn.Parameter(torch.ones(t_out))
            self.temporal_bias = nn.Parameter(torch.zeros(t_out))
        else:
            self.register_buffer("spatial_scale", torch.ones(t_out))
            self.register_buffer("spatial_bias", torch.zeros(t_out))
            self.register_buffer("temporal_scale", torch.ones(t_out))
            self.register_buffer("temporal_bias", torch.zeros(t_out))
        self.horizon_embedding = (
            nn.Embedding(t_out, horizon_dim) if use_horizon_embedding else None
        )
        self.gate = nn.Sequential(
            nn.Linear(2 + (horizon_dim if use_horizon_embedding else 0), hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )
        # Start at the original equal fusion with identity calibration.
        nn.init.zeros_(self.gate[-1].weight)
        nn.init.zeros_(self.gate[-1].bias)

    def forward(self, spatial: torch.Tensor, temporal: torch.Tensor) -> torch.Tensor:
        if spatial.shape != temporal.shape or spatial.ndim != 3:
            raise ValueError(
                "Expected matching fusion tensors [B,T,H*W], got "
                f"{tuple(spatial.shape)} and {tuple(temporal.shape)}"
            )
        batch, horizons, cells = spatial.shape
        if horizons != self.t_out:
            raise ValueError(f"Expected {self.t_out} horizons, got {horizons}")

        spatial = (
            spatial * self.spatial_scale.view(1, horizons, 1)
            + self.spatial_bias.view(1, horizons, 1)
        )
        temporal = (
            temporal * self.temporal_scale.view(1, horizons, 1)
            + self.temporal_bias.view(1, horizons, 1)
        )
        pair = torch.stack((spatial, temporal), dim=-1)
        if self.horizon_embedding is None:
            gate_input = pair
        else:
            horizon_ids = torch.arange(horizons, device=spatial.device)
            horizon_features = self.horizon_embedding(horizon_ids).view(1, horizons, 1, -1)
            horizon_features = horizon_features.expand(batch, -1, cells, -1)
            gate_input = torch.cat((pair, horizon_features), dim=-1)
        alpha = torch.sigmoid(self.gate(gate_input)).squeeze(-1)
        return alpha * spatial + (1.0 - alpha) * temporal


class HorizonLinearFusion(nn.Module):
    """Static per-horizon fusion: V2 calibration capacity without a dynamic gate."""

    def __init__(self, t_out: int) -> None:
        super().__init__()
        self.t_out = t_out
        self.weight = nn.Parameter(torch.full((t_out, 2), 0.5))
        self.bias = nn.Parameter(torch.zeros(t_out))

    def forward(self, spatial: torch.Tensor, temporal: torch.Tensor) -> torch.Tensor:
        if spatial.shape != temporal.shape or spatial.ndim != 3:
            raise ValueError(
                "Expected matching fusion tensors [B,T,H*W], got "
                f"{tuple(spatial.shape)} and {tuple(temporal.shape)}"
            )
        _, horizons, _ = spatial.shape
        if horizons != self.t_out:
            raise ValueError(f"Expected {self.t_out} horizons, got {horizons}")
        pair = torch.stack((spatial, temporal), dim=-1)
        return (
            pair * self.weight.view(1, horizons, 1, 2)
        ).sum(dim=-1) + self.bias.view(1, horizons, 1)


class ConcatLinearFusion(nn.Module):
    """The no-gate comparison: a linear projection of the two branch outputs."""

    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(2, 1)
        with torch.no_grad():
            self.linear.weight.fill_(0.5)
            self.linear.bias.zero_()

    def forward(self, spatial: torch.Tensor, temporal: torch.Tensor) -> torch.Tensor:
        pair = torch.stack((spatial, temporal), dim=-1)
        return self.linear(pair).squeeze(-1)


class GTSRNet(nn.Module):
    """Fixed GTSRNet backbone with a focused V2 component-ablation suite."""

    def __init__(
        self,
        t_in: int,
        t_out: int,
        height: int,
        width: int,
        variant: str = "GTSRNet",
        d_model: int = 128,
        seg_len: int = 6,
        dropout: float = 0.1,
        n_heads: int = 4,
        e_layers: int = 2,
        d_ff: int = 256,
        cycle_len: int = 24,
        joint_rank: int = 16,
    ) -> None:
        super().__init__()
        if variant not in VARIANTS:
            raise ValueError(f"Unknown variant {variant!r}; choices are {VARIANTS}")
        if t_in % seg_len != 0 or t_out % seg_len != 0:
            raise ValueError("t_in and t_out must both divide seg_len")
        self.variant = variant
        self.t_out = t_out
        self.height = height
        self.width = width
        self.n_cells = height * width

        has_spatial = True
        has_temporal = True
        self.spatial_branch: Optional[nn.Module]
        self.temporal_branch: Optional[nn.Module]
        self.spatial_branch = (
            SpatialPredictor(
                t_in=t_in,
                t_out=t_out,
                height=height,
                width=width,
                d_model=d_model,
                n_heads=n_heads,
                e_layers=e_layers,
                d_ff=d_ff,
                dropout=dropout,
                cycle_len=cycle_len,
                use_sthe=True,
                joint_mode={
                    "GTSRNet": "full",
                    "lowrank_joint": "lowrank",
                    "wo_joint": "none",
                }[variant],
                joint_rank=joint_rank,
            )
            if has_spatial
            else None
        )
        self.temporal_branch = (
            SegRNNGrid(
                Tin=t_in,
                Tout=t_out,
                H=height,
                W=width,
                d_model=d_model,
                seg_len=seg_len,
                dropout=dropout,
                # In this project this code path is commented out inside
                # SegRNNGrid, so do not allocate unused attention parameters.
                use_cross_attn=0,
            )
            if has_temporal
            else None
        )
        self.fusion = HorizonCalibratedGatedFusion(t_out=t_out)

    def forward(
        self,
        x: torch.Tensor,
        phase: Optional[torch.Tensor] = None,
        return_branches: bool = False,
    ):
        spatial = self.spatial_branch(x, phase) if self.spatial_branch is not None else None
        temporal = self.temporal_branch(x) if self.temporal_branch is not None else None
        spatial_flat = (
            spatial.reshape(x.shape[0], self.t_out, self.n_cells)
            if spatial is not None
            else None
        )

        assert spatial_flat is not None and temporal is not None
        output = self.fusion(spatial_flat, temporal)
        output = output.reshape(x.shape[0], self.t_out, 1, self.height, self.width)
        if return_branches:
            spatial_output = (
                spatial_flat.reshape(x.shape[0], self.t_out, 1, self.height, self.width)
                if spatial_flat is not None
                else None
            )
            temporal_output = (
                temporal.reshape(x.shape[0], self.t_out, 1, self.height, self.width)
                if temporal is not None
                else None
            )
            return output, {"spatial": spatial_output, "temporal": temporal_output}
        return output

    def fusion_alpha(self):
        # The corrected MLP gate is sample/cell/horizon dependent; there is no
        # scientifically meaningful single alpha to report.
        return None


def count_trainable_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
