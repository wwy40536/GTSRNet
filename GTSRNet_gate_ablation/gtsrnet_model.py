"""GTSRNet model and component/fusion ablation variants.

Spatial Transformer, segmented recurrent prediction and the configured fusion
modules follow the supplied implementation. Public naming does not change
model computations, loss controls or parameter-layer names. Select the complete
model explicitly with variant="GTSRNet"; legacy variant="full" is a distinct
uncalibrated-gate comparison and is not an alias of the complete model.
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


VARIANTS: Tuple[str, ...] = (
    "full",
    "GTSRNet",
    "GTSRNet_no_aux",
    "fixed_avg",
    "horizon_static",
    "wo_g",
    "wo_seg",
    "wo_sthe",
    "wo_s",
    "wo_t",
)

VARIANT_DESCRIPTIONS: Dict[str, str] = {
    "full": "Modified GTSRNet: spatial Transformer + SegRNN + dynamic MLP gate",
    "GTSRNet": (
        "Improved GTSRNet: horizon-calibrated dynamic gate with branch auxiliary loss"
    ),
    "GTSRNet_no_aux": (
        "Same horizon-calibrated dynamic gate as GTSRNet, trained without branch auxiliary loss"
    ),
    "fixed_avg": "Replace the main gate with fixed 0.5/0.5 branch averaging",
    "horizon_static": (
        "Use one learned but sample-independent fusion weight for each forecast horizon"
    ),
    "wo_g": "Replace the main gate with Concat + Linear fusion",
    "wo_seg": "Use an unsegmented temporal branch (seg_len=1), retaining the main gate",
    "wo_sthe": "Remove spatial channel, hour-phase and joint embeddings, retaining the main gate",
    "wo_s": "Remove the spatial Transformer branch",
    "wo_t": "Remove the SegRNN temporal branch",
}


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
            self.joint_embedding = nn.Embedding(cycle_len, self.n_cells * d_model)
            nn.init.xavier_normal_(self.channel_embedding)
            nn.init.xavier_normal_(self.phase_embedding.weight)
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
            joint = self.joint_embedding(phase).reshape(batch, self.n_cells, self.d_model)
            tokens = tokens + channel + phase_embedding + joint

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
    """A dynamic gate that can correct branch bias separately at each horizon.

    ``wo_g`` won on Milan-Call because its unrestricted linear layer can adjust
    branch scale and bias, while the original dynamic gate is only a convex
    blend.  This layer preserves a bounded gate but gives each forecast horizon
    its own affine calibration for both branches.  The gate also receives an
    explicit learned horizon embedding, rather than having to infer the horizon
    from prediction values alone.
    """

    def __init__(self, t_out: int, hidden_dim: int = 32, horizon_dim: int = 8) -> None:
        super().__init__()
        self.t_out = t_out
        self.spatial_scale = nn.Parameter(torch.ones(t_out))
        self.spatial_bias = nn.Parameter(torch.zeros(t_out))
        self.temporal_scale = nn.Parameter(torch.ones(t_out))
        self.temporal_bias = nn.Parameter(torch.zeros(t_out))
        self.horizon_embedding = nn.Embedding(t_out, horizon_dim)
        self.gate = nn.Sequential(
            nn.Linear(2 + horizon_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )
        # Start at the original equal fusion with identity calibration.
        nn.init.zeros_(self.gate[-1].weight)
        nn.init.zeros_(self.gate[-1].bias)

    def calibrate(
        self, spatial: torch.Tensor, temporal: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if spatial.shape != temporal.shape or spatial.ndim != 3:
            raise ValueError(
                "Expected matching fusion tensors [B,T,H*W], got "
                f"{tuple(spatial.shape)} and {tuple(temporal.shape)}"
            )
        batch, horizons, cells = spatial.shape
        if horizons != self.t_out:
            raise ValueError(f"Expected {self.t_out} horizons, got {horizons}")

        calibrated_spatial = (
            spatial * self.spatial_scale.view(1, horizons, 1)
            + self.spatial_bias.view(1, horizons, 1)
        )
        calibrated_temporal = (
            temporal * self.temporal_scale.view(1, horizons, 1)
            + self.temporal_bias.view(1, horizons, 1)
        )
        return calibrated_spatial, calibrated_temporal

    def alpha(self, spatial: torch.Tensor, temporal: torch.Tensor) -> torch.Tensor:
        spatial, temporal = self.calibrate(spatial, temporal)
        batch, horizons, cells = spatial.shape
        horizon_ids = torch.arange(horizons, device=spatial.device)
        horizon_features = self.horizon_embedding(horizon_ids).view(1, horizons, 1, -1)
        horizon_features = horizon_features.expand(batch, -1, cells, -1)
        gate_input = torch.cat(
            (torch.stack((spatial, temporal), dim=-1), horizon_features), dim=-1
        )
        return torch.sigmoid(self.gate(gate_input)).squeeze(-1)

    def forward(self, spatial: torch.Tensor, temporal: torch.Tensor) -> torch.Tensor:
        spatial, temporal = self.calibrate(spatial, temporal)
        alpha = self.alpha_from_calibrated(spatial, temporal)
        return alpha * spatial + (1.0 - alpha) * temporal

    def alpha_from_calibrated(
        self, spatial: torch.Tensor, temporal: torch.Tensor
    ) -> torch.Tensor:
        batch, horizons, cells = spatial.shape
        horizon_ids = torch.arange(horizons, device=spatial.device)
        horizon_features = self.horizon_embedding(horizon_ids).view(1, horizons, 1, -1)
        horizon_features = horizon_features.expand(batch, -1, cells, -1)
        gate_input = torch.cat(
            (torch.stack((spatial, temporal), dim=-1), horizon_features), dim=-1
        )
        return torch.sigmoid(self.gate(gate_input)).squeeze(-1)


class HorizonStaticGatedFusion(nn.Module):
    """Learn one convex fusion weight per horizon, shared by samples and cells.

    The branch calibration parameters are intentionally identical to those in
    :class:`HorizonCalibratedGatedFusion`.  Consequently, this control isolates
    sample/cell-dependent gating from per-horizon calibration.
    """

    def __init__(self, t_out: int) -> None:
        super().__init__()
        self.t_out = t_out
        self.spatial_scale = nn.Parameter(torch.ones(t_out))
        self.spatial_bias = nn.Parameter(torch.zeros(t_out))
        self.temporal_scale = nn.Parameter(torch.ones(t_out))
        self.temporal_bias = nn.Parameter(torch.zeros(t_out))
        self.alpha_logits = nn.Parameter(torch.zeros(t_out))

    def calibrate(
        self, spatial: torch.Tensor, temporal: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if spatial.shape != temporal.shape or spatial.ndim != 3:
            raise ValueError(
                "Expected matching fusion tensors [B,T,H*W], got "
                f"{tuple(spatial.shape)} and {tuple(temporal.shape)}"
            )
        _, horizons, _ = spatial.shape
        if horizons != self.t_out:
            raise ValueError(f"Expected {self.t_out} horizons, got {horizons}")
        calibrated_spatial = (
            spatial * self.spatial_scale.view(1, horizons, 1)
            + self.spatial_bias.view(1, horizons, 1)
        )
        calibrated_temporal = (
            temporal * self.temporal_scale.view(1, horizons, 1)
            + self.temporal_bias.view(1, horizons, 1)
        )
        return calibrated_spatial, calibrated_temporal

    def alpha(self, spatial: torch.Tensor, temporal: torch.Tensor) -> torch.Tensor:
        if spatial.shape != temporal.shape or spatial.ndim != 3:
            raise ValueError(
                "Expected matching fusion tensors [B,T,H*W], got "
                f"{tuple(spatial.shape)} and {tuple(temporal.shape)}"
            )
        batch, horizons, cells = spatial.shape
        if horizons != self.t_out:
            raise ValueError(f"Expected {self.t_out} horizons, got {horizons}")
        return torch.sigmoid(self.alpha_logits).view(1, horizons, 1).expand(
            batch, horizons, cells
        )

    def forward(self, spatial: torch.Tensor, temporal: torch.Tensor) -> torch.Tensor:
        calibrated_spatial, calibrated_temporal = self.calibrate(spatial, temporal)
        alpha = self.alpha(spatial, temporal)
        return alpha * calibrated_spatial + (1.0 - alpha) * calibrated_temporal


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


class FixedAverageFusion(nn.Module):
    """Parameter-free control: equal average of the two branch predictions."""

    def forward(self, spatial: torch.Tensor, temporal: torch.Tensor) -> torch.Tensor:
        if spatial.shape != temporal.shape:
            raise ValueError(
                f"Fusion inputs must match, got {tuple(spatial.shape)} and "
                f"{tuple(temporal.shape)}"
            )
        return 0.5 * spatial + 0.5 * temporal

    def alpha(self, spatial: torch.Tensor, temporal: torch.Tensor) -> torch.Tensor:
        if spatial.shape != temporal.shape:
            raise ValueError(
                f"Fusion inputs must match, got {tuple(spatial.shape)} and "
                f"{tuple(temporal.shape)}"
            )
        return torch.full_like(spatial, 0.5)


class GTSRNet(nn.Module):
    """Strict component/fusion controls and the ``GTSRNet`` main model."""

    def __init__(
        self,
        t_in: int,
        t_out: int,
        height: int,
        width: int,
        variant: str = "full",
        d_model: int = 128,
        seg_len: int = 6,
        dropout: float = 0.1,
        n_heads: int = 4,
        e_layers: int = 2,
        d_ff: int = 256,
        cycle_len: int = 24,
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

        has_spatial = variant != "wo_s"
        has_temporal = variant != "wo_t"
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
                use_sthe=variant != "wo_sthe",
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
                seg_len=1 if variant == "wo_seg" else seg_len,
                dropout=dropout,
                # In this project this code path is commented out inside
                # SegRNNGrid, so do not allocate unused attention parameters.
                use_cross_attn=0,
            )
            if has_temporal
            else None
        )
        if has_spatial and has_temporal:
            if variant == "full":
                self.fusion = DynamicGatedMLPFusion()
            elif variant == "fixed_avg":
                self.fusion = FixedAverageFusion()
            elif variant == "horizon_static":
                self.fusion = HorizonStaticGatedFusion(t_out=t_out)
            elif variant == "wo_g":
                self.fusion = ConcatLinearFusion()
            else:
                # GTSRNet, GTSRNet_no_aux, wo_seg and wo_sthe share the same gate.
                # Therefore each component ablation changes only its named part.
                self.fusion = HorizonCalibratedGatedFusion(t_out=t_out)

    @property
    def supports_branch_auxiliary_loss(self) -> bool:
        """Whether this variant has both predictions needed by auxiliary loss."""
        return self.spatial_branch is not None and self.temporal_branch is not None

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

        if self.variant == "wo_s":
            assert temporal is not None
            output = temporal
        elif self.variant == "wo_t":
            assert spatial_flat is not None
            output = spatial_flat
        else:
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
        fusion = getattr(self, "fusion", None)
        if isinstance(fusion, HorizonStaticGatedFusion):
            return torch.sigmoid(fusion.alpha_logits).detach().cpu().tolist()
        if isinstance(fusion, FixedAverageFusion):
            return [0.5] * self.t_out
        # Dynamic gates are sample/cell/horizon dependent; their distributions
        # are exported by analyze_gate_weights.py rather than collapsed here.
        return None


def count_trainable_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
