"""GTSRNet phase-ablation models.

The three supported settings differ only in the phase terms added to the
spatial-branch token embedding:

* ``none``:          value token + grid identity
* ``daily``:         above + hour embedding + grid-hour joint embedding
* ``daily_weekly``:  above + weekday embedding

The temporal branch and prediction-level fusion are identical in all three
settings.  ``hour_index`` and ``weekday_index`` must describe the first
forecast timestamp of each sample.
"""

import torch
import torch.nn as nn

try:
    from utils.segrnn_gridB import SegRNNGrid
    from utils.layers.Transformer_EncDec import Encoder, EncoderLayer
    from utils.layers.SelfAttention_Family import FullAttention, AttentionLayer
    from utils.layers.Embed import DataEmbedding_inverted
except ImportError:  # Allows importing this file with utils/ on PYTHONPATH.
    from segrnn_gridB import SegRNNGrid
    from layers.Transformer_EncDec import Encoder, EncoderLayer
    from layers.SelfAttention_Family import FullAttention, AttentionLayer
    from layers.Embed import DataEmbedding_inverted


PHASE_MODES = ("none", "daily", "daily_weekly")


class SpatialPredictorPhaseAblation(nn.Module):
    def __init__(
        self,
        T_in,
        T_out,
        H,
        W,
        d_model=128,
        phase_mode="daily",
        n_heads=4,
        e_layers=2,
        d_ff=256,
        dropout=0.1,
        activation="gelu",
        factor=1,
        output_attention=False,
        use_norm=True,
        output_proj_dropout=0.1,
    ):
        super().__init__()
        if phase_mode not in PHASE_MODES:
            raise ValueError(f"phase_mode must be one of {PHASE_MODES}, got {phase_mode!r}")

        self.seq_len = T_in
        self.pred_len = T_out
        self.H = H
        self.W = W
        self.enc_in = H * W
        self.d_model = d_model
        self.phase_mode = phase_mode
        self.output_attention = output_attention
        self.use_norm = use_norm

        self.enc_embedding = DataEmbedding_inverted(
            self.seq_len, d_model, "timeF", "h", dropout
        )
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(
                            False,
                            factor,
                            attention_dropout=dropout,
                            output_attention=output_attention,
                        ),
                        d_model,
                        n_heads,
                    ),
                    d_model,
                    d_ff,
                    dropout=dropout,
                    activation=activation,
                )
                for _ in range(e_layers)
            ],
            norm_layer=nn.LayerNorm(d_model),
        )
        self.projector = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Dropout(output_proj_dropout),
            nn.Linear(d_model * 2, d_model * 4),
            nn.GELU(),
            nn.Dropout(output_proj_dropout),
            nn.Linear(d_model * 4, self.pred_len),
        )

        # E_c: persistent grid identity, retained in every ablation setting.
        self.grid_embedding = nn.Parameter(torch.empty(self.enc_in, d_model))

        # E_p and E_cp: the current daily-phase design (24 hourly states).
        self.daily_embedding = nn.Embedding(24, d_model)
        self.grid_daily_embedding = nn.Embedding(24, self.enc_in * d_model)

        # E_w: a weekday main effect only.  No grid-week interaction is added,
        # so the comparison isolates the incremental value of weekly phase.
        self.weekly_embedding = nn.Embedding(7, d_model)

        nn.init.xavier_normal_(self.grid_embedding)
        nn.init.xavier_normal_(self.daily_embedding.weight)
        nn.init.xavier_normal_(self.grid_daily_embedding.weight)
        nn.init.xavier_normal_(self.weekly_embedding.weight)

    @staticmethod
    def _check_index(name, index, batch_size, upper, device):
        if index is None:
            raise ValueError(f"{name} is required for the selected phase mode")
        index = index.to(device=device, dtype=torch.long).reshape(-1)
        if index.numel() != batch_size:
            raise ValueError(
                f"{name} must contain one value per sample; "
                f"expected {batch_size}, got {index.numel()}"
            )
        if torch.any(index < 0) or torch.any(index >= upper):
            raise ValueError(f"{name} values must be in [0, {upper - 1}]")
        return index

    def forecast(self, x_enc, hour_index=None, weekday_index=None):
        if self.use_norm:
            means = x_enc.mean(1, keepdim=True).detach()
            x_enc = x_enc - means
            stdev = torch.sqrt(
                torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5
            )
            x_enc = x_enc / stdev

        B, _, N = x_enc.shape
        if N != self.enc_in:
            raise ValueError(f"expected {self.enc_in} grids, got {N}")

        # The data embedding creates one token from the complete history of
        # each grid.  Time marks are intentionally not concatenated here.
        enc_out = self.enc_embedding(x_enc, None)[:, :N, :]
        enc_out = enc_out + self.grid_embedding.unsqueeze(0)

        if self.phase_mode in ("daily", "daily_weekly"):
            hour_index = self._check_index(
                "hour_index", hour_index, B, 24, x_enc.device
            )
            daily = self.daily_embedding(hour_index).unsqueeze(1)
            grid_daily = self.grid_daily_embedding(hour_index).reshape(
                B, self.enc_in, self.d_model
            )
            enc_out = enc_out + daily + grid_daily

        if self.phase_mode == "daily_weekly":
            weekday_index = self._check_index(
                "weekday_index", weekday_index, B, 7, x_enc.device
            )
            weekly = self.weekly_embedding(weekday_index).unsqueeze(1)
            enc_out = enc_out + weekly

        residual = enc_out
        enc_out, attns = self.encoder(enc_out, attn_mask=None)
        dec_out = self.projector(enc_out + residual).permute(0, 2, 1)

        if self.use_norm:
            dec_out = dec_out * stdev[:, 0, :].unsqueeze(1)
            dec_out = dec_out + means[:, 0, :].unsqueeze(1)
        return dec_out, attns

    def forward(self, x, hour_index=None, weekday_index=None):
        if x.dim() == 5:
            if x.shape[2] != 1:
                raise ValueError(f"expected one traffic channel, got shape {tuple(x.shape)}")
            x = x.squeeze(2)
        B, T, H, W = x.shape
        if (T, H, W) != (self.seq_len, self.H, self.W):
            raise ValueError(
                f"expected input [B,{self.seq_len},{self.H},{self.W}], "
                f"got {tuple(x.shape)}"
            )
        x_enc = x.reshape(B, T, H * W)
        dec_out, attns = self.forecast(x_enc, hour_index, weekday_index)
        dec_out = dec_out.reshape(B, self.pred_len, H, W)
        return (dec_out, attns) if self.output_attention else dec_out


class GTSRNetPhaseAblation(nn.Module):
    def __init__(
        self,
        T_in,
        T_out,
        H,
        W,
        d_model=128,
        seg_len=6,
        dropout=0.1,
        n_heads=4,
        phase_mode="daily",
    ):
        super().__init__()
        self.H = H
        self.W = W
        self.T_out = T_out
        self.phase_mode = phase_mode

        self.spatial_branch = SpatialPredictorPhaseAblation(
            T_in=T_in,
            T_out=T_out,
            H=H,
            W=W,
            d_model=d_model,
            phase_mode=phase_mode,
            n_heads=n_heads,
            dropout=dropout,
        )
        self.temporal_branch = SegRNNGrid(
            Tin=T_in,
            Tout=T_out,
            H=H,
            W=W,
            d_model=d_model,
            seg_len=seg_len,
            dropout=dropout,
        )
        self.fusion_weight = nn.Parameter(torch.zeros(1))

    def forward(self, x, hour_index=None, weekday_index=None):
        out_spatial = self.spatial_branch(x, hour_index, weekday_index)
        out_spatial = out_spatial.unsqueeze(2)

        out_temporal = self.temporal_branch(x)
        out_temporal = out_temporal.reshape(
            x.shape[0], self.T_out, self.H, self.W
        ).unsqueeze(2)

        alpha = torch.sigmoid(self.fusion_weight)
        return alpha * out_spatial + (1.0 - alpha) * out_temporal

