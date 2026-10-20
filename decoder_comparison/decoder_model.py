"""Controlled PMF-vs-RMF decoder ablation for the SegRNN temporal branch.

Both modes use exactly the same parameters and the same recurrent GRU encoder.
Only the future decoding computation graph changes:

* pmf: every future segment is decoded independently from the same encoder state;
* rmf: future segments are decoded sequentially and the previous predicted
  segment is embedded and fed into the next decoding step.

The output baseline is also switchable, identically for both decoders:

* last (original): prediction = residual + last observed value;
* seasonal: prediction = residual + value one seasonal period earlier, i.e.
  y_hat(t+h) = r(t+h) + x(t+h-P) (repeated for h > P). With a zero residual this
  is exactly the seasonal-naive forecast.

The encoder input is always last-value normalized, and the RMF feedback is
always expressed in that same input space, so ``baseline_mode="last"`` is
numerically identical to the previous implementation and no parameter is added.

The module returns predictions in the same shape used by the original project:
``(batch, prediction_length, number_of_grids)``.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn


class SegRNNGridDecoderAblation(nn.Module):
    """SegRNN grid forecaster with a switchable PMF/RMF decoder."""

    def __init__(
        self,
        Tin: int = 24,
        Tout: int = 12,
        H: int = 20,
        W: int = 20,
        d_model: int = 128,
        seg_len: int = 6,
        dropout: float = 0.1,
        decoder_mode: str = "pmf",
        baseline_mode: str = "last",
        seasonal_period: int = 24,
    ) -> None:
        super().__init__()

        decoder_mode = decoder_mode.lower()
        if decoder_mode not in {"pmf", "rmf"}:
            raise ValueError("decoder_mode must be 'pmf' or 'rmf'")
        baseline_mode = baseline_mode.lower()
        if baseline_mode not in {"last", "seasonal"}:
            raise ValueError("baseline_mode must be 'last' or 'seasonal'")
        if baseline_mode == "seasonal" and not 0 < seasonal_period <= Tin:
            raise ValueError("seasonal baseline requires 0 < seasonal_period <= Tin")
        if Tin % seg_len != 0:
            raise ValueError("Tin must be divisible by seg_len")
        if Tout % seg_len != 0:
            raise ValueError("Tout must be divisible by seg_len")
        if d_model % 4 != 0:
            raise ValueError("d_model must be divisible by 4")

        self.Tin = Tin
        self.Tout = Tout
        self.H = H
        self.W = W
        self.N = H * W
        self.d_model = d_model
        self.seg_len = seg_len
        self.seg_num_x = Tin // seg_len
        self.seg_num_y = Tout // seg_len
        self.decoder_mode = decoder_mode
        self.baseline_mode = baseline_mode
        self.seasonal_period = int(seasonal_period)

        # The spatial-grid identity encoding is identical in PMF and RMF.
        self.row_emb = nn.Parameter(torch.randn(H, d_model // 4) * 0.02)
        self.col_emb = nn.Parameter(torch.randn(W, d_model // 4) * 0.02)
        self.channel_fuse = nn.Sequential(
            nn.Linear(d_model // 2, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, d_model // 2),
        )

        # This layer is shared by the historical encoder and RMF feedback path.
        # Reusing it avoids adding decoder-specific parameters to RMF.
        self.valueEmbedding = nn.Sequential(
            nn.Linear(seg_len, d_model),
            nn.ReLU(inplace=True),
        )

        # The same GRU parameters are used for encoding and decoding, matching
        # the original SegRNN implementation in the supplied project.
        self.rnn = nn.GRU(
            input_size=d_model,
            hidden_size=d_model,
            num_layers=1,
            batch_first=True,
            bidirectional=False,
        )

        self.pos_emb = nn.Parameter(
            torch.randn(self.seg_num_y, d_model // 2) * 0.02
        )
        self.predict = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(d_model, seg_len),
        )

    def _build_cell_emb(self) -> torch.Tensor:
        row = self.row_emb[:, None, :].expand(self.H, self.W, -1)
        col = self.col_emb[None, :, :].expand(self.H, self.W, -1)
        cell = torch.cat([row, col], dim=-1).reshape(self.N, -1)
        return self.channel_fuse(cell)

    def _build_future_tokens(self, batch_size: int) -> torch.Tensor:
        """Return future tokens with shape ``(B*N, m, d_model)``."""
        cell = self._build_cell_emb()
        pos = self.pos_emb[None, None, :, :].expand(
            batch_size, self.N, self.seg_num_y, -1
        )
        grid = cell[None, :, None, :].expand(
            batch_size, self.N, self.seg_num_y, -1
        )
        return torch.cat([pos, grid], dim=-1).reshape(
            batch_size * self.N, self.seg_num_y, self.d_model
        )

    def _build_baseline(self, x_flat: torch.Tensor, seq_last: torch.Tensor) -> torch.Tensor:
        """Return the output baseline with shape ``(B, Tout, N)``."""
        if self.baseline_mode == "last":
            return seq_last.expand(-1, self.Tout, -1)
        steps = torch.arange(self.Tout, device=x_flat.device)
        index = self.Tin - self.seasonal_period + torch.remainder(steps, self.seasonal_period)
        return x_flat.index_select(1, index).detach()

    def _to_segments(self, values: torch.Tensor, batch_size: int) -> torch.Tensor:
        """``(B, Tout, N)`` -> ``(B*N, m, seg_len)``."""
        return (
            values.permute(0, 2, 1)
            .contiguous()
            .view(batch_size * self.N, self.seg_num_y, self.seg_len)
        )

    def _prepare_target_segments(
        self,
        target: torch.Tensor,
        seq_last: torch.Tensor,
        batch_size: int,
    ) -> torch.Tensor:
        """Normalize and segment a target for optional teacher-forced RMF."""
        if target.dim() == 5:
            if target.shape[2] != 1:
                raise ValueError("target channel dimension must equal 1")
            target = target.squeeze(2)
        if target.dim() != 4:
            raise ValueError("target must have shape (B,Tout,1,H,W) or (B,Tout,H,W)")

        expected = (batch_size, self.Tout, self.H, self.W)
        if tuple(target.shape) != expected:
            raise ValueError(
                "target shape mismatch: expected {}, got {}".format(
                    expected, tuple(target.shape)
                )
            )

        target_flat = target.reshape(batch_size, self.Tout, self.N)
        target_norm = target_flat - seq_last
        return (
            target_norm.permute(0, 2, 1)
            .contiguous()
            .view(batch_size * self.N, self.seg_num_y, self.seg_len)
        )

    def _decode_pmf(self, future_tokens: torch.Tensor, hn: torch.Tensor) -> torch.Tensor:
        """Decode all future segments independently in one batched GRU step."""
        bn = future_tokens.shape[0]
        m = self.seg_num_y

        # Future-segment index m is moved into the batch dimension. Therefore,
        # GRU sequence length is one and no g_(q-1) -> g_q edge exists.
        decoder_input = future_tokens.reshape(bn * m, 1, self.d_model)
        repeated_hn = (
            hn.squeeze(0)
            .unsqueeze(1)
            .expand(bn, m, self.d_model)
            .reshape(1, bn * m, self.d_model)
            .contiguous()
        )
        _, future_hidden = self.rnn(decoder_input, repeated_hn)
        future_hidden = future_hidden.squeeze(0).reshape(bn, m, self.d_model)
        return self.predict(future_hidden)

    def _decode_rmf(
        self,
        future_tokens: torch.Tensor,
        hn: torch.Tensor,
        feedback_offset: torch.Tensor,
        target_segments: Optional[torch.Tensor],
        teacher_forcing_ratio: float,
        feedback_noise_std: float,
    ) -> torch.Tensor:
        """Sequentially decode future segments with prediction feedback."""
        hidden = hn
        predictions = []

        for q in range(self.seg_num_y):
            decoder_input = future_tokens[:, q, :]

            if q > 0:
                # Convert the previous residual prediction back to the
                # last-value-normalized input space used by valueEmbedding.
                # feedback_offset is all zeros for baseline_mode="last".
                feedback = predictions[-1] + feedback_offset[:, q - 1, :]

                # Optional oracle/teacher-forcing mode is only a diagnostic.
                # The fair PMF-vs-RMF training default is ratio=0 (free running).
                if target_segments is not None and teacher_forcing_ratio > 0.0:
                    truth_feedback = target_segments[:, q - 1, :]
                    if teacher_forcing_ratio >= 1.0:
                        feedback = truth_feedback
                    else:
                        use_truth = (
                            torch.rand(
                                feedback.shape[0], 1, device=feedback.device
                            )
                            < teacher_forcing_ratio
                        )
                        feedback = torch.where(use_truth, truth_feedback, feedback)

                if feedback_noise_std > 0.0:
                    feedback = feedback + feedback_noise_std * torch.randn_like(feedback)

                decoder_input = decoder_input + self.valueEmbedding(feedback)

            _, hidden = self.rnn(decoder_input.unsqueeze(1), hidden)
            predicted_segment = self.predict(hidden.squeeze(0))
            predictions.append(predicted_segment)

        return torch.stack(predictions, dim=1)

    def forward(
        self,
        x: torch.Tensor,
        target: Optional[torch.Tensor] = None,
        teacher_forcing_ratio: float = 0.0,
        feedback_noise_std: float = 0.0,
    ) -> torch.Tensor:
        """Forecast a grid sequence.

        Args:
            x: Historical input, shape ``(B,Tin,1,H,W)``.
            target: Optional future target used only for RMF oracle diagnostics.
            teacher_forcing_ratio: Probability of feeding the true previous
                segment in RMF. Use 0 for the fair free-running comparison and
                1 only to measure the oracle error-accumulation gap.
            feedback_noise_std: Standard deviation of noise injected into the
                normalized RMF feedback. It has no effect in PMF.
        """
        if x.dim() != 5:
            raise ValueError("x must have shape (B,Tin,1,H,W)")
        batch_size, tin, channels, height, width = x.shape
        expected = (self.Tin, 1, self.H, self.W)
        if (tin, channels, height, width) != expected:
            raise ValueError(
                "input shape mismatch after batch: expected {}, got {}".format(
                    expected, (tin, channels, height, width)
                )
            )
        if not 0.0 <= teacher_forcing_ratio <= 1.0:
            raise ValueError("teacher_forcing_ratio must be in [0,1]")
        if feedback_noise_std < 0.0:
            raise ValueError("feedback_noise_std must be non-negative")

        # Last-value normalization, identical to the supplied implementation.
        x_flat = x.squeeze(2).reshape(batch_size, self.Tin, self.N)
        seq_last = x_flat[:, -1:, :].detach()
        x_norm = x_flat - seq_last

        # Recursive historical encoding is identical for PMF and RMF.
        x_by_node = x_norm.permute(0, 2, 1).contiguous()
        x_segments = x_by_node.view(
            batch_size * self.N, self.seg_num_x, self.seg_len
        )
        x_emb = self.valueEmbedding(x_segments)
        _, hn = self.rnn(x_emb)

        baseline = self._build_baseline(x_flat, seq_last)
        feedback_offset = self._to_segments(baseline - seq_last, batch_size)

        future_tokens = self._build_future_tokens(batch_size)
        target_segments = None
        if target is not None:
            target_segments = self._prepare_target_segments(
                target, seq_last, batch_size
            )

        if self.decoder_mode == "pmf":
            predicted_segments = self._decode_pmf(future_tokens, hn)
        else:
            predicted_segments = self._decode_rmf(
                future_tokens,
                hn,
                feedback_offset,
                target_segments,
                teacher_forcing_ratio,
                feedback_noise_std,
            )

        y_by_node = predicted_segments.reshape(batch_size, self.N, self.Tout)
        return y_by_node.permute(0, 2, 1).contiguous() + baseline

