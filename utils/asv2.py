import torch
import torch.nn as nn
from timm.models.layers import trunc_normal_


# ============================================================
# 1) Adaptive Spectral Block (your original, minimal safe edits)
# ============================================================
class Adaptive_Spectral_Block(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.complex_weight_high = nn.Parameter(torch.randn(dim, 2, dtype=torch.float32) * 0.02)
        self.complex_weight = nn.Parameter(torch.randn(dim, 2, dtype=torch.float32) * 0.02)

        trunc_normal_(self.complex_weight_high, std=.02)
        trunc_normal_(self.complex_weight, std=.02)

        self.threshold_param = nn.Parameter(torch.rand(1))
        self.adaptive_filter = True

    def create_adaptive_high_freq_mask(self, x_fft):
        # x_fft: (B, T/2+1, C)
        B, _, _ = x_fft.shape
        energy = torch.abs(x_fft).pow(2).sum(dim=-1)         # (B, T/2+1)
        flat_energy = energy.view(B, -1)
        median_energy = flat_energy.median(dim=1, keepdim=True)[0].view(B, 1)
        normalized_energy = energy / (median_energy + 1e-6)
        adaptive_mask = ((normalized_energy > self.threshold_param).float() - self.threshold_param).detach() + self.threshold_param
        return adaptive_mask.unsqueeze(-1)                   # (B, T/2+1, 1)

    def forward(self, x_in):
        # x_in: (B, T, C)
        B, T, C = x_in.shape
        dtype = x_in.dtype
        x = x_in.to(torch.float32)

        x_fft = torch.fft.rfft(x, dim=1, norm='ortho')       # (B, T/2+1, C)

        weight = torch.view_as_complex(self.complex_weight)  # (C,)
        x_weighted = x_fft * weight

        if self.adaptive_filter:
            freq_mask = self.create_adaptive_high_freq_mask(x_fft)  # (B, T/2+1, 1)
            x_masked = x_fft * freq_mask.to(x.device)
        else:
            x_masked = x_fft

        weight_high = torch.view_as_complex(self.complex_weight_high)
        x_weighted = x_weighted + x_masked * weight_high

        x = torch.fft.irfft(x_weighted, n=T, dim=1, norm='ortho')   # (B, T, C)
        return x.to(dtype)


# ============================================================
# 2) ASDF-Net v2 (per-horizon heads)  [fast, improves long horizon]
# ============================================================
class ASDFNetV2(nn.Module):
    """
    Input : (B, 25, 1, 20, 20)
    Output: (B, 12, 20, 20)
    """
    def __init__(self, close_len=25, pred_len=12, H=20, W=20, hidden=64):
        super().__init__()
        self.close_len = close_len
        self.pred_len = pred_len
        self.H, self.W = H, W
        self.N = H * W

        self.spectral = Adaptive_Spectral_Block(dim=1)

        self.heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(close_len, hidden),
                nn.GELU(),
                nn.Linear(hidden, 1)
            ) for _ in range(pred_len)
        ])

    def forward(self, x):
        B, L, C, H, W = x.shape
        assert L == self.close_len and C == 1 and H == self.H and W == self.W

        x = x.squeeze(2).reshape(B, L, self.N)          # (B,L,N)
        x = x.permute(0, 2, 1).contiguous()             # (B,N,L)
        x = x.view(B * self.N, L, 1)                    # (B*N,L,1)

        x = self.spectral(x).squeeze(-1)                # (B*N,L)

        outs = [self.heads[k](x) for k in range(self.pred_len)]  # list of (B*N,1)
        y = torch.cat(outs, dim=1)                      # (B*N,pred_len)

        y = y.view(B, self.N, self.pred_len).permute(0, 2, 1).contiguous()
        y = y.view(B, self.pred_len, self.H, self.W)    # (B,pred_len,H,W)
        return y


# ============================================================
# 3) ASDF-Seq2Seq (Encoder-Decoder) [better long horizon]
# ============================================================
class ASDFSeq2Seq(nn.Module):
    """
    Spectral-enhanced Seq2Seq forecaster

    Input : (B, 25, 1, 20, 20)
    Output: (B, 12, 20, 20)
    """
    def __init__(self, close_len=25, pred_len=12, H=20, W=20, d_model=64, num_layers=1):
        super().__init__()
        self.close_len = close_len
        self.pred_len = pred_len
        self.H, self.W = H, W
        self.N = H * W

        self.spectral = Adaptive_Spectral_Block(dim=1)

        # encoder
        self.enc_in = nn.Linear(1, d_model)
        self.encoder = nn.GRU(input_size=d_model, hidden_size=d_model, num_layers=num_layers, batch_first=True)

        # decoder
        self.dec_in = nn.Linear(1, d_model)
        self.decoder = nn.GRU(input_size=d_model, hidden_size=d_model, num_layers=num_layers, batch_first=True)
        self.dec_out = nn.Linear(d_model, 1)

    def _reshape_future_for_teacher_forcing(self, y_future):
        """
        y_future: (B, pred_len, 1, H, W) -> (B*N, pred_len, 1)
        """
        B, T, C, H, W = y_future.shape
        y = y_future.squeeze(2).reshape(B, T, self.N)           # (B,T,N)
        y = y.permute(0, 2, 1).contiguous()                     # (B,N,T)
        y = y.view(B * self.N, T, 1)                            # (B*N,T,1)
        return y

    def forward(self, x, y_future=None, teacher_forcing_ratio=0.0):
        """
        x: (B, 25, 1, 20, 20)
        y_future (optional): (B, 12, 1, 20, 20)
        return: (B, 12, 20, 20)
        """
        B, L, C, H, W = x.shape
        assert L == self.close_len and C == 1 and H == self.H and W == self.W

        # (B,L,1,H,W)->(B*N,L,1)
        x0 = x.squeeze(2).reshape(B, L, self.N).permute(0, 2, 1).contiguous()  # (B,N,L)
        x0 = x0.view(B * self.N, L, 1)                                         # (B*N,L,1)

        # spectral enhance
        x1 = self.spectral(x0)                                                 # (B*N,L,1)

        # encoder
        enc_inp = self.enc_in(x1)                                              # (B*N,L,d)
        _, h = self.encoder(enc_inp)                                           # (layers, B*N, d)

        # prepare teacher forcing target if provided
        y_tf = None
        if y_future is not None:
            y_tf = self._reshape_future_for_teacher_forcing(y_future)          # (B*N,T,1)

        # decoder init: last observed value
        dec_input = x1[:, -1:, :]                                              # (B*N,1,1)

        outs = []
        for t in range(self.pred_len):
            din = self.dec_in(dec_input)                                       # (B*N,1,d)
            dout, h = self.decoder(din, h)                                     # (B*N,1,d)
            y_step = self.dec_out(dout)                                        # (B*N,1,1)
            outs.append(y_step)

            if (y_tf is not None) and (teacher_forcing_ratio > 0.0):
                if torch.rand(1, device=x.device).item() < teacher_forcing_ratio:
                    dec_input = y_tf[:, t:t+1, :]                              # (B*N,1,1)
                else:
                    dec_input = y_step.detach()
            else:
                dec_input = y_step.detach()

        y = torch.cat(outs, dim=1)                                             # (B*N,T,1)

        # reshape back: (B*N,T,1)->(B,T,H,W)
        y = y.view(B, self.N, self.pred_len).permute(0, 2, 1).contiguous()     # (B,T,N)
        y = y.view(B, self.pred_len, self.H, self.W)                           # (B,T,H,W)
        return y


# ============================================================
# 4) Horizon-weighted loss (recommended)
# ============================================================
def horizon_weighted_l1(pred, target, w_min=1.0, w_max=2.5):
    """
    pred/target: (B, T, H, W)
    """
    B, T, H, W = pred.shape
    w = torch.linspace(w_min, w_max, T, device=pred.device, dtype=pred.dtype)
    return ((pred - target).abs() * w.view(1, T, 1, 1)).mean()
