import torch
import torch.nn as nn
from timm.models.layers import trunc_normal_


# ============================================================
# 1) Adaptive Spectral Block (UNCHANGED, your original module)
# ============================================================
class Adaptive_Spectral_Block(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.complex_weight_high = nn.Parameter(
            torch.randn(dim, 2, dtype=torch.float32) * 0.02
        )
        self.complex_weight = nn.Parameter(
            torch.randn(dim, 2, dtype=torch.float32) * 0.02
        )

        trunc_normal_(self.complex_weight_high, std=.02)
        trunc_normal_(self.complex_weight, std=.02)

        self.threshold_param = nn.Parameter(torch.rand(1))
        self.adaptive_filter = True

    def create_adaptive_high_freq_mask(self, x_fft):
        B, _, _ = x_fft.shape  # (B, N/2+1, C)
        energy = torch.abs(x_fft).pow(2).sum(dim=-1)      # (B, N/2+1)
        flat_energy = energy.view(B, -1)
        median_energy = flat_energy.median(dim=1, keepdim=True)[0].view(B, 1)
        normalized_energy = energy / (median_energy + 1e-6)

        adaptive_mask = (
            (normalized_energy > self.threshold_param).float()
            - self.threshold_param
        ).detach() + self.threshold_param

        return adaptive_mask.unsqueeze(-1)                # (B, N/2+1, 1)

    def forward(self, x_in):
        # x_in: (B, T, C)
        B, T, C = x_in.shape
        dtype = x_in.dtype
        x = x_in.to(torch.float32)

        x_fft = torch.fft.rfft(x, dim=1, norm='ortho')     # (B, T/2+1, C)

        weight = torch.view_as_complex(self.complex_weight)
        x_weighted = x_fft * weight

        if self.adaptive_filter:
            freq_mask = self.create_adaptive_high_freq_mask(x_fft)
            x_masked = x_fft * freq_mask.to(x.device)
        else:
            x_masked = x_fft

        weight_high = torch.view_as_complex(self.complex_weight_high)
        x_weighted += x_masked * weight_high

        x = torch.fft.irfft(x_weighted, n=T, dim=1, norm='ortho')
        return x.to(dtype)


# ============================================================
# 2) ASDF-Net: Adaptive Spectral Domain Forecasting Network
# ============================================================
class ASDFNet(nn.Module):
    """
    Adaptive Spectral Domain Forecasting Network (ASDF-Net)

    Input : (B, 25, 1, 20, 20)
    Output: (B, 12, 20, 20)
    """
    def __init__(self, close_len=25, pred_len=12, H=20, W=20):
        super().__init__()
        self.close_len = close_len
        self.pred_len = pred_len
        self.H, self.W = H, W
        self.N = H * W

        # Frequency-domain adaptive filtering
        self.spectral_block = Adaptive_Spectral_Block(dim=1)

        # Direct multi-step prediction head
        self.pred_head = nn.Sequential(
            nn.Linear(close_len, 32),
            nn.GELU(),
            nn.Linear(32, pred_len)
        )

    def forward(self, x):
        # x: (B, 25, 1, 20, 20)
        B, L, C, H, W = x.shape
        assert L == self.close_len and C == 1

        # (B,25,1,20,20) -> (B*400,25,1)
        x = x.squeeze(2).reshape(B, L, self.N)
        x = x.permute(0, 2, 1).contiguous()
        x = x.view(B * self.N, L, 1)

        # Adaptive spectral filtering (time-domain enhancement)
        x = self.spectral_block(x)                  # (B*400,25,1)

        # Direct multi-step forecasting
        x = x.squeeze(-1)                            # (B*400,25)
        y = self.pred_head(x)                        # (B*400,12)

        # (B*400,12) -> (B,12,20,20)
        y = y.view(B, self.N, self.pred_len)
        y = y.permute(0, 2, 1).contiguous()
        y = y.view(B, self.pred_len, H, W)

        return y


# ============================================================
# 3) Quick sanity check
# ============================================================
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = ASDFNet(close_len=25, pred_len=12, H=20, W=20).to(device)
    x = torch.randn(4, 25, 1, 20, 20).to(device)

    y = model(x)
    print("Output shape:", y.shape)  # expected: (4,12,20,20)
