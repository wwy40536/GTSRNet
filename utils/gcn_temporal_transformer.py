# utils/gcn_temporal_transformer.py
# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F


def build_grid_adj(h: int, w: int, self_loop: bool = True, device=None, dtype=torch.float32):
    """
    20x20 栅格 4-neighbor 邻接，返回 A_hat = D^{-1/2} A D^{-1/2}
    节点 idx = r*w + c
    """
    n = h * w
    A = torch.zeros((n, n), dtype=dtype, device=device)

    def add_edge(i, j):
        A[i, j] = 1.0
        A[j, i] = 1.0

    for r in range(h):
        for c in range(w):
            i = r * w + c
            if r > 0:
                add_edge(i, (r - 1) * w + c)
            if r < h - 1:
                add_edge(i, (r + 1) * w + c)
            if c > 0:
                add_edge(i, r * w + (c - 1))
            if c < w - 1:
                add_edge(i, r * w + (c + 1))

    if self_loop:
        A.fill_diagonal_(1.0)

    deg = A.sum(dim=1)
    deg_inv_sqrt = (deg + 1e-6).pow(-0.5)
    A_hat = deg_inv_sqrt[:, None] * A * deg_inv_sqrt[None, :]
    return A_hat


class GraphConv(nn.Module):
    """
    X' = A_hat @ X @ W
    输入:  (B, T, N, Fin)
    输出:  (B, T, N, Fout)
    """
    def __init__(self, fin: int, fout: int, bias: bool = True):
        super().__init__()
        self.lin = nn.Linear(fin, fout, bias=bias)

    def forward(self, x: torch.Tensor, A_hat: torch.Tensor) -> torch.Tensor:
        x = self.lin(x)                      # (B,T,N,Fout)
        x = torch.matmul(A_hat, x)           # (B,T,N,Fout)  (N,N) @ (B,T,N,F)
        return x


class SpatialGCN(nn.Module):
    """
    每个时间步做两层 GCN（共享权重）
    输入:  (B,T,N,Fin)
    输出:  (B,T,N,D)
    """
    def __init__(self, fin: int, d_model: int, dropout: float = 0.1):
        super().__init__()
        self.g1 = GraphConv(fin, d_model)
        self.g2 = GraphConv(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor, A_hat: torch.Tensor) -> torch.Tensor:
        h = F.gelu(self.g1(x, A_hat))
        h = self.dropout(h)
        h = F.gelu(self.g2(h, A_hat))
        h = self.dropout(h)
        return self.norm(h)


class TemporalTransformer(nn.Module):
    """
    对每个节点的时间序列做 TransformerEncoder
    输入:  (B,T,N,D)
    输出:  (B,T,N,D)
    """
    def __init__(self, d_model: int, nhead: int = 4, num_layers: int = 2, dim_ff: int = 128, dropout: float = 0.1):
        super().__init__()
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_ff,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.enc = nn.TransformerEncoder(enc_layer, num_layers=num_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B,T,N,D) -> (B*N, T, D)
        B, T, N, D = x.shape
        xn = x.permute(0, 2, 1, 3).contiguous().view(B * N, T, D)
        xn = self.enc(xn)  # (B*N, T, D)
        return xn.view(B, N, T, D).permute(0, 2, 1, 3).contiguous()


class GCN_TemporalTransformer(nn.Module):
    """
    空间: GCN
    时间: TransformerEncoder
    输入:  (B, Tin, 1, H, W)
    输出:  (B, Tout, 1, H, W)
    """
    def __init__(
        self,
        Tin: int = 25,
        Tout: int = 8,
        H: int = 20,
        W: int = 20,
        in_channels: int = 1,
        d_model: int = 64,
        nhead: int = 4,
        t_layers: int = 2,
        dim_ff: int = 128,
        dropout: float = 0.1,
        self_loop: bool = True,
        use_skip_last: bool = True,   # 输出加 last frame，收敛更稳
    ):
        super().__init__()
        self.Tin = Tin
        self.Tout = Tout
        self.H = H
        self.W = W
        self.N = H * W
        self.use_skip_last = use_skip_last

        A_hat = build_grid_adj(H, W, self_loop=self_loop, device=None)
        self.register_buffer("A_hat", A_hat)

        # (flow=1) -> d_model
        self.in_proj = nn.Linear(in_channels, d_model)

        self.spatial = SpatialGCN(d_model, d_model, dropout=dropout)
        self.temporal = TemporalTransformer(d_model, nhead=nhead, num_layers=t_layers, dim_ff=dim_ff, dropout=dropout)

        # 用最后时刻 token 预测 Tout
        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, Tout)
        )

        # 可选：空间平滑一下
        self.refine = nn.Conv2d(Tout, Tout, kernel_size=3, padding=1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, Tin, 1, H, W)
        """
        B, T, C, H, W = x.shape
        assert T == self.Tin and C == 1 and H == self.H and W == self.W

        last = x[:, -1, 0]  # (B,H,W)

        # (B,T,1,H,W)->(B,T,H,W,1)->(B,T,N,1)
        x = x.permute(0, 1, 3, 4, 2).contiguous().view(B, T, self.N, 1)

        # proj
        h = self.in_proj(x)  # (B,T,N,D)

        # spatial GCN per time
        h = self.spatial(h, self.A_hat)

        # temporal transformer per node
        h = self.temporal(h)  # (B,T,N,D)

        # last token -> Tout
        h_last = h[:, -1]              # (B,N,D)
        y = self.head(h_last)          # (B,N,Tout)
        y = y.permute(0, 2, 1).contiguous().view(B, self.Tout, H, W)  # (B,Tout,H,W)
        y = self.refine(y)

        if self.use_skip_last:
            y = y + last.unsqueeze(1)  # residual

        return y.unsqueeze(2)          # (B,Tout,1,H,W)


if __name__ == "__main__":
    m = GCN_TemporalTransformer(Tin=25, Tout=8, H=20, W=20, d_model=64, nhead=4, t_layers=2)
    x = torch.randn(2, 25, 1, 20, 20)
    y = m(x)
    print(x.shape, y.shape)
