# utils/gstcn.py
# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F


def build_grid_adj(h: int = 20, w: int = 20, k: int = 8) -> torch.Tensor:
    """
    构建 hxw 栅格的邻接矩阵 A (N,N)。
    k=4 表示4邻域，k=8表示8邻域。
    """
    assert k in (4, 8)
    N = h * w
    A = torch.zeros(N, N, dtype=torch.float32)

    def idx(r, c):
        return r * w + c

    for r in range(h):
        for c in range(w):
            u = idx(r, c)
            nbrs = []
            # 4-neighbors
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                rr, cc = r + dr, c + dc
                if 0 <= rr < h and 0 <= cc < w:
                    nbrs.append((rr, cc))
            # 8-neighbors
            if k == 8:
                for dr, dc in [(-1, -1), (-1, 1), (1, -1), (1, 1)]:
                    rr, cc = r + dr, c + dc
                    if 0 <= rr < h and 0 <= cc < w:
                        nbrs.append((rr, cc))

            for rr, cc in nbrs:
                v = idx(rr, cc)
                A[u, v] = 1.0
                A[v, u] = 1.0

    # self-loop
    A.fill_diagonal_(1.0)
    return A


def normalize_adj(A: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    归一化: D^{-1/2} A D^{-1/2}
    """
    deg = A.sum(dim=1)  # (N,)
    deg_inv_sqrt = torch.pow(deg + eps, -0.5)
    D_inv_sqrt = torch.diag(deg_inv_sqrt)
    return D_inv_sqrt @ A @ D_inv_sqrt


class GraphConv(nn.Module):
    """
    基础 GCN layer: H' = A_norm H W
    输入: (B, N, Cin)
    输出: (B, N, Cout)
    """
    def __init__(self, cin: int, cout: int, bias: bool = True):
        super().__init__()
        self.lin = nn.Linear(cin, cout, bias=bias)

    def forward(self, x: torch.Tensor, A_norm: torch.Tensor) -> torch.Tensor:
        # x: (B,N,Cin), A_norm: (N,N)
        x = torch.matmul(A_norm, x)  # (B,N,Cin)  A_norm broadcast to batch
        return self.lin(x)           # (B,N,Cout)


class AdaptiveAdjacency(nn.Module):
    """
    Graph WaveNet 风格自适应邻接：
      A_adapt = softmax(ReLU(E1 E2^T)) + I
    注意：这是“可学习图”，不是严格的时变图（但常用于写动态图/自适应图）。
    """
    def __init__(self, num_nodes: int, emb_dim: int = 10):
        super().__init__()
        self.E1 = nn.Parameter(torch.randn(num_nodes, emb_dim) * 0.1)
        self.E2 = nn.Parameter(torch.randn(num_nodes, emb_dim) * 0.1)

    def forward(self) -> torch.Tensor:
        A = F.relu(self.E1 @ self.E2.t())      # (N,N)
        A = F.softmax(A, dim=1)                # row-softmax
        I = torch.eye(A.size(0), device=A.device, dtype=A.dtype)
        return A + I


class TemporalBlock(nn.Module):
    """
    轻量 TCN block（dilated causal conv）
    输入: (B, C, T)
    输出: (B, C, T)
    """
    def __init__(self, channels: int, kernel_size: int = 3, dilation: int = 1, dropout: float = 0.1):
        super().__init__()
        pad = (kernel_size - 1) * dilation
        self.conv1 = nn.Conv1d(channels, channels, kernel_size, padding=pad, dilation=dilation)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size, padding=pad, dilation=dilation)
        self.dropout = nn.Dropout(dropout)
        self.pad = pad

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # causal: 裁掉右侧 padding
        y = self.conv1(x)
        if self.pad > 0:
            y = y[..., :-self.pad]
        y = F.relu(y)
        y = self.dropout(y)

        y = self.conv2(y)
        if self.pad > 0:
            y = y[..., :-self.pad]
        y = F.relu(y)
        y = self.dropout(y)

        return x + y  # residual





# 你的 build_grid_adj / normalize_adj / GraphConv / AdaptiveAdjacency / TemporalBlock 保持不变

class GSTCN(nn.Module):
    """
    Graph (GCN / Adaptive) + Temporal (TCN) + Multi-output head
    输入:  (B, T, 1, H, W)
    输出:  (B, pred_len, H, W)   —— 注意：输出的是 “delta” 还是 “absolute”，由你训练脚本决定
    """
    def __init__(
        self,
        pred_len: int,
        close_size: int,
        grid_h: int = 20,
        grid_w: int = 20,
        gcn_dim: int = 32,          # 先用可跑的
        tcn_layers: int = 4,
        tcn_kernel: int = 3,
        dropout: float = 0.1,
        use_adaptive_adj: bool = True,
        lambda_geo: float = 0.5,    # 建议 0.5 起步，后面再试 0.3/0.7
        neighbor_k: int = 8,
        emb_dim: int = 10,
        pool_type: str = "attn"     # ✅ 推荐：attn；可选 "mean"/"max"/"last"
    ):
        super().__init__()
        assert pool_type in ("attn", "mean", "max", "last")

        self.pred_len = pred_len
        self.close_size = close_size
        self.h = grid_h
        self.w = grid_w
        self.N = grid_h * grid_w
        self.use_adaptive_adj = use_adaptive_adj
        self.lambda_geo = lambda_geo
        self.pool_type = pool_type

        # 固定地理邻接
        A_geo = build_grid_adj(grid_h, grid_w, k=neighbor_k)
        self.register_buffer("A_geo_norm", normalize_adj(A_geo))  # (N,N)

        # 自适应邻接（可学习图）
        if use_adaptive_adj:
            self.adapt = AdaptiveAdjacency(self.N, emb_dim=emb_dim)

        # 空间 GCN
        self.gcn1 = GraphConv(1, gcn_dim)
        self.gcn2 = GraphConv(gcn_dim, gcn_dim)

        # 时间 TCN
        blocks = []
        for i in range(tcn_layers):
            blocks.append(TemporalBlock(gcn_dim, kernel_size=tcn_kernel, dilation=2**i, dropout=dropout))
        self.tcn = nn.Sequential(*blocks)

        # ✅ 时间注意力池化：给每个时间点一个权重
        # 输入 z: (B*N, C, T) -> 生成权重 w: (B*N, 1, T)
        self.time_attn = nn.Conv1d(gcn_dim, 1, kernel_size=1)

        # 输出头
        self.out_proj = nn.Conv1d(gcn_dim, pred_len, kernel_size=1)

    def _get_A_norm(self) -> torch.Tensor:
        if not self.use_adaptive_adj:
            return self.A_geo_norm
        A_adapt = self.adapt()                  # (N,N)
        A_adapt_norm = normalize_adj(A_adapt)   # (N,N)
        return self.lambda_geo * self.A_geo_norm + (1 - self.lambda_geo) * A_adapt_norm

    def _time_pool(self, z: torch.Tensor) -> torch.Tensor:
        """
        z: (B*N, C, T) -> (B*N, C, 1)
        """
        if self.pool_type == "last":
            return z[:, :, -1:].contiguous()
        if self.pool_type == "max":
            return z.max(dim=-1, keepdim=True).values.contiguous()
        if self.pool_type == "mean":
            return z.mean(dim=-1, keepdim=True).contiguous()

        # ✅ attn pooling
        w = self.time_attn(z)                  # (B*N, 1, T)
        w = torch.softmax(w, dim=-1)           # 时间维归一化
        z_pool = (z * w).sum(dim=-1, keepdim=True)
        return z_pool.contiguous()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, T, 1, H, W) -> (B, pred_len, H, W)
        """
        B, T, C, H, W = x.shape
        assert C == 1
        assert H == self.h and W == self.w

        # (B,T,1,H,W) -> (B,T,N,1)
        x = x.reshape(B, T, 1, self.N).permute(0, 1, 3, 2).contiguous()

        A_norm = self._get_A_norm()  # (N,N)

        # ✅ 向量化空间图卷积： (B*T, N, 1) 一次性过两层 GCN
        xt = x.reshape(B * T, self.N, 1)
        ht = F.relu(self.gcn1(xt, A_norm))
        ht = F.relu(self.gcn2(ht, A_norm))  # (B*T, N, gcn_dim)
        z = ht.reshape(B, T, self.N, -1)    # (B, T, N, gcn_dim)

        # (B,T,N,C) -> (B,N,C,T) -> (B*N,C,T)
        z = z.permute(0, 2, 3, 1).contiguous().reshape(B * self.N, -1, T)

        z = self.tcn(z)                     # (B*N, gcn_dim, T)
        z_pool = self._time_pool(z)         # (B*N, gcn_dim, 1)

        out = self.out_proj(z_pool)         # (B*N, pred_len, 1)
        out = out.squeeze(-1).reshape(B, self.N, self.pred_len)     # (B, N, pred_len)
        out = out.permute(0, 2, 1).contiguous().reshape(B, self.pred_len, self.h, self.w)
        return out
