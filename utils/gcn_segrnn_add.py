# utils/gcn_segrnn_add.py
# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F


def build_grid_adj(h: int = 20, w: int = 20, k: int = 8) -> torch.Tensor:
    """构建 hxw 栅格的邻接矩阵 A (N,N)。k=4/8 邻域。"""
    assert k in (4, 8)
    N = h * w
    A = torch.zeros(N, N, dtype=torch.float32)

    def idx(r, c):
        return r * w + c

    for r in range(h):
        for c in range(w):
            u = idx(r, c)
            nbrs = []
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                rr, cc = r + dr, c + dc
                if 0 <= rr < h and 0 <= cc < w:
                    nbrs.append((rr, cc))
            if k == 8:
                for dr, dc in [(-1, -1), (-1, 1), (1, -1), (1, 1)]:
                    rr, cc = r + dr, c + dc
                    if 0 <= rr < h and 0 <= cc < w:
                        nbrs.append((rr, cc))

            for rr, cc in nbrs:
                v = idx(rr, cc)
                A[u, v] = 1.0
                A[v, u] = 1.0

    A.fill_diagonal_(1.0)  # self-loop
    return A


def normalize_adj(A: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """D^{-1/2} A D^{-1/2}"""
    deg = A.sum(dim=1)
    deg_inv_sqrt = torch.pow(deg + eps, -0.5)
    D_inv_sqrt = torch.diag(deg_inv_sqrt)
    return D_inv_sqrt @ A @ D_inv_sqrt


class GraphConv(nn.Module):
    """基础 GCN: H' = A_norm H W"""
    def __init__(self, cin: int, cout: int, bias: bool = True):
        super().__init__()
        self.lin = nn.Linear(cin, cout, bias=bias)

    def forward(self, x: torch.Tensor, A_norm: torch.Tensor) -> torch.Tensor:
        # x: (B,N,Cin), A_norm: (N,N)
        x = torch.matmul(A_norm, x)  # broadcast batch
        return self.lin(x)


class AdaptiveAdjacency(nn.Module):
    """Graph WaveNet 风格自适应邻接：softmax(relu(E1E2^T)) + I"""
    def __init__(self, num_nodes: int, emb_dim: int = 10):
        super().__init__()
        self.E1 = nn.Parameter(torch.randn(num_nodes, emb_dim) * 0.1)
        self.E2 = nn.Parameter(torch.randn(num_nodes, emb_dim) * 0.1)

    def forward(self) -> torch.Tensor:
        A = F.relu(self.E1 @ self.E2.t())
        A = F.softmax(A, dim=1)
        I = torch.eye(A.size(0), device=A.device, dtype=A.dtype)
        return A + I


class SegmentRNN(nn.Module):
    """
    SegRNN-like：把 Tin 按 seg_len 分段 -> 每段做聚合 -> GRU -> decode Tout
    输入:  (B, Tin, N)   (每个节点是一条时间序列)
    输出:  (B, Tout, N)
    """
    def __init__(self, Tin: int, Tout: int, d_model: int = 64, seg_len: int = 6, dropout: float = 0.1):
        super().__init__()
        assert Tin % seg_len == 0, f"Tin={Tin} 必须能被 seg_len={seg_len} 整除"
        self.Tin = Tin
        self.Tout = Tout
        self.seg_len = seg_len
        self.nseg = Tin // seg_len

        self.in_proj = nn.Linear(1, d_model)
        self.gru = nn.GRU(input_size=d_model, hidden_size=d_model, num_layers=1,
                          batch_first=True, dropout=0.0)
        self.drop = nn.Dropout(dropout)
        self.dec = nn.Linear(d_model, Tout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, Tin, N)
        B, T, N = x.shape
        x = x.permute(0, 2, 1).contiguous()          # (B, N, Tin)
        x = x.view(B * N, T, 1)                      # (B*N, Tin, 1)

        # 分段： (B*N, nseg, seg_len, 1) -> 段内均值 (B*N, nseg, 1)
        x = x.view(B * N, self.nseg, self.seg_len, 1).mean(dim=2)  # (B*N, nseg, 1)

        z = self.in_proj(x)                           # (B*N, nseg, d_model)
        z = self.drop(z)
        out, h = self.gru(z)                          # h: (1, B*N, d_model)
        h = h[-1]                                     # (B*N, d_model)
        h = self.drop(h)
        y = self.dec(h)                               # (B*N, Tout)
        y = y.view(B, N, self.Tout).permute(0, 2, 1).contiguous()  # (B, Tout, N)
        return y


class GCNBranch(nn.Module):
    """
    GCN 空间分支：逐时间步做 GCN -> 时间聚合 -> 预测 Tout
    输入:  (B, Tin, N)
    输出:  (B, Tout, N)
    """
    def __init__(
        self,
        Tin: int,
        Tout: int,
        N: int,
        gcn_dim: int = 32,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.Tin = Tin
        self.Tout = Tout
        self.N = N

        self.gcn1 = GraphConv(1, gcn_dim)
        self.gcn2 = GraphConv(gcn_dim, gcn_dim)
        self.drop = nn.Dropout(dropout)

        # 时间注意力池化：对 (Tin) 做加权
        self.time_attn = nn.Linear(gcn_dim, 1)
        self.readout = nn.Linear(gcn_dim, Tout)

    def forward(self, x: torch.Tensor, A_norm: torch.Tensor) -> torch.Tensor:
        # x: (B, Tin, N)
        B, T, N = x.shape
        xt = x.permute(0, 1, 2).contiguous().view(B * T, N, 1)    # (B*T, N, 1)

        h = F.relu(self.gcn1(xt, A_norm))
        h = self.drop(h)
        h = F.relu(self.gcn2(h, A_norm))
        h = self.drop(h)                                         # (B*T, N, gcn_dim)

        h = h.view(B, T, N, -1)                                   # (B, Tin, N, gcn_dim)

        # 时间注意力：对每个 (B,N) 的 Tin 个时刻加权
        # scores: (B, Tin, N, 1) -> softmax over Tin
        scores = self.time_attn(h)                                # (B, Tin, N, 1)
        alpha = torch.softmax(scores, dim=1)                      # (B, Tin, N, 1)
        h_pool = (h * alpha).sum(dim=1)                           # (B, N, gcn_dim)

        y = self.readout(h_pool)                                  # (B, N, Tout)
        y = y.permute(0, 2, 1).contiguous()                       # (B, Tout, N)
        return y


class GCN_SegRNN_Add(nn.Module):
    """
    并行：
      - GCN 分支提空间依赖
      - SegRNN 分支提时间依赖
    融合：直接相加 y = y_gcn + y_rnn

    输入:  (B, Tin, 1, H, W)
    输出:  (B, Tout, 1, H, W)
    """
    def __init__(
        self,
        Tin: int,
        Tout: int,
        H: int = 20,
        W: int = 20,
        neighbor_k: int = 8,
        use_adaptive_adj: bool = True,
        lambda_geo: float = 0.5,
        emb_dim: int = 10,
        gcn_dim: int = 32,
        rnn_d_model: int = 64,
        seg_len: int = 6,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.Tin, self.Tout = Tin, Tout
        self.H, self.W = H, W
        self.N = H * W
        self.use_adaptive_adj = use_adaptive_adj
        self.lambda_geo = lambda_geo

        # 固定地理邻接
        A_geo = build_grid_adj(H, W, k=neighbor_k)
        self.register_buffer("A_geo_norm", normalize_adj(A_geo))

        # 自适应邻接
        if use_adaptive_adj:
            self.adapt = AdaptiveAdjacency(self.N, emb_dim=emb_dim)

        self.gcn_branch = GCNBranch(Tin=Tin, Tout=Tout, N=self.N, gcn_dim=gcn_dim, dropout=dropout)
        self.rnn_branch = SegmentRNN(Tin=Tin, Tout=Tout, d_model=rnn_d_model, seg_len=seg_len, dropout=dropout)

    def _get_A_norm(self, device, dtype):
        A_geo = self.A_geo_norm.to(device=device, dtype=dtype)
        if not self.use_adaptive_adj:
            return A_geo
        A_adapt = self.adapt().to(device=device, dtype=dtype)
        A_adapt_norm = normalize_adj(A_adapt)
        return self.lambda_geo * A_geo + (1.0 - self.lambda_geo) * A_adapt_norm

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, Tin, 1, H, W)
        B, T, C, H, W = x.shape
        assert T == self.Tin and C == 1 and H == self.H and W == self.W

        # (B,T,1,H,W) -> (B,T,N)
        xN = x.view(B, T, 1, self.N).squeeze(2)  # (B,T,N)

        A_norm = self._get_A_norm(device=x.device, dtype=x.dtype)

        y_gcn = self.gcn_branch(xN, A_norm)      # (B,Tout,N)
        y_rnn = self.rnn_branch(xN)              # (B,Tout,N)

        y = y_gcn + y_rnn                        # (B,Tout,N)
        y = y.view(B, self.Tout, H, W).contiguous()  # (B,Tout,1,H,W)
        # y = y.view(B, self.Tout, H, W).unsqueeze(2).contiguous()  # (B,Tout,1,H,W)
        return y
