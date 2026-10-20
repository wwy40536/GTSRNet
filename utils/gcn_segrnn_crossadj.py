# utils/gcn_segrnn_crossadj.py
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


def normalize_adj_batch(A: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    Batched normalize:  D^{-1/2} A D^{-1/2}
    A: (B, N, N)
    """
    deg = A.sum(dim=-1)  # (B, N)
    deg_inv_sqrt = torch.pow(deg + eps, -0.5)  # (B, N)
    D_left = deg_inv_sqrt.unsqueeze(-1)        # (B, N, 1)
    D_right = deg_inv_sqrt.unsqueeze(-2)       # (B, 1, N)
    return A * D_left * D_right                # (B, N, N)


class GraphConv(nn.Module):
    """
    简单 GCN: H' = A_norm @ H @ W
    H: (B,T,N,Cin) 或 (B,N,Cin)
    """
    def __init__(self, cin: int, cout: int, bias: bool = True):
        super().__init__()
        self.lin = nn.Linear(cin, cout, bias=bias)

    def forward(self, x: torch.Tensor, A_norm: torch.Tensor) -> torch.Tensor:
        # 支持两种形状： (B,T,N,C) 或 (B,N,C)
        if x.dim() == 4:
            # (B,T,N,Cin)
            x = torch.einsum('bij,btjc->btic', A_norm, x)  # (B,T,N,Cin)
            return self.lin(x)
        else:
            # (B,N,Cin)
            x = torch.einsum('bij,bjc->bic', A_norm, x)    # (B,N,Cin)
            return self.lin(x)


class GCN_SegRNN_CrossAdj(nn.Module):
    """
    ✅ 你要的：GCN(空间) + GRU(时间，简化版SegRNN) + cross-adj 融合
    输入:
      x:     (B, T, 1, H, W)
      cross: (B, Cx=4, H, W)
    输出:
      y:     (B, pred_len, 1, H, W)
    """
    def __init__(
        self,
        close_size: int = 25,
        pred_len: int = 8,
        H: int = 20,
        W: int = 20,
        cross_dim: int = 4,
        gcn_hidden: int = 32,
        rnn_hidden: int = 64,
        gcn_layers: int = 2,
        dropout: float = 0.1,
        lambda_geo: float = 0.5,   # geo vs cross
        neighbor_k: int = 8,
        cross_l2norm: bool = True, # 仅做 L2 normalize（避免尺度2707干扰）
        sim_relu: bool = True,     # 让相似度非负
    ):
        super().__init__()
        self.close_size = close_size
        self.pred_len = pred_len
        self.H = H
        self.W = W
        self.N = H * W
        self.cross_dim = cross_dim
        self.lambda_geo = lambda_geo
        self.cross_l2norm = cross_l2norm
        self.sim_relu = sim_relu

        # 固定几何邻接（归一化后作为 buffer）
        A_geo = build_grid_adj(H, W, k=neighbor_k)  # (N,N)
        A_geo = A_geo / (A_geo.sum(dim=-1, keepdim=True) + 1e-8)  # 简单行归一化也行
        self.register_buffer("A_geo", A_geo)  # (N,N)

        # ---------- 空间编码：GCN ----------
        self.gcn_in = GraphConv(1, gcn_hidden)
        self.gcn_mid = nn.ModuleList()
        for _ in range(max(0, gcn_layers - 1)):
            self.gcn_mid.append(GraphConv(gcn_hidden, gcn_hidden))
        self.gcn_drop = nn.Dropout(dropout)

        # ---------- 时间编码：GRU（简化版 SegRNN） ----------
        # 以 node 为单位：输入 (B*N, T, gcn_hidden) -> 输出 hidden -> 预测 pred_len
        self.gru = nn.GRU(
            input_size=gcn_hidden,
            hidden_size=rnn_hidden,
            num_layers=1,
            batch_first=True,
            bidirectional=False
        )
        self.tem_drop = nn.Dropout(dropout)
        self.head = nn.Linear(rnn_hidden, pred_len)

    def _build_cross_adj(self, cross: torch.Tensor) -> torch.Tensor:
        """
        cross: (B, Cx, H, W) -> A_cross: (B, N, N)
        用“节点外部特征相似度”构图：A_ij = cos(ci, cj)
        """
        B = cross.size(0)
        # (B, Cx, H, W) -> (B, N, Cx)
        C = cross.view(B, self.cross_dim, self.N).permute(0, 2, 1).contiguous()

        if self.cross_l2norm:
            C = F.normalize(C, p=2, dim=-1)  # (B,N,Cx)

        # cos-sim = C @ C^T
        A = torch.bmm(C, C.transpose(1, 2))  # (B,N,N)

        if self.sim_relu:
            A = F.relu(A)

        # self-loop
        I = torch.eye(self.N, device=A.device, dtype=A.dtype).unsqueeze(0)
        A = A + I

        # 归一化
        A = normalize_adj_batch(A)
        return A

    def forward(self, x: torch.Tensor, cross: torch.Tensor = None) -> torch.Tensor:
        """
        x: (B,T,1,H,W)
        cross: (B,4,H,W)
        """
        B, T, C, H, W = x.shape
        assert C == 1, f"expect x channel=1, got {C}"
        assert H == self.H and W == self.W, f"expect {(self.H,self.W)}, got {(H,W)}"

        # 1) 建图：A = lambda*A_geo + (1-lambda)*A_cross
        if cross is None:
            A_norm = self.A_geo.unsqueeze(0).repeat(B, 1, 1)  # (B,N,N)
        else:
            A_cross = self._build_cross_adj(cross)            # (B,N,N)
            A_geo = self.A_geo.unsqueeze(0).repeat(B, 1, 1)   # (B,N,N)
            # A_geo 已是行归一/近似归一，A_cross 是 D^{-1/2} A D^{-1/2}
            A_norm = self.lambda_geo * A_geo + (1.0 - self.lambda_geo) * A_cross

        # 2) reshape: (B,T,1,H,W) -> (B,T,N,1)
        xt = x.view(B, T, 1, self.N).permute(0, 1, 3, 2).contiguous()  # (B,T,N,1)

        # 3) 空间 GCN
        h = F.relu(self.gcn_in(xt, A_norm))    # (B,T,N,gcn_hidden)
        h = self.gcn_drop(h)
        for layer in self.gcn_mid:
            h = F.relu(layer(h, A_norm))
            h = self.gcn_drop(h)

        # 4) 时间 GRU：按 node 展开
        # (B,T,N,C) -> (B,N,T,C) -> (B*N,T,C)
        h = h.permute(0, 2, 1, 3).contiguous().view(B * self.N, T, -1)  # (B*N,T,gcn_hidden)

        out_seq, h_last = self.gru(h)  # h_last: (1, B*N, rnn_hidden)
        h_last = h_last.squeeze(0)     # (B*N, rnn_hidden)
        h_last = self.tem_drop(h_last)

        # 5) head：每个 node 输出 pred_len
        y = self.head(h_last)          # (B*N, pred_len)

        # 6) reshape 回网格
        y = y.view(B, self.N, self.pred_len).permute(0, 2, 1).contiguous()  # (B,pred_len,N)
        y = y.view(B, self.pred_len, self.H, self.W).unsqueeze(2)           # (B,pred_len,1,H,W)
        return y
