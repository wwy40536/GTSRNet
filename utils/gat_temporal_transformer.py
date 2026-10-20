# utils/gat_temporal_transformer.py
# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F


def build_grid_edges(h: int, w: int, self_loop: bool = True, device=None):
    """
    20x20 栅格 4-neighbor edge_index (2,E)，适合 GAT。
    节点 idx = r*w + c
    """
    edges = []

    def idx(r, c):
        return r * w + c

    for r in range(h):
        for c in range(w):
            u = idx(r, c)
            # 4-neighbor
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                rr, cc = r + dr, c + dc
                if 0 <= rr < h and 0 <= cc < w:
                    v = idx(rr, cc)
                    edges.append((u, v))

    if self_loop:
        for n in range(h * w):
            edges.append((n, n))

    edge_index = torch.tensor(edges, dtype=torch.long, device=device).t().contiguous()  # (2,E)
    return edge_index


class GATLayer(nn.Module):
    """
    纯 PyTorch 版 GAT（不依赖 torch_geometric），适合 N=400 的栅格。
    输入:  x (B, N, Fin)
    输出:  (B, N, Fout)  (多头 concat 或 mean)
    """
    def __init__(self, fin: int, fout: int, heads: int = 4, dropout: float = 0.1, concat: bool = True):
        super().__init__()
        self.fin = fin
        self.fout = fout
        self.heads = heads
        self.concat = concat
        self.dropout = dropout

        # 线性变换：Fin -> heads*Fout
        self.W = nn.Linear(fin, heads * fout, bias=False)

        # 注意力参数 a = [a_src || a_dst]，每个 head 一个
        self.a_src = nn.Parameter(torch.empty(heads, fout))
        self.a_dst = nn.Parameter(torch.empty(heads, fout))
        nn.init.xavier_uniform_(self.a_src)
        nn.init.xavier_uniform_(self.a_dst)

        self.leaky = nn.LeakyReLU(0.2)
        self.out_norm = nn.LayerNorm(heads * fout if concat else fout)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """
        x: (B,N,Fin)
        edge_index: (2,E)
        """
        B, N, _ = x.shape
        src, dst = edge_index[0], edge_index[1]   # (E,), (E,)
        E = src.numel()

        # 1) 线性投影
        h = self.W(x)  # (B,N,heads*fout)
        h = h.view(B, N, self.heads, self.fout)  # (B,N,H,F)

        # 2) 计算边注意力 e_ij
        h_src = h[:, src]  # (B,E,H,F)
        h_dst = h[:, dst]  # (B,E,H,F)

        # e = a_src·h_i + a_dst·h_j
        e = (h_src * self.a_src[None, None, :, :]).sum(-1) + (h_dst * self.a_dst[None, None, :, :]).sum(-1)  # (B,E,H)
        e = self.leaky(e)

        # 3) 对每个 dst 节点做 softmax（按入边归一化）
        #    用稳定的 scatter-softmax：对每个 dst、每个 head 分组
        #    这里用一个简洁实现：先按 dst 排序再 segment softmax
        order = torch.argsort(dst)
        dst_sorted = dst[order]
        e_sorted = e[:, order]  # (B,E,H)

        # segment softmax
        alpha = torch.empty_like(e_sorted)
        # 找到每个 dst 段的起止
        change = torch.ones_like(dst_sorted, dtype=torch.bool)
        change[1:] = dst_sorted[1:] != dst_sorted[:-1]
        idx_start = torch.nonzero(change).flatten()
        idx_end = torch.cat([idx_start[1:], torch.tensor([E], device=idx_start.device)])

        for s, t in zip(idx_start.tolist(), idx_end.tolist()):
            seg = e_sorted[:, s:t, :]  # (B,segE,H)
            seg = seg - seg.max(dim=1, keepdim=True).values
            seg = torch.exp(seg)
            seg = seg / (seg.sum(dim=1, keepdim=True) + 1e-9)
            alpha[:, s:t, :] = seg

        alpha = F.dropout(alpha, p=self.dropout, training=self.training)

        # 4) 聚合：out[dst] += alpha * h_src
        out = torch.zeros((B, N, self.heads, self.fout), device=x.device, dtype=x.dtype)
        msg = h_src[:, order] * alpha.unsqueeze(-1)  # (B,E,H,F)

        # scatter-add 到 dst_sorted
        # 用 index_add_ 实现（对每个 head/f 维度循环会慢，这里用 reshape 合并维度）
        out2 = out.view(B, N, self.heads * self.fout)
        msg2 = msg.view(B, E, self.heads * self.fout)

        for b in range(B):
            out2[b].index_add_(0, dst_sorted, msg2[b])

        out = out2.view(B, N, self.heads, self.fout)

        # concat 或 mean heads
        if self.concat:
            out = out.reshape(B, N, self.heads * self.fout)  # (B,N,H*F)
        else:
            out = out.mean(dim=2)  # (B,N,F)

        return self.out_norm(out)


class SpatialGAT(nn.Module):
    """
    两层 GAT + 残差
    输入: (B,T,N,D)
    输出: (B,T,N,D)
    """
    def __init__(self, d_model: int, heads: int = 4, dropout: float = 0.1):
        super().__init__()
        # 第一层输出 concat -> d_model
        assert d_model % heads == 0, "d_model 必须能被 heads 整除（方便设 fout）"
        fout = d_model // heads

        self.gat1 = GATLayer(d_model, fout, heads=heads, dropout=dropout, concat=True)
        self.gat2 = GATLayer(d_model, fout, heads=heads, dropout=dropout, concat=True)
        self.drop = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        # x: (B,T,N,D)
        B, T, N, D = x.shape
        y = x.view(B * T, N, D)
        y = F.gelu(self.gat1(y, edge_index))
        y = self.drop(y)
        y = F.gelu(self.gat2(y, edge_index))
        y = self.drop(y)
        y = y.view(B, T, N, D)
        return self.norm(x + y)


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
        B, T, N, D = x.shape
        xn = x.permute(0, 2, 1, 3).contiguous().view(B * N, T, D)
        xn = self.enc(xn)
        return xn.view(B, N, T, D).permute(0, 2, 1, 3).contiguous()


class GAT_TemporalTransformer(nn.Module):
    """
    空间: GAT
    时间: Transformer
    输入:  (B, Tin, 1, H, W)
    输出:  (B, Tout, 1, H, W)
    """
    def __init__(
        self,
        Tin: int = 25,
        Tout: int = 8,
        H: int = 20,
        W: int = 20,
        d_model: int = 32,       # ✅ 先用 32 更快
        gat_heads: int = 4,
        t_heads: int = 4,
        t_layers: int = 2,
        dim_ff: int = 128,
        dropout: float = 0.1,
        self_loop: bool = True,
        use_skip_last: bool = True,
    ):
        super().__init__()
        self.Tin = Tin
        self.Tout = Tout
        self.H = H
        self.W = W
        self.N = H * W
        self.use_skip_last = use_skip_last

        # edges buffer（不训练）
        edge_index = build_grid_edges(H, W, self_loop=self_loop, device=None)
        self.register_buffer("edge_index", edge_index)

        self.in_proj = nn.Linear(1, d_model)

        self.spatial = SpatialGAT(d_model=d_model, heads=gat_heads, dropout=dropout)
        self.temporal = TemporalTransformer(d_model=d_model, nhead=t_heads, num_layers=t_layers, dim_ff=dim_ff, dropout=dropout)

        # last token -> Tout
        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, Tout)
        )

        # 可选：输出空间 refine（很轻）
        self.refine = nn.Conv2d(Tout, Tout, kernel_size=3, padding=1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B,Tin,1,H,W)
        """
        B, T, C, H, W = x.shape
        assert T == self.Tin and C == 1 and H == self.H and W == self.W

        last = x[:, -1, 0]  # (B,H,W)

        # (B,T,1,H,W)->(B,T,H,W,1)->(B,T,N,1)
        x = x.permute(0, 1, 3, 4, 2).contiguous().view(B, T, self.N, 1)

        h = self.in_proj(x)                    # (B,T,N,D)
        h = self.spatial(h, self.edge_index)   # (B,T,N,D)
        h = self.temporal(h)                   # (B,T,N,D)

        h_last = h[:, -1]                      # (B,N,D)
        y = self.head(h_last)                  # (B,N,Tout)
        y = y.permute(0, 2, 1).contiguous().view(B, self.Tout, H, W)  # (B,Tout,H,W)
        y = self.refine(y)

        if self.use_skip_last:
            y = y + last.unsqueeze(1)

        return y.unsqueeze(2)  # (B,Tout,1,H,W)


if __name__ == "__main__":
    m = GAT_TemporalTransformer(Tin=25, Tout=8, H=20, W=20, d_model=32, gat_heads=4, t_heads=4, t_layers=2)
    x = torch.randn(2, 25, 1, 20, 20)
    y = m(x)
    print(x.shape, y.shape)
