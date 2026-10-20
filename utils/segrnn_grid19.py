# utils/segrnn_grid.py
# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F

# =========================================================
# 1) Fixed grid adjacency (4/8-neighbor) + normalization
# =========================================================
def build_grid_adj(H: int = 20, W: int = 20, k: int = 8, self_loop: bool = True, device=None):
    """
    Fixed geographic adjacency (4/8-neighbor) with symmetric normalization.
    Return: A_geo (N,N) = D^{-1/2} A D^{-1/2}
    """
    assert k in (4, 8)
    N = H * W
    A = torch.zeros(N, N, dtype=torch.float32, device=device)

    def idx(r, c): return r * W + c

    nbr4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    nbr8 = nbr4 + [(-1, -1), (-1, 1), (1, -1), (1, 1)]
    nbrs = nbr8 if k == 8 else nbr4

    for r in range(H):
        for c in range(W):
            u = idx(r, c)
            for dr, dc in nbrs:
                rr, cc = r + dr, c + dc
                if 0 <= rr < H and 0 <= cc < W:
                    v = idx(rr, cc)
                    A[u, v] = 1.0

    if self_loop:
        A.fill_diagonal_(1.0)

    return normalize_adj_sym(A)


def normalize_adj_sym(A: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    Symmetric normalization: D^{-1/2} A D^{-1/2}
    A: (N,N)
    """
    deg = A.sum(-1).clamp(min=eps)
    D_inv_sqrt = deg.pow(-0.5)
    return (D_inv_sqrt[:, None] * A) * D_inv_sqrt[None, :]


# =========================================================
# 2) Learnable correlation adjacency (small MLP) + topk
# =========================================================
class LearnableCorrAdj(nn.Module):
    """
    Build data-driven adjacency from input sequence using a small MLP (learnable),
    instead of Pearson correlation.

    Input: x (B,Tin,1,H,W)
    Steps:
      - flatten -> (B,T,N)
      - optional use last K frames
      - per-node stats: mean/std/last (B,N,3)
      - small MLP -> node embedding E (B,N,dk)
      - aggregate across batch (mean) -> (N,dk)
      - cosine similarity -> S (N,N)
      - topk sparsify per row
      - softmax row-normalize (optional) + self-loop
    Output: A_corr (N,N) dense but topk-sparse
    """
    def __init__(
        self,
        H=20, W=20,
        Tin=24,
        dk=64,
        topk=8,
        tau=0.2,
        use_last_k=0,          # 0 means use all Tin
        use_softmax=True,      # True: attention-style weights; False: raw positive sim
        self_loop=True,
        eps=1e-6
    ):
        super().__init__()
        self.H, self.W, self.N = H, W, H * W
        self.Tin = Tin
        self.dk = dk
        self.topk = topk
        self.tau = nn.Parameter(torch.tensor(float(tau)))
        self.use_last_k = int(use_last_k)
        self.use_softmax = bool(use_softmax)
        self.self_loop = bool(self_loop)
        self.eps = eps

        # small MLP on (mean, std, last) -> dk
        self.mlp = nn.Sequential(
            nn.Linear(3, dk),
            nn.GELU(),
            nn.Linear(dk, dk),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B,Tin,1,H,W)
        return A_corr: (N,N)
        """
        B, Tin, C, H, W = x.shape
        assert C == 1 and H == self.H and W == self.W
        if Tin != self.Tin:
            # allow mismatch if你 train里 Tin 变了，但最好保持一致
            self.Tin = Tin

        N = self.N
        xt = x.squeeze(2).reshape(B, Tin, N)  # (B,T,N)
        if self.use_last_k and self.use_last_k > 0:
            xt = xt[:, -min(self.use_last_k, Tin):, :]  # (B,Tk,N)

        # stats over time
        mean = xt.mean(dim=1)                 # (B,N)
        std = xt.std(dim=1, unbiased=False)   # (B,N)
        last = xt[:, -1, :]                   # (B,N)

        feat = torch.stack([mean, std, last], dim=-1)  # (B,N,3)
        E = self.mlp(feat)                              # (B,N,dk)

        # aggregate across batch -> stable graph (N,dk)
        E = E.mean(dim=0)                               # (N,dk)
        E = F.normalize(E, dim=-1)                      # cosine

        # similarity
        S = E @ E.t()                                   # (N,N)
        S.fill_diagonal_(-1e9)                           # exclude self for topk

        k = min(self.topk, N - 1)
        vals, idx = torch.topk(S, k=k, dim=-1)          # (N,k)

        A = torch.full_like(S, -1e9)                    # masked logits
        A.scatter_(dim=-1, index=idx, src=vals)

        if self.use_softmax:
            A = F.softmax(A / (self.tau.abs() + self.eps), dim=-1)  # row-stochastic
        else:
            # keep only positive sims (after exp if you want stronger)
            A = torch.where(A > -1e8, A, torch.zeros_like(A))
            A = A.clamp(min=0.0)
            A = A / (A.sum(dim=-1, keepdim=True) + self.eps)

        if self.self_loop:
            I = torch.eye(N, device=A.device, dtype=A.dtype)
            A = A + I
            A = A / (A.sum(dim=-1, keepdim=True) + self.eps)

        return A


# =========================================================
# 3) Spatial branch: GCN + lightweight global attention -> gate
#     Use mixed graph: A_mix = (1-lam)*A_geo + lam*A_corr
# =========================================================
class GCNLayer(nn.Module):
    def __init__(self, d_in: int, d_out: int, dropout: float = 0.1):
        super().__init__()
        self.lin = nn.Linear(d_in, d_out)
        self.drop = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_out)

    def forward(self, x: torch.Tensor, A: torch.Tensor) -> torch.Tensor:
        # x: (B,N,d), A: (N,N)
        x = torch.einsum('nm,bmd->bnd', A, x)
        x = self.lin(x)
        x = F.gelu(x)
        x = self.drop(x)
        x = self.norm(x)
        return x


class GlobalAttentionInject(nn.Module):
    """
    Global attention across all nodes to inject global dependency.
    """
    def __init__(self, d: int, dropout: float = 0.1):
        super().__init__()
        self.q = nn.Linear(d, d, bias=False)
        self.k = nn.Linear(d, d, bias=False)
        self.v = nn.Linear(d, d, bias=False)
        self.gate = nn.Sequential(
            nn.Linear(2 * d, d),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d, d),
            nn.Sigmoid()
        )
        self.drop = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x0 = self.norm(x)
        Q = self.q(x0); K = self.k(x0); V = self.v(x0)
        attn = torch.softmax(torch.einsum('bnd,bmd->bnm', Q, K) / (x0.shape[-1] ** 0.5), dim=-1)
        ctx = torch.einsum('bnm,bmd->bnd', attn, V)
        g = self.gate(torch.cat([x0, ctx], dim=-1))
        return x + self.drop(g * ctx)


class SpatialGateFromLastFrameMixedGraph(nn.Module):
    """
    Spatial gate from last frame using mixed graph:
      A_mix = normalize_sym((1-lam)*A_geo + lam*A_corr)

    Output: g (B,N,1) in [0,1]
    """
    def __init__(
        self,
        Tin=24, H=20, W=20,
        d_model=128,
        geo_k=8,
        gcn_layers=2,
        dropout=0.1,
        mix_lam=0.3,          # λ
        corr_topk=8,
        corr_use_last_k=0,
        corr_dk=64,
        corr_tau=0.2,
        corr_use_softmax=True
    ):
        super().__init__()
        self.H, self.W, self.N = H, W, H * W
        self.mix_lam = float(mix_lam)

        self.in_proj = nn.Linear(1, d_model)
        self.gcn = nn.ModuleList([GCNLayer(d_model, d_model, dropout=dropout) for _ in range(gcn_layers)])
        self.gattn = GlobalAttentionInject(d_model, dropout=dropout)

        self.to_gate = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1),
            nn.Sigmoid()
        )

        # fixed geographic graph
        A_geo = build_grid_adj(H, W, k=geo_k, self_loop=True, device=None)
        self.register_buffer("A_geo", A_geo)

        # learnable correlation graph builder
        self.corr = LearnableCorrAdj(
            H=H, W=W, Tin=Tin, dk=corr_dk, topk=corr_topk, tau=corr_tau,
            use_last_k=corr_use_last_k, use_softmax=corr_use_softmax, self_loop=True
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B,Tin,1,H,W)
        B, Tin, _, H, W = x.shape

        # build A_corr from x (learnable)
        A_corr = self.corr(x)  # (N,N)

        # mix + re-normalize for stability
        A_geo = self.A_geo.to(x.device)
        A_mix = (1.0 - self.mix_lam) * A_geo + self.mix_lam * A_corr
        A_mix = normalize_adj_sym(A_mix)

        # last frame node feature
        x_last = x[:, -1].reshape(B, 1, self.N).permute(0, 2, 1).contiguous()  # (B,N,1)
        h = self.in_proj(x_last)  # (B,N,d)

        for layer in self.gcn:
            h = layer(h, A_mix)
        h = self.gattn(h)

        g = self.to_gate(h)  # (B,N,1)
        return g


# =========================================================
# 4) Temporal branch: SegRNN（保留你最初的 Seg 思路 + row/col embedding）
# =========================================================
class TemporalSegRNN(nn.Module):
    """
    输入:  x (B,Tin,1,H,W)
    输出:  y (B,Tout,1,H,W)
    """
    def __init__(self, Tin=24, Tout=12, H=20, W=20, d_model=128, seg_len=6, dropout=0.1):
        super().__init__()
        assert Tin % seg_len == 0 and Tout % seg_len == 0
        self.Tin, self.Tout, self.H, self.W = Tin, Tout, H, W
        self.N = H * W
        self.d_model = d_model
        self.seg_len = seg_len
        self.seg_num_x = Tin // seg_len
        self.seg_num_y = Tout // seg_len

        self.valueEmbedding = nn.Sequential(
            nn.Linear(seg_len, d_model),
            nn.ReLU(inplace=True)
        )

        self.rnn = nn.GRU(
            input_size=d_model,
            hidden_size=d_model,
            num_layers=1,
            batch_first=True
        )

        # 2D factorized channel embedding
        self.row_emb = nn.Parameter(torch.randn(self.H, d_model // 4) * 0.02)
        self.col_emb = nn.Parameter(torch.randn(self.W, d_model // 4) * 0.02)
        self.channel_fuse = nn.Sequential(
            nn.Linear(d_model // 2, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, d_model // 2),
        )

        self.pos_emb = nn.Parameter(torch.randn(self.seg_num_y, d_model // 2) * 0.02)

        # 预测每个 token 的 seg_len
        self.predict = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(d_model, seg_len)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, Tin, C, H, W = x.shape
        assert Tin == self.Tin and C == 1 and H == self.H and W == self.W

        x = x.squeeze(2).reshape(B, Tin, self.N)          # (B,T,N)
        seq_last = x[:, -1:, :].detach()                  # (B,1,N)
        x = x - seq_last                                  # (B,T,N)

        x = x.permute(0, 2, 1).contiguous()               # (B,N,T)
        x_seg = x.view(B * self.N, self.seg_num_x, self.seg_len)  # (B*N,n,seg_len)
        x_emb = self.valueEmbedding(x_seg)                # (B*N,n,d)
        _, hn = self.rnn(x_emb)                           # (1,B*N,d)

        # cell emb
        row = self.row_emb[:, None, :].expand(self.H, self.W, -1)
        col = self.col_emb[None, :, :].expand(self.H, self.W, -1)
        cell = torch.cat([row, col], dim=-1).reshape(self.N, -1)   # (N,d/2)
        cell = self.channel_fuse(cell)                             # (N,d/2)

        # decoder tokens
        pos = self.pos_emb.unsqueeze(0).repeat(self.N, 1, 1)       # (N,m,d/2)
        chn = cell.unsqueeze(1).repeat(1, self.seg_num_y, 1)       # (N,m,d/2)
        dec_tok = torch.cat([pos, chn], dim=-1).view(-1, 1, self.d_model)  # (N*m,1,d)
        dec_tok = dec_tok.repeat(B, 1, 1)                          # (B*N*m,1,d)

        hn_rep = hn.repeat(1, 1, self.seg_num_y).view(1, -1, self.d_model)  # (1,B*N*m,d)
        _, hy = self.rnn(dec_tok, hn_rep)                           # (1,B*N*m,d)

        y_seg = self.predict(hy)                                    # (1,B*N*m,seg_len)
        y_seg = y_seg.view(B, self.N, self.seg_num_y, self.seg_len) # (B,N,m,seg_len)
        y = y_seg.reshape(B, self.N, self.Tout)                     # (B,N,Tout)

        y = y.permute(0, 2, 1).contiguous() + seq_last              # (B,Tout,N)
        y = y.view(B, self.Tout, 1, self.H, self.W)
        return y


# =========================================================
# 5) Serial model: Spatial gate -> reweight input sequence -> SegRNN
# =========================================================
class STSegRNN_SerialSpatial2Temporal(nn.Module):
    """
    串行：空间分支只看最后一帧，输出空间门控 g(B,N,1)，
          用 g 对整段历史序列做重标定，再送进 SegRNN。
    输出: (B,Tout,1,H,W)
    """
    def __init__(
        self,
        Tin=24, Tout=12, H=20, W=20,
        d_model=128,
        seg_len=6,
        geo_k=8,
        dropout=0.1,
        gate_strength=0.5,
        # mixed graph (geo + learnable corr)
        mix_lam=0.3,
        corr_topk=8,
        corr_use_last_k=0,
        corr_dk=64,
        corr_tau=0.2,
        corr_use_softmax=True,
        gcn_layers=2
    ):
        super().__init__()
        self.Tin, self.Tout, self.H, self.W = Tin, Tout, H, W
        self.N = H * W
        self.gate_strength = gate_strength

        self.spatial_gate = SpatialGateFromLastFrameMixedGraph(
            Tin=Tin, H=H, W=W,
            d_model=d_model,
            geo_k=geo_k,
            gcn_layers=gcn_layers,
            dropout=dropout,
            mix_lam=mix_lam,
            corr_topk=corr_topk,
            corr_use_last_k=corr_use_last_k,
            corr_dk=corr_dk,
            corr_tau=corr_tau,
            corr_use_softmax=corr_use_softmax
        )

        self.temporal = TemporalSegRNN(
            Tin=Tin, Tout=Tout, H=H, W=W,
            d_model=d_model, seg_len=seg_len, dropout=dropout
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B,Tin,1,H,W)
        B, Tin, C, H, W = x.shape
        assert Tin == self.Tin and C == 1 and H == self.H and W == self.W

        g = self.spatial_gate(x)  # (B,N,1) in [0,1]

        # broadcast to all history steps
        g_bn = g.squeeze(-1).unsqueeze(1)  # (B,1,N)

        # x -> (B,T,N)
        x_flat = x.squeeze(2).reshape(B, Tin, self.N)

        # stable reweighting
        # scale centered around 1.0
        scale = 1.0 + self.gate_strength * (g_bn - 0.5)  # (B,1,N)
        x_flat = x_flat * scale                           # (B,T,N)

        # back to (B,T,1,H,W)
        x_in = x_flat.view(B, Tin, 1, H, W)

        # SegRNN prediction
        y = self.temporal(x_in)  # (B,Tout,1,H,W)
        return y
