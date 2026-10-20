# utils/segrnn_grid.py
# -*- coding: utf-8 -*-
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
# class SpatialDWRefine(nn.Module):
#     """
#     输入/输出: (B, T, H, W)
#     对每个预测步 t 做独立的 depthwise 3x3 空间卷积（groups=T），再残差融合
#     """
#     def __init__(self, T: int, k: int = 3, dropout: float = 0.0):
#         super().__init__()
#         assert k in (3, 5, 7)
#         pad = k // 2
#         self.dw = nn.Conv2d(T, T, kernel_size=k, padding=pad, groups=T, bias=False)
#         self.bn = nn.BatchNorm2d(T)
#         self.act = nn.SiLU()
#         self.drop = nn.Dropout2d(p=dropout) if dropout > 0 else nn.Identity()
#         # 可学习缩放，防止一开始把输出带偏
#         self.gamma = nn.Parameter(torch.tensor(0.1))
#
#     def forward(self, y):
#         # y: (B,T,H,W)
#         r = self.dw(y)
#         r = self.bn(r)
#         r = self.act(r)
#         r = self.drop(r)
#         return y + self.gamma * r


class SegRNNGrid(nn.Module):
    """
    SegRNN（SegGRU）用于网格多步时空预测（带“历史段-未来段 Cross-Attention”增强）

    输入:  x (B, Tin, 1, H, W)
    输出:  y (B, Tout, H, W)

    核心思路：
    1) 展平网格： (H,W)->N=H*W，把每个格点当作一个“变量/节点”
    2) last-value 归一化：预测相对变化，提升稳定性
    3) 分段嵌入：Tin 按 seg_len 切成 seg_num_x 段，每段映射到 d_model
    4) GRU 编码得到 enc_out（所有历史段表示）与 hn（最终隐藏态）
    5) 构造解码 token（pos_emb + 2D factorized cell emb），GRU 解码得到 hy
    6) Cross-Attn：以 hy 为 Query，对 enc_out（历史段）做注意力汇聚，增强长预测
    7) 线性投影到 seg_len，拼回 Tout，反归一化并 reshape 回网格
    """

    def __init__(
        self,
        Tin=25,
        Tout=8,
        H=20,
        W=20,
        d_model=128,
        seg_len=1,
        dropout=0.1,
        use_cross_attn=1,
    ):
        super().__init__()
        assert Tin % seg_len == 0, "Tin 必须能被 seg_len 整除"
        assert Tout % seg_len == 0, "Tout 必须能被 seg_len 整除"

        self.Tin = Tin
        self.Tout = Tout
        self.H = H
        self.W = W
        self.N = H * W
        self.d_model = d_model
        self.seg_len = seg_len
        self.use_cross_attn = use_cross_attn

        self.seg_num_x = Tin // seg_len  # n
        self.seg_num_y = Tout // seg_len # m

        self.feature_expand = nn.Linear(self.d_model, self.seg_len * self.d_model)

        # ===== 2D factorized cell embedding =====
        self.row_emb = nn.Parameter(torch.randn(self.H, d_model // 4) * 0.02)  # (H, d/4)
        self.col_emb = nn.Parameter(torch.randn(self.W, d_model // 4) * 0.02)  # (W, d/4)
        self.channel_fuse = nn.Sequential(
            nn.Linear(d_model // 2, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, d_model // 2),
        )
        # self.spatial_refine = SpatialDWRefine(Tout, k=3, dropout=0.0)

        # ===== segment value embedding (seg_len -> d_model) =====
        self.valueEmbedding = nn.Sequential(
            nn.Linear(seg_len, d_model),
            nn.ReLU(inplace=True),
        )

        # ===== GRU encoder/decoder =====
        self.rnn = nn.GRU(
            input_size=d_model,
            hidden_size=d_model,
            num_layers=1,
            batch_first=True,
            bidirectional=False,
        )

        # ===== decoder token embedding =====
        self.pos_emb = nn.Parameter(torch.randn(self.seg_num_y, d_model // 2) * 0.02)  # (m, d/2)
        # 新增 1D 版本 (独立学习每个节点的嵌入，不包含行列结构信息)
        # self.channel_emb = nn.Parameter(torch.randn(self.N, d_model // 2) * 0.02)

        # ===== Cross-Attention (decoder query attends encoder segments) =====
        if self.use_cross_attn:
            self.q_proj = nn.Linear(d_model, d_model, bias=False)
            self.k_proj = nn.Linear(d_model, d_model, bias=False)
            self.v_proj = nn.Linear(d_model, d_model, bias=False)
            self.ctx_proj = nn.Linear(d_model, d_model, bias=False)
            self.attn_drop = nn.Dropout(dropout)

        # ===== output projection =====
        self.predict = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(d_model, seg_len),
        )

    def _build_cell_emb(self, device, dtype):
        # row: (H, d/4) -> (H, W, d/4)
        # col: (W, d/4) -> (H, W, d/4)
        row = self.row_emb[:, None, :].expand(self.H, self.W, -1)  # (H, W, d/4)
        col = self.col_emb[None, :, :].expand(self.H, self.W, -1)  # (H, W, d/4)

        cell = torch.cat([row, col], dim=-1).reshape(self.N, -1)   # (N, d/2)
        cell = self.channel_fuse(cell)                              # (N, d/2)
        return cell.to(device=device, dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, Tin, 1, H, W)
        return: (B, Tout, H, W)
        """
        B, Tin, C, H, W = x.shape
        assert Tin == self.Tin and C == 1 and H == self.H and W == self.W

        # ===== (B,T,1,H,W) -> (B,T,N) =====
        x = x.squeeze(2).reshape(B, Tin, self.N)  # (B,T,N)

        # ===== last-value normalization =====
        seq_last = x[:, -1:, :].detach()  # (B,1,N)
        x = x - seq_last                  # (B,T,N)

        # ===== (B,T,N) -> (B,N,T) =====
        x = x.permute(0, 2, 1).contiguous()  # (B,N,T)

        # ===== segment: (B,N,T) -> (B*N, n, seg_len) =====
        x_seg = x.view(B * self.N, self.seg_num_x, self.seg_len)  # (B*N,n,seg_len)
        x_emb = self.valueEmbedding(x_seg)                        # (B*N,n,d)

        # ===== GRU encoder: keep all outputs for attention =====
        enc_out, hn = self.rnn(x_emb)  # enc_out: (B*N,n,d), hn: (1,B*N,d)

        # ===== build decoder tokens =====
        # pos: (m,d/2) -> (N,m,d/2)
        pos = self.pos_emb.unsqueeze(0).repeat(self.N, 1, 1)  # (N,m,d/2)
        # cell = self.channel_emb
        # cell emb: (N,d/2) -> (N,m,d/2)
        cell = self._build_cell_emb(x.device, x.dtype)        # (N,d/2)
        chn = cell.unsqueeze(1).repeat(1, self.seg_num_y, 1)  # (N,m,d/2)

        # token: concat -> (N*m,1,d) -> repeat B
        dec_tok = torch.cat([pos, chn], dim=-1).view(-1, 1, self.d_model)  # (N*m,1,d)
        dec_tok = dec_tok.repeat(B, 1, 1)                                  # (B*N*m,1,d)

        # expand hidden to match (B*N*m)
        hn_rep = hn.repeat(1, 1, self.seg_num_y).view(1, -1, self.d_model)  # (1,B*N*m,d)

        # ===== GRU decoder =====
        _, hy = self.rnn(dec_tok, hn_rep)  # hy: (1, B*N*m, d)

        # # 还原 Batch 和 N
        # temporal_feats = hy.view(B, self.N, self.seg_num_y, self.d_model)  # (B, N, m, D)
        #
        # feats_expanded = self.feature_expand(  temporal_feats)
        # B N m D*w

        # 4. 合并 m 和 w 成为 Tout
        # # (B, N, m, w, D) -> (B, N, Tout, D)
        # feats =  feats_expanded.view(B, self.N, self.Tout, self.d_model)
        #
        # # 5. 调整为 (B, Tout, N, D) 以适配 Spatial Transformer
        # feats = feats.permute(0, 2, 1, 3).contiguous()  # B TOUT N D

        #
        # if self.use_cross_attn:
        #     # hy -> (B*N, m, d)
        #     BN = B * self.N
        #     m = self.seg_num_y
        #     n = self.seg_num_x
        #     d = self.d_model
        #
        #     hy_q = hy.squeeze(0).view(BN, m, d)          # (BN,m,d)
        #
        #     # project
        #     q = self.q_proj(hy_q)                        # (BN,m,d)
        #     k = self.k_proj(enc_out)                     # (BN,n,d)
        #     v = self.v_proj(enc_out)                     # (BN,n,d)
        #
        #     # attn: (BN,m,n)
        #     attn = torch.matmul(q, k.transpose(1, 2)) / math.sqrt(d)
        #     attn = F.softmax(attn, dim=-1)
        #     attn = self.attn_drop(attn)
        #
        #     # ctx: (BN,m,d)
        #     ctx = torch.matmul(attn, v)
        #     ctx = self.ctx_proj(ctx)
        #
        #     # residual fuse
        #     hy_fuse = hy_q + ctx                          # (BN,m,d)
        #
        #     # back to (1, B*N*m, d)
        #     hy = hy_fuse.reshape(BN * m, d).unsqueeze(0)  # (1,B*N*m,d)


        # # ===== Cross-Attention (optional) =====
        # if self.use_cross_attn:
        #     # hy -> (B*N, m, d)
        #     BN = B * self.N
        #     m = self.seg_num_y
        #     n = self.seg_num_x
        #     d = self.d_model
        #
        #     hy_q = hy.squeeze(0).view(BN, m, d)          # (BN,m,d)
        #
        #     # project
        #     q = self.q_proj(hy_q)                        # (BN,m,d)
        #     k = self.k_proj(enc_out)                     # (BN,n,d)
        #     v = self.v_proj(enc_out)                     # (BN,n,d)
        #
        #     # attn: (BN,m,n)
        #     attn = torch.matmul(q, k.transpose(1, 2)) / math.sqrt(d)
        #     attn = F.softmax(attn, dim=-1)
        #     attn = self.attn_drop(attn)
        #
        #     # ctx: (BN,m,d)
        #     ctx = torch.matmul(attn, v)
        #     ctx = self.ctx_proj(ctx)
        #
        #     # residual fuse
        #     hy_fuse = hy_q + ctx                          # (BN,m,d)
        #
        #     # back to (1, B*N*m, d)
        #     hy = hy_fuse.reshape(BN * m, d).unsqueeze(0)  # (1,B*N*m,d)

        #===== predict seg_len =====
        y = self.predict(hy)  # (1,B*N*m,seg_len)

        # ===== reshape to (B,N,Tout) =====
        y = y.view(B, self.N, self.Tout)  # (B,N,Tout)

        # ===== (B,N,Tout) -> (B,Tout,N) and denorm =====
        y = y.permute(0, 2, 1).contiguous()   +seq_last# (B,Tout,N)

        # # ===== reshape back to grid =====
        # y = y.view(B, self.Tout, self.H, self.W)        # (B,Tout,H,W)
        # y = self.spatial_refine(y)  # 轻量空间细化（depthwise）
        return y
        # 如果你的训练脚本期望 (B,Tout,1,H,W)，用下面这一行：
        # y = y.unsqueeze(2)

