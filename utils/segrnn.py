# utils/segrnn_grid.py
# -*- coding: utf-8 -*-
import math
import torch
import torch.nn as nn

import torch
import torch.nn as nn
import torch.nn.functional as F

class LayerNorm(nn.Module):
    r""" LayerNorm that supports two data formats: channels_last (default) or channels_first.
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs
    with shape (batch_size, channels, height, width).
    """

    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_first"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x


class ConvMod(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.norm = LayerNorm(dim, eps=1e-6)
        self.a = nn.Sequential(
            nn.Conv2d(dim, dim, 1),
            nn.GELU(),
            nn.Conv2d(dim, dim, 11, padding=5, groups=dim)
        )
        self.v = nn.Conv2d(dim, dim, 1)
        self.proj = nn.Conv2d(dim, dim, 1)

    def forward(self, x):
        N, C, H, W = x.shape
        x = self.norm(x)
        a = self.a(x)
        v = self.v(x)
        x = a * v
        x = self.proj(x)
        return x


class SegRNNGrid(nn.Module):
    """
    纯 SegRNN（SegGRU）用于网格时空预测：
    输入:  x (B, Tin, 1, H, W)
    输出:  y (B, Tout, 1, H, W)

    思路：
    - 把 (H,W) 展平为 400 个“变量/通道”，enc_in=H*W
    - SegRNN 在时间维建模： (B, Tin, enc_in) -> (B, Tout, enc_in)
    - reshape 回 (B, Tout, 1, H, W)
    """
    def __init__(self, Tin=25, Tout=8, H=20, W=20, d_model=128, seg_len=1, dropout=0.1):
        super().__init__()
        assert Tin % seg_len == 0, "Tin 必须能被 seg_len 整除"
        assert Tout % seg_len == 0, "Tout 必须能被 seg_len 整除"
        self.embed_size = 128  # embed_size
        self.Tin = Tin
        self.Tout = Tout
        self.H = H
        self.W = W
        self.N = H * W
        self.d_model = d_model
        self.seg_len = seg_len
        self.embeddings = nn.Parameter(torch.randn(1, self.embed_size))
        self.row_emb = nn.Parameter(torch.randn(self.H, d_model // 4) * 0.02)  # (H, d/4)
        self.col_emb = nn.Parameter(torch.randn(self.W, d_model // 4) * 0.02)  # (W, d/4)
        # optional: fuse row/col into a better token (still output d/2)
        self.channel_fuse = nn.Sequential(
            nn.Linear(d_model // 2, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, d_model // 2),
        )
        self.seg_num_x = Tin // seg_len
        self.seg_num_y = Tout // seg_len

        # (seg_len -> d_model)
        self.valueEmbedding = nn.Sequential(
            nn.Linear(seg_len, d_model),
            nn.ReLU(inplace=True)
        )

        self.rnn = nn.GRU(
            input_size=d_model,
            hidden_size=d_model,
            num_layers=1,
            batch_first=True,
            bidirectional=False
        )
        self.fc = nn.Sequential(
            nn.Linear(self.Tin * self.embed_size, 2*self.embed_size),
            nn.LeakyReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(2*self.embed_size, self.Tin)
        )
        # pos emb + channel emb (与你贴的代码一致)
        self.pos_emb = nn.Parameter(torch.randn(self.seg_num_y, d_model // 2) * 0.02)   # (m, d/2)
        self.channel_emb = nn.Parameter(torch.randn(self.N, d_model // 2) * 0.02)      # (enc_in, d/2)
        self. Model = ConvMod(12)
        self.predict = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(d_model, seg_len)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, Tin, 1, H, W)
        return: (B, Tout, 1, H, W)
        """
        B, Tin, C, H, W = x.shape
        assert Tin == self.Tin and C == 1 and H == self.H and W == self.W



        # flatten grid as channels: (B,T,1,H,W)->(B,T,N)
        x = x.squeeze(2).reshape(B, Tin, self.N)  # (B,T,N)

        # x = x.permute(0, 2, 1)  # $ btn->b n t
        # x = x.unsqueeze(3)
        # y = self.embeddings
        #
        # x=x*y   # B    N  t D
        #
        #
        #
        # # 现在的形状: (B, N, D, t)
        # x = self.Model(x)  # out:  (B, N, D, t)
        #
        # x = x.reshape(B, H*W, -1)  # B N  (T*D)
        # x = self.fc(x)  # B T D
        # x = x.permute(0, 2, 1)  # (T*D)xN


        # normalization (last step) like original code
        seq_last = x[:, -1:, :].detach()          # (B,1,N)
        x = x - seq_last                          # (B,T,N)


        # permute to (B,N,T)
        x = x.permute(0, 2, 1).contiguous()       # (B,N,T)

        # segment: (B,N,T)->(B*N, seg_num_x, seg_len)
        x_seg = x.view(B * self.N, self.seg_num_x, self.seg_len)
        x_emb = self.valueEmbedding(x_seg)        # (B*N, n, d)

        _, hn = self.rnn(x_emb)                   # hn: (1, B*N, d)




        # decoder tokens: (N,m,d/2)+(N,m,d/2)->(N*m,1,d)->repeat B
        pos = self.pos_emb.unsqueeze(0).repeat(self.N, 1, 1)             # (N,m,d/2)

        # -------- build 2D factorized channel embedding --------
        # row: (H, d/4) -> (H, W, d/4) ; col: (W, d/4) -> (H, W, d/4)
        row = self.row_emb[:, None, :].expand(self.H, self.W, -1)  # (H, W, d/4)
        col = self.col_emb[None, :, :].expand(self.H, self.W, -1)  # (H, W, d/4)

        cell = torch.cat([row, col], dim=-1).reshape(self.N, -1)  # (N, d/2)
        cell = self.channel_fuse(cell)  # (N, d/2) optional but recommended

        chn = cell.unsqueeze(1).repeat(1, self.seg_num_y, 1)  # (N, m, d/2)
        #
        # chn = self.channel_emb.unsqueeze(1).repeat(1, self.seg_num_y, 1)  # (N,m,d/2)
        dec_tok = torch.cat([pos, chn], dim=-1).view(-1, 1, self.d_model) # (N*m,1,d)
        dec_tok = dec_tok.repeat(B, 1, 1)                                 # (B*N*m,1,d)


        hn_rep = hn.repeat(1, 1, self.seg_num_y).view(1, -1, self.d_model)  # (1,B*N*m,d)
        _, hy = self.rnn(dec_tok, hn_rep)                                    # (1,B*N*m,d)

        y = self.predict(hy)                                                 # (1,B*N*m,seg_len)
        y = y.view(B, self.N, self.Tout)                                     # (B,N,Tout)

        # back: (B,N,Tout)->(B,Tout,N) + denorm
        y = y.permute(0, 2, 1).contiguous() + seq_last                       # (B,Tout,N)

        # reshape back to grid
        y = y.view(B, self.Tout,  self.H, self.W)
        y = y+self.Model(y)  # out:  (B, N, D, t)
        return y

# class SegRNNGrid(nn.Module):
#     """
#     纯 SegRNN（SegGRU）用于网格时空预测：
#     输入:  x (B, Tin, 1, H, W)
#     输出:  y (B, Tout, 1, H, W)
#
#     思路：
#     - 把 (H,W) 展平为 400 个“变量/通道”，enc_in=H*W
#     - SegRNN 在时间维建模： (B, Tin, enc_in) -> (B, Tout, enc_in)
#     - reshape 回 (B, Tout, 1, H, W)
#     """
#     def __init__(self, Tin=25, Tout=8, H=20, W=20, d_model=128, seg_len=1, dropout=0.1):
#         super().__init__()
#         assert Tin % seg_len == 0, "Tin 必须能被 seg_len 整除"
#         assert Tout % seg_len == 0, "Tout 必须能被 seg_len 整除"
#         # ===== Embedding-driven Adaptive Graph (from cell embedding) =====
#         self.k = 8  # top-k neighbors
#         self.tau = nn.Parameter(torch.tensor(0.2))  # temperature
#         self.alpha = nn.Parameter(torch.tensor(0.1))  # residual mixing strength
#
#         dk = max(32, d_model // 4)  # small dim for similarity
#         self.emb_proj = nn.Linear(d_model // 2, dk, bias=False)
# #############################################################################
#         self.Tin = Tin
#         self.Tout = Tout
#         self.H = H
#         self.W = W
#         self.N = H * W
#         self.d_model = d_model
#         self.seg_len = seg_len
#         self.row_emb = nn.Parameter(torch.randn(self.H, d_model // 4) * 0.02)  # (H, d/4)
#         self.col_emb = nn.Parameter(torch.randn(self.W, d_model // 4) * 0.02)  # (W, d/4)
#         # optional: fuse row/col into a better token (still output d/2)
#         self.channel_fuse = nn.Sequential(
#             nn.Linear(d_model // 2, d_model // 2),
#             nn.GELU(),
#             nn.Linear(d_model // 2, d_model // 2),
#         )
#         self.seg_num_x = Tin // seg_len
#         self.seg_num_y = Tout // seg_len
#
#         # (seg_len -> d_model)
#         self.valueEmbedding = nn.Sequential(
#             nn.Linear(seg_len, d_model),
#             nn.ReLU(inplace=True)
#         )
#
#         self.rnn = nn.GRU(
#             input_size=d_model,
#             hidden_size=d_model,
#             num_layers=1,
#             batch_first=True,
#             bidirectional=False
#         )
#
#         # pos emb + channel emb (与你贴的代码一致)
#         self.pos_emb = nn.Parameter(torch.randn(self.seg_num_y, d_model // 2) * 0.02)   # (m, d/2)
#         self.channel_emb = nn.Parameter(torch.randn(self.N, d_model // 2) * 0.02)      # (enc_in, d/2)
#
#         self.predict = nn.Sequential(
#             nn.Dropout(dropout),
#             nn.Linear(d_model, seg_len)
#         )
#
#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         """
#         x: (B, Tin, 1, H, W)
#         return: (B, Tout, 1, H, W)
#         """
#         B, Tin, C, H, W = x.shape
#         assert Tin == self.Tin and C == 1 and H == self.H and W == self.W
#
#         # (B,T,1,H,W) -> (B,T,N)
#         x = x.squeeze(2).reshape(B, Tin, self.N)  # (B,T,N)
#
#         # normalization (last step)
#         seq_last = x[:, -1:, :].detach()  # (B,1,N)
#         x = x - seq_last  # (B,T,N)
#
#         # ===== build 2D factorized cell embedding (once) =====
#         row = self.row_emb[:, None, :].expand(self.H, self.W, -1)  # (H,W,d/4)
#         col = self.col_emb[None, :, :].expand(self.H, self.W, -1)  # (H,W,d/4)
#         cell = torch.cat([row, col], dim=-1).reshape(self.N, -1)  # (N,d/2)
#         cell = self.channel_fuse(cell)  # (N,d/2)
#
#         # ===== build adaptive adjacency A from cell embedding (once) =====
#         E = self.emb_proj(cell)  # (N, dk)
#         E = F.normalize(E, dim=-1)  # cosine normalize
#         S = E @ E.t()  # (N, N)
#         S.fill_diagonal_(-1e9)  # exclude self for topk
#
#         k = min(self.k, self.N - 1)
#         vals, idx = torch.topk(S, k=k, dim=-1)  # (N, k)
#
#         A = torch.full_like(S, -1e9)  # masked logits
#         A.scatter_(dim=-1, index=idx, src=vals)
#
#         A = F.softmax(A / (self.tau.abs() + 1e-6), dim=-1)  # row-stochastic
#
#         # add self-loop + renormalize (stability)
#         I = torch.eye(self.N, device=A.device, dtype=A.dtype)
#         A = A + I
#         A = A / (A.sum(dim=-1, keepdim=True) + 1e-6)
#
#         # ===== INPUT spatial mixing (关键新增) =====
#         x = x + self.alpha * (x @ A)  # (B,T,N)
#
#         # (B,T,N) -> (B,N,T)
#         x = x.permute(0, 2, 1).contiguous()  # (B,N,T)
#
#         # segment: (B,N,T)->(B*N, seg_num_x, seg_len)
#         x_seg = x.view(B * self.N, self.seg_num_x, self.seg_len)  # (B*N,n,seg_len)
#         x_emb = self.valueEmbedding(x_seg)  # (B*N,n,d)
#
#         _, hn = self.rnn(x_emb)  # hn: (1,B*N,d)
#
#         # decoder tokens
#         pos = self.pos_emb.unsqueeze(0).repeat(self.N, 1, 1)  # (N,m,d/2)
#         chn = cell.unsqueeze(1).repeat(1, self.seg_num_y, 1)  # (N,m,d/2)
#
#         dec_tok = torch.cat([pos, chn], dim=-1).view(-1, 1, self.d_model)  # (N*m,1,d)
#         dec_tok = dec_tok.repeat(B, 1, 1)  # (B*N*m,1,d)
#
#         hn_rep = hn.repeat(1, 1, self.seg_num_y).view(1, -1, self.d_model)  # (1,B*N*m,d)
#         _, hy = self.rnn(dec_tok, hn_rep)  # (1,B*N*m,d)
#
#         y = self.predict(hy)  # (1,B*N*m,seg_len)
#         y = y.view(B, self.N, self.Tout)  # (B,N,Tout)
#
#         # (B,N,Tout) -> (B,Tout,N)
#         y = y.permute(0, 2, 1).contiguous()  # (B,Tout,N)
#
#         # ===== OUTPUT spatial mixing (你已有，保留) =====
#         y = y + self.alpha * (y @ A)  # (B,Tout,N)
#
#         # denorm
#         y = y + seq_last  # (B,Tout,N)
#
#         # reshape back to grid
#         y = y.view(B, self.Tout, self.H, self.W)
#         return y
#
#     # def forward(self, x: torch.Tensor) -> torch.Tensor:
#     #     """
#     #     x: (B, Tin, 1, H, W)
#     #     return: (B, Tout, 1, H, W)
#     #     """
#     #     B, Tin, C, H, W = x.shape
#     #     assert Tin == self.Tin and C == 1 and H == self.H and W == self.W
#     #
#     #     # flatten grid as channels: (B,T,1,H,W)->(B,T,N)
#     #     x = x.squeeze(2).reshape(B, Tin, self.N)  # (B,T,N)
#     #
#     #     # normalization (last step) like original code
#     #     seq_last = x[:, -1:, :].detach()          # (B,1,N)
#     #     x = x - seq_last                          # (B,T,N)
#     #
#     #
#     #     # permute to (B,N,T)
#     #     x = x.permute(0, 2, 1).contiguous()       # (B,N,T)
#     #
#     #     # segment: (B,N,T)->(B*N, seg_num_x, seg_len)
#     #     x_seg = x.view(B * self.N, self.seg_num_x, self.seg_len)
#     #     x_emb = self.valueEmbedding(x_seg)        # (B*N, n, d)
#     #
#     #     _, hn = self.rnn(x_emb)                   # hn: (1, B*N, d)
#     #
#     #     # decoder tokens: (N,m,d/2)+(N,m,d/2)->(N*m,1,d)->repeat B
#     #     pos = self.pos_emb.unsqueeze(0).repeat(self.N, 1, 1)             # (N,m,d/2)
#     #
#     #     # -------- build 2D factorized channel embedding --------
#     #     # row: (H, d/4) -> (H, W, d/4) ; col: (W, d/4) -> (H, W, d/4)
#     #     row = self.row_emb[:, None, :].expand(self.H, self.W, -1)  # (H, W, d/4)
#     #     col = self.col_emb[None, :, :].expand(self.H, self.W, -1)  # (H, W, d/4)
#     #
#     #     cell = torch.cat([row, col], dim=-1).reshape(self.N, -1)  # (N, d/2)
#     #     cell = self.channel_fuse(cell)  # (N, d/2) optional but recommended
#     #
#     #     # ===== build adaptive adjacency A from cell embedding =====
#     #     E = self.emb_proj(cell)  # (N, dk)
#     #     E = F.normalize(E, dim=-1)  # cosine normalize
#     #     S = E @ E.t()  # (N, N)
#     #     S.fill_diagonal_(-1e9)  # exclude self for topk
#     #
#     #     k = min(self.k, self.N - 1)
#     #     vals, idx = torch.topk(S, k=k, dim=-1)  # (N, k)
#     #
#     #     A = torch.full_like(S, -1e9)  # masked logits
#     #     A.scatter_(dim=-1, index=idx, src=vals)
#     #
#     #     A = F.softmax(A / (self.tau.abs() + 1e-6), dim=-1)  # row-stochastic
#     #
#     #     # add self-loop + renormalize (stability)
#     #     I = torch.eye(self.N, device=A.device, dtype=A.dtype)
#     #     A = A + I
#     #     A = A / (A.sum(dim=-1, keepdim=True) + 1e-6)
#     #
#     #     chn = cell.unsqueeze(1).repeat(1, self.seg_num_y, 1)  # (N, m, d/2)
#     #     #
#     #     # chn = self.channel_emb.unsqueeze(1).repeat(1, self.seg_num_y, 1)  # (N,m,d/2)
#     #     dec_tok = torch.cat([pos, chn], dim=-1).view(-1, 1, self.d_model) # (N*m,1,d)
#     #     dec_tok = dec_tok.repeat(B, 1, 1)                                 # (B*N*m,1,d)
#     #
#     #
#     #     hn_rep = hn.repeat(1, 1, self.seg_num_y).view(1, -1, self.d_model)  # (1,B*N*m,d)
#     #     _, hy = self.rnn(dec_tok, hn_rep)                                    # (1,B*N*m,d)
#     #
#     #     y = self.predict(hy)                                                 # (1,B*N*m,seg_len)
#     #     y = y.view(B, self.N, self.Tout)                                     # (B,N,Tout)
#     #
#     #     # (B,N,Tout) -> (B,Tout,N)
#     #     y = y.permute(0, 2, 1).contiguous()  # (B,Tout,N)
#     #
#     #     # ===== output spatial mixing =====
#     #     y = y + self.alpha * (y @ A)  # (B,Tout,N)
#     #
#     #     # denorm
#     #     y = y + seq_last  # (B,Tout,N)
#     #
#     #     # reshape back to grid
#     #     y = y.view(B, self.Tout, self.H, self.W)
#     #     return y
#     #
#     #
#     #
#     #
#     #
#     #     # # back: (B,N,Tout)->(B,Tout,N) + denorm
#     #     # y = y.permute(0, 2, 1).contiguous() + seq_last                       # (B,Tout,N)
#     #     #
#     #     # # reshape back to grid
#     #     # y = y.view(B, self.Tout,  self.H, self.W)
#     #     # return y
