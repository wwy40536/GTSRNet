# utils/factorized_st_transformer.py
# -*- coding: utf-8 -*-
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class SinusoidalPosEmb(nn.Module):
    """正弦时间位置编码: (1, T, D)"""
    def __init__(self, d_model: int, max_len: int = 512):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))  # (1,max_len,D)

    def forward(self, T: int):
        return self.pe[:, :T]


class LearnableSpatialPosEmb(nn.Module):
    """可学习空间位置编码: (1, N, D)"""
    def __init__(self, num_nodes: int, d_model: int):
        super().__init__()
        self.emb = nn.Parameter(torch.zeros(1, num_nodes, d_model))
        nn.init.trunc_normal_(self.emb, std=0.02)

    def forward(self):
        return self.emb


class MLP(nn.Module):
    def __init__(self, d_model: int, dim_ff: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, dim_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class SpatialTransformerBlock(nn.Module):
    """
    对每个时间步做空间注意力：
      输入:  (B, T, N, D)
      输出:  (B, T, N, D)
    实现：把 (B*T, N, D) 送入 MultiheadAttention
    """
    def __init__(self, d_model: int, nhead: int, dim_ff: int, dropout: float):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(d_model)
        self.mlp = MLP(d_model, dim_ff, dropout)

    def forward(self, x):
        B, T, N, D = x.shape
        xt = x.reshape(B * T, N, D)
        h = self.norm1(xt)
        h, _ = self.attn(h, h, h, need_weights=False)
        xt = xt + h
        h2 = self.mlp(self.norm2(xt))
        xt = xt + h2
        return xt.reshape(B, T, N, D)


class TemporalTransformerBlock(nn.Module):
    """
    对每个格点做时间注意力：
      输入:  (B, T, N, D)
      输出:  (B, T, N, D)
    实现：把 (B*N, T, D) 送入 MultiheadAttention
    """
    def __init__(self, d_model: int, nhead: int, dim_ff: int, dropout: float):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(d_model)
        self.mlp = MLP(d_model, dim_ff, dropout)

    def forward(self, x):
        B, T, N, D = x.shape
        xn = x.permute(0, 2, 1, 3).contiguous().reshape(B * N, T, D)
        h = self.norm1(xn)
        h, _ = self.attn(h, h, h, need_weights=False)
        xn = xn + h
        h2 = self.mlp(self.norm2(xn))
        xn = xn + h2
        xn = xn.reshape(B, N, T, D).permute(0, 2, 1, 3).contiguous()
        return xn


class FactorizedSTTransformer(nn.Module):
    """
    Factorized ST-Transformer: Spatial MHA per time + Temporal MHA per node
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
        depth: int = 2,          # 一个深度包含：SpatialBlock + TemporalBlock
        dim_ff: int = 128,
        dropout: float = 0.1,
        use_time_pos: bool = True,
        use_space_pos: bool = True,
        use_input_skip: bool = True,   # 让训练更稳：y = f(x)+last_x
    ):
        super().__init__()
        self.Tin = Tin
        self.Tout = Tout
        self.H = H
        self.W = W
        self.N = H * W
        self.d_model = d_model
        self.use_input_skip = use_input_skip

        # 每个时间步： (1,H,W)->(D,H,W)
        self.embed = nn.Conv2d(in_channels, d_model, kernel_size=1, bias=True)

        # 位置编码
        self.time_pos = SinusoidalPosEmb(d_model) if use_time_pos else None
        self.space_pos = LearnableSpatialPosEmb(self.N, d_model) if use_space_pos else None

        # 堆叠 factorized blocks
        self.s_blocks = nn.ModuleList([SpatialTransformerBlock(d_model, nhead, dim_ff, dropout) for _ in range(depth)])
        self.t_blocks = nn.ModuleList([TemporalTransformerBlock(d_model, nhead, dim_ff, dropout) for _ in range(depth)])

        # readout：用“最后时刻 token”预测 Tout
        self.readout = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, Tout)
        )

        # 轻量空间 refine（可选，保留更平滑）
        self.spatial_refine = nn.Conv2d(Tout, Tout, kernel_size=3, padding=1, bias=False)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.Conv2d)):
                nn.init.xavier_uniform_(m.weight)
                if getattr(m, "bias", None) is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, Tin, 1, H, W)
        return: (B, Tout, 1, H, W)
        """
        B, T, C, H, W = x.shape
        assert T == self.Tin and H == self.H and W == self.W and C == 1

        # 保存最后一帧做 skip（可选）
        x_last = x[:, -1, 0]  # (B,H,W)

        # embed: (B*T,1,H,W)->(B*T,D,H,W)
        xt = x.reshape(B * T, C, H, W)
        feat = self.embed(xt).reshape(B, T, self.d_model, H, W)  # (B,T,D,H,W)
        feat = feat.flatten(3, 4).permute(0, 1, 3, 2).contiguous()  # (B,T,N,D)

        # 加位置编码
        if self.space_pos is not None:
            feat = feat + self.space_pos()  # (1,N,D) broadcast 到 (B,T,N,D)
        if self.time_pos is not None:
            feat = feat + self.time_pos(T).to(feat.dtype).to(feat.device).unsqueeze(2)  # (1,T,1,D)

        # factorized blocks: [Spatial -> Temporal] x depth
        z = feat
        for sb, tb in zip(self.s_blocks, self.t_blocks):
            z = sb(z)
            z = tb(z)

        # readout：取最后时刻 token
        last = z[:, -1, :, :]                # (B,N,D)
        out = self.readout(last)             # (B,N,Tout)
        out = out.permute(0, 2, 1).contiguous().reshape(B, self.Tout, H, W)  # (B,Tout,H,W)

        out = self.spatial_refine(out)

        if self.use_input_skip:
            # 用 last frame 做残差：输出更稳（对 traffic 很常用）
            out = out + x_last.unsqueeze(1)  # (B,1,H,W) -> broadcast 到 Tout

        return out.unsqueeze(2)              # (B,Tout,1,H,W)


if __name__ == "__main__":
    m = FactorizedSTTransformer(Tin=25, Tout=8, H=20, W=20, d_model=64, nhead=4, depth=2)
    x = torch.randn(2, 25, 1, 20, 20)
    y = m(x)
    print(x.shape, y.shape)
