# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F


class STBlock(nn.Module):
    """
    一个轻量的时空块：TemporalConv(沿时间) + SpatialConv(沿空间)
    输入:  (B, C, T, H, W)
    输出:  (B, C, T, H, W)
    """
    def __init__(self, channels: int, dropout: float = 0.1):
        super().__init__()
        # temporal conv: kernel=(3,1,1)
        self.tconv = nn.Sequential(
            nn.Conv3d(channels, channels, kernel_size=(3, 1, 1), padding=(1, 0, 0)),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        # spatial conv: kernel=(1,3,3)
        self.sconv = nn.Sequential(
            nn.Conv3d(channels, channels, kernel_size=(1, 3, 3), padding=(0, 1, 1)),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.norm = nn.BatchNorm3d(channels)

    def forward(self, x):
        # residual
        out = x + self.tconv(x)
        out = out + self.sconv(out)
        out = self.norm(out)
        return out


class CrossEncoder(nn.Module):
    """
    跨域静态特征编码器
    输入:  (B, Cc=4, H, W)
    输出:  (B, hidden, H, W)
    """
    def __init__(self, cross_dim: int, hidden: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(cross_dim, hidden // 2, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden // 2, hidden, kernel_size=3, padding=1),
            nn.GELU(),
        )

    def forward(self, cross):
        return self.net(cross)


class GatedFusion(nn.Module):
    """
    门控融合：用 cross 来调制主分支特征
    x_feat:    (B, hidden, T, H, W)
    cross_emb: (B, hidden, H, W) -> 扩展到 (B, hidden, T, H, W)
    """
    def __init__(self, hidden: int):
        super().__init__()
        # 生成 gate，输入拼接 [x, cross] -> gate in (0,1)
        self.gate = nn.Conv3d(hidden * 2, hidden, kernel_size=1)

    def forward(self, x_feat, cross_emb):
        B, C, T, H, W = x_feat.shape
        cross_exp = cross_emb.unsqueeze(2).expand(B, C, T, H, W)  # (B,C,T,H,W)
        g = torch.sigmoid(self.gate(torch.cat([x_feat, cross_exp], dim=1)))
        # gated add
        out = x_feat + g * cross_exp
        return out


class DualBranchSTPredictor(nn.Module):
    """
    输入:
      x     : (B, Tin=25, 1, H=20, W=20)
      cross : (B, Cc=4, H=20, W=20)
    输出:
      yhat  : (B, Tout=8, 1, H, W)
    """
    def __init__(
        self,
        tin: int = 25,
        tout: int = 8,
        in_channels: int = 1,
        cross_dim: int = 4,
        hidden: int = 64,
        n_blocks: int = 3,
        dropout: float = 0.1,
        fusion: str = "gated_add",  # "gated_add" / "add" / "none"
    ):
        super().__init__()
        self.tin = tin
        self.tout = tout
        self.hidden = hidden
        self.fusion_mode = fusion

        # --- 主分支输入投影: (B,1,T,H,W) -> (B,hidden,T,H,W)
        self.in_proj = nn.Conv3d(in_channels, hidden, kernel_size=1)

        # --- 跨域分支
        self.cross_encoder = CrossEncoder(cross_dim=cross_dim, hidden=hidden)

        # --- 融合模块
        if fusion == "gated_add":
            self.fusion = GatedFusion(hidden=hidden)
        else:
            self.fusion = None

        # --- 时空编码器
        self.blocks = nn.ModuleList([STBlock(hidden, dropout=dropout) for _ in range(n_blocks)])

        # --- 解码器（简单稳定）：把 Tin 压成 1，再映射到 Tout
        # x: (B,hidden,T,H,W) -> (B,hidden,1,H,W)
        self.time_pool = nn.Conv3d(hidden, hidden, kernel_size=(tin, 1, 1), padding=0)

        # (B,hidden,1,H,W) -> (B,tout,1,H,W)
        self.out_proj = nn.Conv3d(hidden, tout, kernel_size=1)
        self.out_norm = nn.BatchNorm3d(tout)

    def forward(self, x, cross=None):
        # x: (B,T,1,H,W) -> (B,1,T,H,W)
        x = x.permute(0, 2, 1, 3, 4).contiguous()
        feat = self.in_proj(x)  # (B,hidden,T,H,W)

        # cross: (B,4,H,W)
        if (cross is not None) and (self.fusion_mode != "none"):
            cross_emb = self.cross_encoder(cross)  # (B,hidden,H,W)
            if self.fusion_mode == "gated_add":
                feat = self.fusion(feat, cross_emb)
            elif self.fusion_mode == "add":
                feat = feat + cross_emb.unsqueeze(2).expand_as(feat)

        # encoder blocks
        for blk in self.blocks:
            feat = blk(feat)

        # decode
        z = self.time_pool(feat)          # (B,hidden,1,H,W)
        out = self.out_proj(z)            # (B,tout,1,H,W)
        out = self.out_norm(out)
        # -> (B,tout,1,H,W) already
        return out
