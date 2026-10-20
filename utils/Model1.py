import torch
import torch.nn as nn
# 引入基础模块
from utils.model import Model as iTransformerModel
from utils.segrnn_gridB import SegRNNGrid


class Serial_ST_Model(nn.Module):
    def __init__(self, T_in, T_out, H, W, d_model=128, seg_len=6, dropout=0.1, n_heads=4):
        super(Serial_ST_Model, self).__init__()
        self.H = H
        self.W = W

        # 1. 空间模块 (iTransformer)
        # 关键点：作为特征提取器，它的输出长度必须等于输入长度 (T_in)
        # 这样 SegRNN 才能拿到完整的历史序列
        self.spatial_encoder = iTransformerModel(
            T_in=T_in,
            T_out=T_in,  # ⚠️ 这里强制设为 T_in，不进行时间预测
            H=H, W=W,
            d_model=d_model,
            n_heads=n_heads,
            dropout=dropout
        )

        # 2. 时间模块 (SegRNN)
        # 负责将 T_in 长度的历史特征，推演为 T_out 长度的未来预测
        self.temporal_predictor = SegRNNGrid(
            Tin=T_in,
            Tout=T_out,  # ⚠️ 这里才是真正的预测长度
            H=H, W=W,
            d_model=d_model,
            seg_len=seg_len,
            dropout=dropout
        )

        # 3. 最终投影层
        # SegRNN 输出特征维度是 d_model，映射回 1 (流量值)
        self.final_projector = nn.Linear(d_model, 1)

    def forward(self, x):
        # Input x: (B, T_in, 1, H, W)

        # === Step 1: 空间增强 ===
        # iTransformer 期望 (B, T, H, W) 或 (B, T, 1, H, W)
        # 我们的修改版 iTransformer 会自动处理维度
        x_spatial = self.spatial_encoder(x)  # 输出 (B, T_in, H, W)

        # 残差连接: 原始输入 + 空间增强特征
        # 确保维度一致进行相加
        x_in_squeezed = x.squeeze(2) if x.dim() == 5 else x
        x_enhanced = x_in_squeezed + x_spatial

        # 变回 (B, T_in, 1, H, W) 喂给 SegRNN
        x_enhanced = x_enhanced.unsqueeze(2)

        # === Step 2: 时间推演 ===
        # SegRNN 输出特征: (B, T_out, N, D)
        feats = self.temporal_predictor(x_enhanced)

        # === Step 3: 最终输出 ===
        # (B, T_out, N, D) -> (B, T_out, N, 1)
        pred = self.final_projector(feats)

        # Reshape: (B, T_out, N, 1) -> (B, T_out, 1, H, W)
        pred = pred.squeeze(-1)  # (B, T_out, N)
        pred = pred.reshape(pred.shape[0], pred.shape[1], self.H, self.W).unsqueeze(2)

        return pred