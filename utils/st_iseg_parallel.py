import torch
import torch.nn as nn
# 引入基础模块
from utils.model import Model as iTransformerModel
from utils.segrnn_gridB import SegRNNGrid


class Parallel_ST_Model(nn.Module):
    def __init__(self, T_in, T_out, H, W, d_model=128, seg_len=6, dropout=0.1, n_heads=4):
        super(Parallel_ST_Model, self).__init__()
        self.H = H
        self.W = W
        self.N = H * W
        self.T_out = T_out

        # ==========================================
        # 分支 A: 空间预测 (iTransformer)
        # ==========================================
        # 注意：这里 T_out 直接设为预测步长 (例如 12)
        # 让 iTransformer 直接干预测的活
        self.spatial_branch = iTransformerModel(
            T_in=T_in,
            T_out=T_out,  # ⚠️ 并联模式下，这里是预测长度
            H=H, W=W,
            d_model=d_model,
            n_heads=n_heads,
            dropout=dropout
        )

        # ==========================================
        # 分支 B: 时间预测 (SegRNN)
        # ==========================================
        self.temporal_branch = SegRNNGrid(
            Tin=T_in,
            Tout=T_out,
            H=H, W=W,
            d_model=d_model,
            seg_len=seg_len,
            dropout=dropout
        )
        # SegRNN 输出的是特征 (D维)，需要一个头把它转成流量值 (1维)
        self.temporal_head = nn.Linear(d_model, 1)

        # ==========================================
        # 融合模块 (Fusion)
        # ==========================================
        # 定义一个可学习的参数 w，初始值 0.5 (sigmoid后)
        self.fusion_weight = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        # Input x: (B, T_in, 1, H, W)
        B, T_in, _, H, W = x.shape

        # === Branch A: iTransformer 预测 ===
        # iTransformer 内部会自动处理 squeeze/unsqueeze
        # 输出: (B, T_out, H, W)
        out_spatial = self.spatial_branch(x)

        # 统一维度: (B, T_out, H, W) -> (B, T_out, 1, H, W)
        if out_spatial.dim() == 4:
            out_spatial = out_spatial.unsqueeze(2)

        # === Branch B: SegRNN 预测 ===
        # SegRNN 期望 (B, T_in, 1, H, W)
        # 输出特征: (B, T_out, N, D)
        feats_temporal = self.temporal_branch(x)

        # 投影: (B, T_out, N, D) -> (B, T_out, N, 1)
        out_temporal = self.temporal_head(feats_temporal)

        # Reshape 回网格: (B, T_out, N, 1) -> (B, T_out, 1, H, W)
        out_temporal = out_temporal.squeeze(-1)  # (B, T_out, N)
        out_temporal = out_temporal.reshape(B, self.T_out, H, W).unsqueeze(2)

        # === Fusion 融合 ===
        # 公式: y = w * spatial + (1-w) * temporal
        w = torch.sigmoid(self.fusion_weight)

        final_pred = w * out_spatial + (1 - w) * out_temporal

        return final_pred