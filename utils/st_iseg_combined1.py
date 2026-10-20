import torch
import torch.nn as nn
import torch.nn.functional as F

# ================================================================
# 1. 导入依赖
# ================================================================
from utils.segrnn_gridB import SegRNNGrid


from utils.layers.Transformer_EncDec import Encoder, EncoderLayer
from utils.layers.SelfAttention_Family import FullAttention, AttentionLayer
from utils.layers.Embed import DataEmbedding_inverted

# from layers.Transformer_EncDec import Encoder, EncoderLayer
# from layers.SelfAttention_Family import FullAttention, AttentionLayer
# from layers.Embed import DataEmbedding_inverted


class GatedMLPFusion(nn.Module):
    def __init__(self, hidden_dim=16):
        super(GatedMLPFusion, self).__init__()
        # 输入维度是 2 (Spatial + Temporal)

        # 门控分支: 决定"通过多少信息"
        self.w_gate = nn.Linear(2, hidden_dim)

        # 数值分支: 决定"信息的内容是什么"
        self.w_val = nn.Linear(2, hidden_dim)

        # 输出层: 融合后的结果
        self.w_out = nn.Linear(hidden_dim, 1)

        # 激活函数
        self.act = nn.GELU()

    def forward(self, spatial_pred, temporal_pred):
        """
        Input:
            spatial_pred:  (B, T, N)
            temporal_pred: (B, T, N)
        Output:
            final_pred:    (B, T, N)
        """
        # 1. 堆叠: (B, T, N, 2)
        # 在最后一个维度把两个预测值拼起来
        x = torch.stack([spatial_pred, temporal_pred], dim=-1)

        # 2. Gated Mechanism (类似 GLU)
        # gate: (B, T, N, hidden) -> 0~1 之间
        gate = torch.sigmoid(self.w_gate(x))

        # val: (B, T, N, hidden) -> 特征变换
        val = self.act(self.w_val(x))

        # 3. 元素级相乘与映射
        # (B, T, N, hidden) -> (B, T, N, 1)
        out = self.w_out(gate * val)

        # 4. 去掉最后一个维度
        return out.squeeze(-1)
# ================================================================
# 2. 空间预测分支: SpatialPredictor (原改进版 iTransformer)
# ================================================================
class SpatialPredictor(nn.Module):
    def __init__(self, T_in, T_out, H, W, d_model=64, cycle=24,
                 n_heads=4, e_layers=2, d_ff=256, dropout=0.1, activation='gelu',
                 embed='timeF', freq='h', factor=1, output_attention=False,
                 use_norm=True, output_proj_dropout=0.1):
        super(SpatialPredictor, self).__init__()
        self.seq_len = T_in
        self.pred_len = T_out  # ⚠️ 并联模式下，这里直接是预测长度
        self.H = H
        self.W = W
        self.enc_in = H * W
        self.cycle_len = cycle
        self.d_model = d_model
        self.output_attention = output_attention
        self.use_norm = use_norm

        self.enc_embedding = DataEmbedding_inverted(self.seq_len, self.d_model, embed, freq, dropout)

        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(False, factor, attention_dropout=dropout,
                                      output_attention=output_attention), d_model, n_heads),
                    d_model,
                    d_ff,
                    dropout=dropout,
                    activation=activation
                ) for l in range(e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(d_model)
        )

        # 投影层: 将 d_model 映射到 pred_len (直接预测未来)
        self.projector = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Dropout(output_proj_dropout),
            nn.Linear(d_model * 2, d_model * 4),
            nn.GELU(),
            nn.Dropout(output_proj_dropout),
            nn.Linear(d_model * 4, self.pred_len),
        )

        self.channel_embedding = nn.Parameter(torch.zeros(self.enc_in, d_model))
        self.phase_embedding = nn.Embedding(self.cycle_len, d_model)
        nn.init.xavier_normal_(self.phase_embedding.weight)
        self.joint_embedding = nn.Embedding(self.cycle_len, self.enc_in * d_model)
        nn.init.xavier_normal_(self.joint_embedding.weight)
        nn.init.xavier_normal_(self.channel_embedding)

    def forecast(self, x_enc, x_mark_enc, phase):
        if self.use_norm:
            means = x_enc.mean(1, keepdim=True).detach()
            x_enc = x_enc - means
            stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
            x_enc /= stdev

        B, L, N = x_enc.shape
        enc_out = self.enc_embedding(x_enc, x_mark_enc)

        # Embeddings Add
        channel_emb = self.channel_embedding.expand(enc_out.shape[0], N, -1)
        phase_emb = self.phase_embedding(phase.view(-1, 1).expand(B, N))
        joint_emb = self.joint_embedding(phase).reshape(B, self.enc_in, self.d_model)

        enc_out = enc_out[:, :N, :] + channel_emb + phase_emb + joint_emb
        enc_orgin = enc_out

        enc_out, attns = self.encoder(enc_out, attn_mask=None)
        dec_out = self.projector(enc_out + enc_orgin).permute(0, 2, 1)[:, :, :N]

        if self.use_norm:
            dec_out = dec_out * (stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
            dec_out = dec_out + (means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))

        return dec_out, attns

    def forward(self, x, x_mark_enc=None, cycle_index=None):
        if x.dim() == 5:
            x = x.squeeze(2)
        B, T, H, W = x.shape
        x_enc = x.reshape(B, T, -1)
        x_mark_enc = None
        if cycle_index is None:
            cycle_index = torch.zeros(B, dtype=torch.long).to(x.device)

        dec_out, attns = self.forecast(x_enc, x_mark_enc, cycle_index)

        # Reshape to (B, T_out, H, W)
        dec_out = dec_out.reshape(B, self.pred_len, H* W)

        if self.output_attention:
            return dec_out, attns
        else:
            return dec_out


class ConvFusion(nn.Module):
    def __init__(self):
        super().__init__()
        # 输入通道为 2 (Spatial + Temporal)，输出通道为 1
        # 1x1 卷积相当于在每个像素点上做一个全连接层
        self.fusion_conv = nn.Conv2d(2, 1, kernel_size=1)

    def forward(self, spatial_out, temporal_out):
        # spatial_out: (B, T, 1, H, W)
        # temporal_out: (B, T, 1, H, W)
        B, T, C, H, W = spatial_out.shape

        # 1. 拼接: (B, T, 2, H, W)
        cat = torch.cat([spatial_out, temporal_out], dim=2)

        # 2. Reshape 为 (B*T, 2, H, W) 以进行 2D 卷积
        cat_flat = cat.view(-1, 2, H, W)

        # 3. 卷积融合
        out = self.fusion_conv(cat_flat)  # (B*T, 1, H, W)

        # 4. 还原
        return out.view(B, T, 1, H, W)
# ================================================================
# 3. 并联模型: Parallel_ST_Model (Fusion)
#    架构:
#    Input --> [Spatial Branch] --> Predict A \
#                                              (+) --> Output
#    Input --> [Temporal Branch]--> Predict B /
# ================================================================
class Parallel_ST_Model1(nn.Module):
    def __init__(self, T_in, T_out, H, W, d_model=128, seg_len=6, dropout=0.1, n_heads=4):
        super(Parallel_ST_Model1, self).__init__()
        self.H = H
        self.W = W
        self.T_out = T_out

        # === Branch A: 空间预测分支 ===
        # 注意：这里 T_out 传入真实的预测步长
        self.spatial_branch = SpatialPredictor(
            T_in=T_in,
            T_out=T_out,  # <--- 直接预测未来
            H=H, W=W,
            d_model=d_model,
            n_heads=n_heads,
            dropout=dropout,
            cycle=24
        )

        # === Branch B: 时间预测分支 ===
        self.temporal_branch = SegRNNGrid(
            Tin=T_in,
            Tout=T_out,  # <--- 直接预测未来
            H=H, W=W,
            d_model=d_model,
            seg_len=seg_len,
            dropout=dropout
        )

        # === Fusion: 可学习的融合权重 ===
        # 初始值设为 0 (sigmoid后为 0.5)，表示两边各占 50%
        # self.fusion_weight = nn.Parameter(torch.zeros(1))
        # # 替换原来的 self.fusion_weight
        # self.fusion_layer = nn.Parameter(torch.zeros(1, 1, 1, H, W))
        self.fusion_layer = GatedMLPFusion(hidden_dim=32)
    def forward(self, x):
        """
        Input x: (B, T_in, 1, H, W)
        Output: (B, T_out, 1, H, W)
        """
        # 1. 维度适配 (B, T, 1, H, W) -> (B, T, H, W) 用于空间分支内部处理
        # 但 SegRNN 需要带 channel 的，所以我们在传入各分支时单独处理

        # === A: 空间分支预测 ===
        # 输入: (B, T_in, 1, H, W) (内部会自动 squeeze)
        # 输出: (B, T_out, H, W)
        out_spatial = self.spatial_branch(x)

        # 增加 Channel 维度: (B, T_out, 1, H, W)
        if out_spatial.dim() == 4:
            out_spatial = out_spatial.unsqueeze(2)

        # === B: 时间分支预测 ===
        # 输入: (B, T_in, 1, H, W)
        # 输出: (B, T_out, N) 或者 (B, T_out, H, W)
        # (根据你最新的代码 SegRNNGrid 输出是平铺的数值)
        out_temporal = self.temporal_branch(x)  # (B, Tout, N)
        # 关键步骤：展平为 (B, T, N) 以便和 MLP 对接
        # out_spatial_flat = out_spatial.reshape(-1, self.T_out, self.N)

        # # Reshape & Unsqueeze: (B, T_out, 1, H, W)
        # out_temporal = out_temporal.view(-1, self.T_out, self.H, self.W).unsqueeze(2)
        final_flat = self.fusion_layer(out_spatial, out_temporal)
        # === Fusion: 加权融合 ===
        # # 限制 w 在 0~1 之间
        # w = torch.sigmoid(self.fusion_weight)
        # === Fusion ===
        # w = torch.sigmoid(self.fusion_layer)
        # final_pred = w * out_spatial + (1 - w) * out_temporal
        # 融合公式: y = w * Spatial + (1-w) * Temporal
        # final_pred = w * out_spatial + (1 - w) * out_temporal
        final_pred = final_flat.view(-1, self.T_out, self.H, self.W).unsqueeze(2)

        return final_pred