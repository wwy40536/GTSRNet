import torch
import torch.nn as nn
import torch.nn.functional as F

# ================================================================
# 1. 导入依赖
# ================================================================
# 假设你的 SegRNNGrid 在 utils.segrnn_gridB (根据你之前的代码路径)

from utils.layers.Transformer_EncDec import Encoder, EncoderLayer
from utils.layers.SelfAttention_Family import FullAttention, AttentionLayer
from utils.layers.Embed import DataEmbedding_inverted
from utils.segrnn_gridB import SegRNNGrid


# ================================================================
# 2. 你的改进版 iTransformer (作为空间模块)
#    (完全保留你提供的代码逻辑)
# ================================================================
class SpatialEnhancedModel(nn.Module):
    def __init__(self, T_in, T_out, H, W, d_model=64, cycle=24,
                 n_heads=4, e_layers=2, d_ff=256, dropout=0.1, activation='gelu',
                 embed='timeF', freq='h', factor=1, output_attention=False,
                 use_norm=True, output_proj_dropout=0.1):
        super(SpatialEnhancedModel, self).__init__()

        # 1. 绑定参数
        self.seq_len = T_in
        self.pred_len = T_out
        self.H = H
        self.W = W
        self.enc_in = H * W
        self.cycle_len = cycle
        self.d_model = d_model
        self.output_attention = output_attention
        self.use_norm = use_norm

        # Embedding
        self.enc_embedding = DataEmbedding_inverted(self.seq_len, self.d_model, embed, freq, dropout)

        # Encoder
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

        self.projector = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Dropout(output_proj_dropout),
            nn.Linear(d_model * 2, d_model * 4),
            nn.GELU(),
            nn.Dropout(output_proj_dropout),
            nn.Linear(d_model * 4, self.pred_len),
        )

        # 自定义 Embedding
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

        # Embedding
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

        x_mark_enc = None  # 强制 None，由 DataEmbedding_inverted 处理

        if cycle_index is None:
            cycle_index = torch.zeros(B, dtype=torch.long).to(x.device)

        dec_out, attns = self.forecast(x_enc, x_mark_enc, cycle_index)

        # Reshape Back
        dec_out = dec_out.reshape(B, self.pred_len, H, W)

        if self.output_attention:
            return dec_out, attns
        else:
            return dec_out


# ================================================================
# 3. 串联模型: Serial_ST_Model
#    架构: Input -> [你的改进版iTransformer] -> Residual -> [SegRNN] -> Output
# ================================================================
class Serial_ST_Model(nn.Module):
    def __init__(self, T_in, T_out, H, W, d_model=128, seg_len=6, dropout=0.1, n_heads=4):
        super(Serial_ST_Model, self).__init__()
        self.H = H
        self.W = W
        self.T_in = T_in
        self.T_out = T_out

        # === 1. 空间模块 (使用你的改进版 iTransformer) ===
        # ⚠️ 关键点：
        # 我们希望它作为特征增强器，输入 T_in，输出 T_in。
        # 所以这里初始化时，第二个参数 (预测长度) 必须传 T_in，而不是 T_out。
        self.spatial_encoder = SpatialEnhancedModel(
            T_in=T_in,
            T_out=T_in,  # <--- 强制输出长度等于输入长度，以便做残差连接
            H=H, W=W,
            d_model=d_model,
            n_heads=n_heads,
            dropout=dropout,
            cycle=24  # 周期参数
        )

        # === 2. 时间模块 (SegRNNGrid) ===
        # 负责真正的预测：从 T_in 推演到 T_out
        self.temporal_predictor = SegRNNGrid(
            Tin=T_in,
            Tout=T_out,  # <--- 这里才是真正的预测长度
            H=H, W=W,
            d_model=d_model,
            seg_len=seg_len,
            dropout=dropout
        )

        # === 3. 最终投影层 ===
        # SegRNN 的输出通常是 d_model 维度的特征
        self.final_projector = nn.Linear(d_model, 1)

    def forward(self, x):
        """
        Input x: (B, T_in, 1, H, W) 或 (B, T_in, H, W)
        Output: (B, T_out, 1, H, W)
        """
        # 维度标准化: (B, T, 1, H, W)
        if x.dim() == 4:
            x = x.unsqueeze(2)

        # === Step 1: 空间增强 ===
        # 调用你的 Transformer 模型
        # 输入: (B, T_in, 1, H, W) -> 内部自动处理 squeeze -> 输出: (B, T_in, H, W)
        x_spatial = self.spatial_encoder(x)

        # === Step 2: 残差连接 ===
        # 原始输入 x 是 (B, T, 1, H, W)，需要 squeeze 掉 channel 才能和 x_spatial 相加
        x_squeezed = x.squeeze(2)
        x_enhanced = x_squeezed + x_spatial  # (B, T_in, H, W)

        # 变回 (B, T_in, 1, H, W) 喂给 SegRNN
        x_enhanced = x_enhanced.unsqueeze(2)

        # === Step 3: 时间推演 (SegRNN) ===
        # SegRNN 输入: (B, T_in, 1, H, W)
        # SegRNN 输出特征: (B, T_out, N, D)  (假设 SegRNNGrid 返回特征)
        # 注意：如果你的 SegRNNGrid 直接返回数值结果，这步之后就不需要 final_projector 了
        # 这里假设它返回的是特征 feats
        feats = self.temporal_predictor(x_enhanced)

        # === Step 4: 最终输出 ===
        # (B, T_out, N, D) -> (B, T_out, N, 1)
        pred = self.final_projector(feats)

        # Reshape: (B, T_out, N, 1) -> (B, T_out, 1, H, W)
        pred = pred.squeeze(-1)  # (B, T_out, N)
        pred = pred.reshape(pred.shape[0], pred.shape[1], self.H, self.W).unsqueeze(2)

        return pred