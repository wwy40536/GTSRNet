import torch
import torch.nn as nn
import torch.nn.functional as F
# 假设您已经有了这些官方文件
from utils. layers.Transformer_EncDec import Encoder, EncoderLayer
from utils. layers.SelfAttention_Family import FullAttention, AttentionLayer
from utils. layers.Embed import DataEmbedding_inverted
import numpy as np


class Model(nn.Module):
    """
    修改版 iTransformer:
    1. 支持 (B, T, H, W) 输入
    2. 支持直接参数传递 (T_in, T_out, H, W...) 而非 configs 对象
    """

    def __init__(self, T_in, T_out, H, W, d_model=128, seg_len=6,
                 # 以下是默认参数，你可以根据需要修改
                 n_heads=4, e_layers=2, d_ff=256, dropout=0.1,
                 embed='timeF', freq='h', factor=1, output_attention=False,
                 use_norm=False, class_strategy='projection'):
        super(Model, self).__init__()

        # 1. 绑定参数
        self.seq_len = T_in  # 对应 opt.close_size
        self.pred_len = T_out  # 对应 opt.predict_size
        self.H = H
        self.W = W
        self.output_attention = output_attention
        self.use_norm = use_norm

        # 注意：seg_len 是给 SegRNN 用的，如果是纯 iTransformer 空间模块，这里暂时用不到
        # 但为了不报错，我们先存起来
        self.seg_len = seg_len

        # 2. Embedding
        # 官方 iTransformer 的倒置嵌入
        self.enc_embedding = DataEmbedding_inverted(self.seq_len, d_model, embed, freq, dropout)

        self.class_strategy = class_strategy

        # 3. Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(False, factor, attention_dropout=dropout,
                                      output_attention=output_attention), d_model, n_heads),
                    d_model,
                    d_ff,
                    dropout=dropout,
                    activation='gelu'  # 默认激活函数
                ) for l in range(e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(d_model)
        )

        # 4. Projector
        # 如果是作为 ST-iSeg 的第一阶段（空间特征提取），这里输出长度应该保持 seq_len (即 24)
        # 如果是直接预测未来，这里输出长度就是 pred_len (即 12)
        # 根据您之前的需求（空间模块不改变时间步长），这里我们用 pred_len
        # *请确保您传入的 T_out 如果是做特征提取，要等于 T_in*
        self.projector = nn.Linear(d_model, self.pred_len, bias=True)

    def forecast(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None):

        if x_enc.dim() == 5:
            # 输入是 (Batch, Time, 1, H, W)，squeeze掉第2维
            x_enc = x_enc.squeeze(2)
        # x_enc shape: [Batch, T, H, W]

        # ================= 1. 维度变换 [B, T, H, W] -> [B, T, N] =================
        B, T, H, W = x_enc.shape
        # 强制检查 H 和 W 是否匹配
        assert H == self.H and W == self.W, f"Input shape {H}x{W} does not match model init {self.H}x{self.W}"

        # 展平 H, W 为 Variates (N)
        x_enc = x_enc.reshape(B, T, H * W)

        # Normalization (Non-stationary Transformer logic)
        if self.use_norm:
            means = x_enc.mean(1, keepdim=True).detach()
            x_enc = x_enc - means
            stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
            x_enc /= stdev

        _, _, N = x_enc.shape
        if x_mark_enc is None:
            x_mark_enc = torch.zeros(B, T, 4).to(x_enc.device)
        # ================= 2. iTransformer 核心逻辑 =================
        # Embedding: [B, T, N] -> [B, N, E] (注意这里官方做了倒置)
        enc_out = self.enc_embedding(x_enc, x_mark_enc)

        # Encoder: [B, N, E] -> [B, N, E] (计算空间相关性)
        enc_out, attns = self.encoder(enc_out, attn_mask=None)

        # Projector: [B, N, E] -> [B, N, T_out] -> [B, T_out, N]
        dec_out = self.projector(enc_out).permute(0, 2, 1)[:, :, :N]

        # De-Normalization
        if self.use_norm:
            dec_out = dec_out * (stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
            dec_out = dec_out + (means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))

        # ================= 3. 还原维度 [B, T_out, N] -> [B, T_out, H, W] =================
        dec_out = dec_out.reshape(B, self.pred_len, H, W)

        return dec_out, attns

    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None, mask=None):
        dec_out, attns = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
        if self.output_attention:
            return dec_out, attns
        else:
            return dec_out