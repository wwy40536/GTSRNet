import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# 假设您的文件路径如下，请根据实际情况调整
from utils. layers.Transformer_EncDec import Encoder, EncoderLayer
from utils. layers.SelfAttention_Family import FullAttention, AttentionLayer
from utils. layers.Embed import DataEmbedding_inverted


class Model(nn.Module):
    def __init__(self, T_in, T_out, H, W, d_model=64, cycle=24,
                 n_heads=4, e_layers=2, d_ff=256, dropout=0.1, activation='gelu',
                 embed='timeF', freq='h', factor=1, output_attention=False,
                 use_norm=True, output_proj_dropout=0.1):
        """
        改为直接传参模式：Model(24, 12, 20, 20, ...)
        """
        super(Model, self).__init__()

        # 1. 绑定参数 (按照您的要求修改)
        self.seq_len = T_in  # 对应 opt.close_size
        self.pred_len = T_out  # 对应 opt.predict_size
        self.H = H
        self.W = W
        self.enc_in = H * W  # 自动计算变量数量
        self.cycle_len = cycle  # 周期长度
        self.d_model = d_model
        self.output_attention = output_attention
        self.use_norm = use_norm

        # Embedding
        self.enc_embedding = DataEmbedding_inverted(self.seq_len, self.d_model, embed, freq, dropout)

        # Encoder-only architecture
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

        # 您的自定义 Embedding
        self.channel_embedding = nn.Parameter(torch.zeros(self.enc_in, d_model))
        self.phase_embedding = nn.Embedding(self.cycle_len, d_model)
        nn.init.xavier_normal_(self.phase_embedding.weight)

        self.joint_embedding = nn.Embedding(self.cycle_len, self.enc_in * d_model)
        nn.init.xavier_normal_(self.joint_embedding.weight)
        nn.init.xavier_normal_(self.channel_embedding)

    # forecast 保持原样不动
    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec, phase):



        if self.use_norm:
            # Normalization from Non-stationary Transformer
            means = x_enc.mean(1, keepdim=True).detach()
            x_enc = x_enc - means
            stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
            x_enc /= stdev

        B, L, N = x_enc.shape  # B L N

        # Embedding
        enc_out = self.enc_embedding(x_enc, x_mark_enc)

        # Get the four embeddings
        channel_emb = self.channel_embedding.expand(enc_out.shape[0], N, -1)
        phase_emb = self.phase_embedding(phase.view(-1, 1).expand(B, N))

        # 注意：这里 reshape 需要用到 self.enc_in 和 self.d_model
        joint_emb = self.joint_embedding(phase).reshape(B, self.enc_in, self.d_model)

        enc_out = enc_out[:, :N, :] + channel_emb + phase_emb + joint_emb
        enc_orgin = enc_out

        enc_out, attns = self.encoder(enc_out, attn_mask=None)
        dec_out = self.projector(enc_out + enc_orgin).permute(0, 2, 1)[:, :, :N]

        if self.use_norm:
            # De-Normalization from Non-stationary Transformer
            dec_out = dec_out * (stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
            dec_out = dec_out + (means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))

        return dec_out, attns

    # forward 改为适配您的输入 [B, T, H, W]
    def forward(self, x, x_mark_enc=None, cycle_index=None, *args, **kwargs):

        if x.dim() == 5:
            x = x.squeeze(2)  # [B, T, 1, H, W] -> [B, T, H, W]
        """
        x: [B, T, H, W]
        """
        # 1. 维度调整 [B, T, H, W] -> [B, T, N]
        B, T, H, W = x.shape
        x_enc = x.reshape(B, T, -1)  # Flatten H*W -> N

        # 2. 自动处理 x_mark (设为 None)
        x_mark_enc = None
        x_dec = None
        x_mark_dec = None

        # 3. 自动生成 cycle_index (phase)
        # forecast 里面 phase 需要是 [B]，用于 joint_embedding(phase) 查找
        # 我们生成全 0 的索引作为默认 phase
        if cycle_index is None:
            cycle_index = torch.zeros(B, dtype=torch.long).to(x.device)

        # 4. 调用核心 forecast
        dec_out, attns = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec, cycle_index)

        # 5. 还原维度 [B, T_out, N] -> [B, T_out, H, W]
        dec_out = dec_out.reshape(B, self.pred_len, H, W)

        if self.output_attention:
            return dec_out, attns
        else:
            return dec_out