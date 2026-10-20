#  -*- coding: UTF-8 -*
import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np
import torch

adaptive_filter = True



class FreTS(nn.Module):
    def __init__(self, pred_len, enc_in, seq_len, channel_independence):
        super(FreTS, self).__init__()
        self.embed_size = 128  #embed_size
        self.hidden_size = 256  #hidden_size
        self.pre_length = pred_len
        self.feature_size = enc_in  #channels
        self.seq_length = seq_len
        self.channel_independence = channel_independence
        self.sparsity_threshold = 0.01
        self.scale = 0.02
        self.embeddings = nn.Parameter(torch.randn(1, self.embed_size))
        self.r1 = nn.Parameter(self.scale * torch.randn(self.embed_size, self.embed_size))
        self.i1 = nn.Parameter(self.scale * torch.randn(self.embed_size, self.embed_size))
        self.rb1 = nn.Parameter(self.scale * torch.randn(self.embed_size))
        self.ib1 = nn.Parameter(self.scale * torch.randn(self.embed_size))
        self.r2 = nn.Parameter(self.scale * torch.randn(self.embed_size, self.embed_size))
        self.i2 = nn.Parameter(self.scale * torch.randn(self.embed_size, self.embed_size))
        self.rb2 = nn.Parameter(self.scale * torch.randn(self.embed_size))
        self.ib2 = nn.Parameter(self.scale * torch.randn(self.embed_size))

        self.fc = nn.Sequential(
            nn.Linear(self.seq_length * self.embed_size, self.hidden_size),
            nn.LeakyReLU(),
            nn.Linear(self.hidden_size, self.pre_length)
        )
        self.residual_block = nn.Sequential(
            nn.Conv2d(25, self.feature_size, kernel_size=1, padding=0),
            nn.BatchNorm2d(self.feature_size),
            nn.ReLU(),
            nn.Conv2d(self.feature_size, 25, kernel_size=1, padding=0),
            nn.BatchNorm2d(25)
        )

    def tokenEmb(self, x):
        # x: [Batch, Input length, Channel]
        x = x.permute(0, 2, 1)
        x = x.unsqueeze(3)
        # N*T*1 x 1*D = N*T*D
        y = self.embeddings
        return x * y

        # frequency temporal learner

    def MLP_temporal(self, x, B, N, L):
        # [B, N, T, D]
        x = torch.fft.rfft(x, dim=2, norm='ortho')  # FFT on L dimension
        y = self.FreMLP(B, N, L, x, self.r2, self.i2, self.rb2, self.ib2)
        x = torch.fft.irfft(y, n=self.seq_length, dim=2, norm="ortho")
        return x

        # frequency channel learner

    def MLP_channel(self, x, B, N, L):
        # [B, N, T, D]
        x = x.permute(0, 2, 1, 3)
        # [B, T, N, D]
        x = torch.fft.rfft(x, dim=2, norm='ortho')  # FFT on N dimension
        y = self.FreMLP(B, L, N, x, self.r1, self.i1, self.rb1, self.ib1)
        x = torch.fft.irfft(y, n=self.feature_size, dim=2, norm="ortho")
        x = x.permute(0, 2, 1, 3)
        # [B, N, T, D]
        return x

        # frequency-domain MLPs
        # dimension: FFT along the dimension, r: the real part of weights, i: the imaginary part of weights
        # rb: the real part of bias, ib: the imaginary part of bias

    def FreMLP(self, B, nd, dimension, x, r, i, rb, ib):
        o1_real = torch.zeros([B, nd, dimension // 2 + 1, self.embed_size],
                              device=x.device)
        o1_imag = torch.zeros([B, nd, dimension // 2 + 1, self.embed_size],
                              device=x.device)

        o1_real = F.relu(
            torch.einsum('bijd,dd->bijd', x.real, r) - \
            torch.einsum('bijd,dd->bijd', x.imag, i) + \
            rb
        )

        o1_imag = F.relu(
            torch.einsum('bijd,dd->bijd', x.imag, r) + \
            torch.einsum('bijd,dd->bijd', x.real, i) + \
            ib
        )

        y = torch.stack([o1_real, o1_imag], dim=-1)
        y = F.softshrink(y, lambd=self.sparsity_threshold)
        y = torch.view_as_complex(y)
        return y

    # frequency-domain MLPs
    # dimension: FFT along the dimension, r: the real part of weights, i: the imaginary part of weights
    # rb: the real part of bias, ib: the imaginary part of bias

    def forward(self, x):
        # x: [Batch, Input length, Channel]
        B, T, N = x.shape
        # embedding x: [B, N, T, D]
        x = self.tokenEmb(x)
        # [B, N, T, D]
        if self.channel_independence == '1':
            x = self.MLP_channel(x, B, N, T)
        # [B, N, T, D]
        x = self.MLP_temporal(x, B, N, T)
        x = self.fc(x.reshape(B, N, -1)).permute(0, 2, 1)
        return x
class Frets(nn.Module):
    def __init__(self, input_shape, meta_shape, cross_shape, num_init_features=12, growth_rate=12, bn_size=4,
                 drop_rate=0.1, nb_flows=25, fusion=1, maps=1, d_model=25, pred_len=25, enc_in=400, seq_len=25,
                 channel_independence='1',
                 flags_meta=0, flags_cross=0):
        super().__init__()
        self.input_shape = input_shape
        self.meta_shape = meta_shape
        self.cross_shape = cross_shape
        self.filters = num_init_features
        self.channels = nb_flows
        self.fusion = fusion
        self.maps = maps
        self.h, self.w = self.input_shape[-2], self.input_shape[-1]
        self.inner_shape = self.input_shape[:2] + (self.filters,) + self.input_shape[-2:]
        self.seq_len = self.input_shape[1]
        self.d_model = d_model
        self.model4 = FreTS(pred_len=pred_len, enc_in=enc_in, seq_len=seq_len,
                            channel_independence=channel_independence)

    def forward(self, out, meta=None, cross=None):
        batch_size, seq_len, features, row_of_cells, col_of_cells = out.shape
        #   (b,s,f,20,20)
        out = out.squeeze(2)
        out = out.view(batch_size, seq_len, features, row_of_cells * col_of_cells).transpose(2, 3).squeeze(
            -1)  #       # (32, 25, 400)
        out = self.model4(out)
        out = out.view(batch_size, -1, row_of_cells, col_of_cells)
        return out


if __name__ == '__main__':
    device = torch.device('cpu')
    input = torch.randn(128, 25, 1, 20, 20).to(device)
    mete = torch.randn(32, 4).to(device)  #小时的特征
    cros = torch.randn(32, 4, 20, 20)  #其他特征
    net = STMLP(input.shape, mete.shape, cros.shape).to(device)
    out = net(input)
    print(out.shape)
