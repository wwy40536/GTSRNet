#  -*- coding: UTF-8 -*
import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np
import torch
adaptive_filter=True
from torch import nn, einsum, Tensor
from utils.layers.Transformer_EncDec import Encoder, EncoderLayer

from utils. layers.Autoformer_EncDec import series_decomp


class SegRNN(nn.Module):
    """
    Paper link: https://arxiv.org/abs/2308.11200.pdf
    """

    def __init__(self, seq_len=25, enc_in=400, d_model=256, dropout=0.1, task_name='short_term_forecast', pred_len=25, seg_len=5):
        super(SegRNN, self).__init__()

        # Parameters
        self.seq_len = seq_len
        self.enc_in = enc_in
        self.d_model = d_model
        self.dropout = dropout

        self.task_name = task_name
        self.pred_len = pred_len

        self.seg_len = seg_len
        self.seg_num_x = self.seq_len // self.seg_len
        self.seg_num_y = self.pred_len // self.seg_len

        # Building model
        self.valueEmbedding = nn.Sequential(
            nn.Linear(self.seg_len, self.d_model),
            nn.ReLU()
        )
        self.rnn = nn.GRU(input_size=self.d_model, hidden_size=self.d_model, num_layers=1, bias=True,
                          batch_first=True, bidirectional=False)
        self.pos_emb = nn.Parameter(torch.randn(self.seg_num_y, self.d_model // 2))
        self.channel_emb = nn.Parameter(torch.randn(self.enc_in, self.d_model // 2))

        self.predict = nn.Sequential(
            nn.Dropout(self.dropout),
            nn.Linear(self.d_model, self.seg_len)
        )

    def encoder(self, x):
        # b:batch_size c:channel_size s:seq_len s:seq_len
        # d:d_model w:seg_len n:seg_num_x m:seg_num_y
        batch_size = x.size(0)

        # normalization and permute     b,s,c -> b,c,s
        seq_last = x[:, -1:, :].detach()
        x = (x - seq_last).permute(0, 2, 1)  # b,c,s

        # segment and embedding    b,c,s -> bc,n,w -> bc,n,d
        x = self.valueEmbedding(x.reshape(-1, self.seg_num_x, self.seg_len))

        # encoding
        _, hn = self.rnn(x)  # bc,n,d  1,bc,d

        # m,d//2 -> 1,m,d//2 -> c,m,d//2
        # c,d//2 -> c,1,d//2 -> c,m,d//2
        # c,m,d -> cm,1,d -> bcm, 1, d
        pos_emb = torch.cat([
            self.pos_emb.unsqueeze(0).repeat(self.enc_in, 1, 1),
            self.channel_emb.unsqueeze(1).repeat(1, self.seg_num_y, 1)
        ], dim=-1).view(-1, 1, self.d_model).repeat(batch_size, 1, 1)

        _, hy = self.rnn(pos_emb, hn.repeat(1, 1, self.seg_num_y).view(1, -1, self.d_model))  # bcm,1,d  1,bcm,d

        # 1,bcm,d -> 1,bcm,w -> b,c,s
        y = self.predict(hy).view(-1, self.enc_in, self.pred_len)

        # permute and denorm
        y = y.permute(0, 2, 1) + seq_last
        return y

    def forecast(self, x_enc):
        # Encoder
        return self.encoder(x_enc)

    def forward(self, x_enc):
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            dec_out = self.forecast(x_enc)
            return dec_out[:, -self.pred_len:, :]  # [B, L, D]

        return None

class EMA(nn.Module):
    def __init__(self, channels, factor=16):
        super(EMA, self).__init__()
        self.groups = factor
        assert channels // self.groups > 0
        self.softmax = nn.Softmax(-1)
        self.agp = nn.AdaptiveAvgPool2d((1, 1))
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        self.gn = nn.GroupNorm(channels // self.groups, channels // self.groups)
        self.conv1x1 = nn.Conv2d(channels // self.groups, channels // self.groups, kernel_size=1, stride=1, padding=0)
        self.conv3x3 = nn.Conv2d(channels // self.groups, channels // self.groups, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        b, c, h, w = x.size()
        group_x = x.reshape(b * self.groups, -1, h, w)  # b*g,c//g,h,w
        x_h = self.pool_h(group_x)
        x_w = self.pool_w(group_x).permute(0, 1, 3, 2)
        hw = self.conv1x1(torch.cat([x_h, x_w], dim=2))
        x_h, x_w = torch.split(hw, [h, w], dim=2)
        x1 = self.gn(group_x * x_h.sigmoid() * x_w.permute(0, 1, 3, 2).sigmoid())
        x2 = self.conv3x3(group_x)
        x11 = self.softmax(self.agp(x1).reshape(b * self.groups, -1, 1).permute(0, 2, 1))
        x12 = x2.reshape(b * self.groups, c // self.groups, -1)  # b*g, c//g, hw
        x21 = self.softmax(self.agp(x2).reshape(b * self.groups, -1, 1).permute(0, 2, 1))
        x22 = x1.reshape(b * self.groups, c // self.groups, -1)  # b*g, c//g, hw
        weights = (torch.matmul(x11, x12) + torch.matmul(x21, x22)).reshape(b * self.groups, 1, h, w)
        return (group_x * weights.sigmoid()).reshape(b, c, h, w)
class moving_avg(nn.Module):
    """
    Moving average block to highlight the trend of time series
    """
    def __init__(self, kernel_size, stride):
        super(moving_avg, self).__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x):
        # padding on the both ends of time series
        front = x[:, 0:1, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        end = x[:, -1:, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        x = torch.cat([front, x, end], dim=1)
        x = self.avg(x.permute(0, 2, 1))
        x = x.permute(0, 2, 1)
        return x
class _ScaleModule(nn.Module):
    def __init__(self, dims, init_scale=1.0, init_bias=0):
        super(_ScaleModule, self).__init__()
        self.dims = dims
        self.weight = nn.Parameter(torch.ones(*dims) * init_scale)
        self.bias = None

    def forward(self, x):
        return torch.mul(self.weight, x)
class Flatten_Head(nn.Module):
    def __init__(self, individual, n_vars, target_window, context_window, head_dropout=0):
        super().__init__()

        self.individual = individual
        self.n_vars = n_vars
        self.linear = nn.Linear(context_window, target_window)
        self.dropout = nn.Dropout(head_dropout)

    def forward(self, x):
        x = self.linear(x)
        x = self.dropout(x)
        return x
class series_decomp(nn.Module):
    """
    Series decomposition block
    """

    def __init__(self, kernel_size):
        super(series_decomp, self).__init__()
        self.moving_avg = moving_avg(kernel_size, stride=1)

    def forward(self, x):
        moving_mean = self.moving_avg(x)
        res = x - moving_mean
        return res, moving_mean
class Block(nn.Module):
    def __init__(self,c_in,context_window, dropout=0., n_layers=1):
        super().__init__()

        self.layers = nn.ModuleList([EncoderLayer(c_in,context_window,dropout) for i in range(n_layers)])
    def forward(self, src:Tensor):
        output = src
        for mod in self.layers: output = mod(output)
        return output
class FTblock(nn.Module):
    def __init__(self, c_in, context_window, n_layers=3, d_model=128, dropout=0.):
        super().__init__()

        self.dropout = nn.Dropout(dropout)

        self.encoder = Block(c_in, context_window, dropout=dropout, n_layers=n_layers)

    def forward(self, x) -> Tensor:
        x = x.permute(0, 2, 1)
        z = self.encoder(x)

        return z
class DFT_series_decomp(nn.Module):
    """
    Series decomposition block
    """
    def __init__(self, top_k=5):
        super(DFT_series_decomp, self).__init__()
        self.top_k = top_k

    def forward(self, x):
        xf = torch.fft.rfft(x)
        freq = abs(xf)
        freq[0] = 0
        top_k_freq, top_list = torch.topk(freq, self.top_k)
        xf[freq <= top_k_freq.min()] = 0
        x_season = torch.fft.irfft(xf)
        x_trend = x - x_season
        return x_season, x_trend

class Filter(nn.Module):
    def __init__(self, feature_num, sequence_len):
        super().__init__()
        self.complex_weight = nn.Parameter(torch.randn(sequence_len, 2, dtype=torch.float32)).contiguous()
        self.sequence_len = sequence_len
    def forward(self, x):

        weight = torch.view_as_complex(self.complex_weight).contiguous()
        x = torch.fft.fft (x,norm='ortho',dim = -2)
        x = x * weight
        x = torch.fft.ifft(x,norm='ortho',dim = -2)
        x = x.real
        return x
class Temporalmodule(nn.Module):
    def __init__(self, d_model,d_model1, mlp_ratio=0.25, act_layer=nn.GELU, skip_connect=False,dropout=0.):
        super().__init__()
        self.bn = nn.BatchNorm1d(d_model1)
        self.bn1 = nn.BatchNorm1d(d_model1)
        self.bn2 = nn.BatchNorm1d(d_model1)
        self.bn3 = nn.BatchNorm1d(d_model1)
        self.skip_connect = skip_connect
        hidden_features = int(d_model * mlp_ratio)
        self.act = act_layer()
        self.T_fc1 = nn.Linear(d_model,hidden_features)
        self.T_fc2 = nn.Linear(hidden_features, d_model)
        self.dropout = nn.Dropout(dropout)
        self.filter = Filter(d_model1,d_model)
    def forward(self, x):
        xs = self.bn(x)
        xs = self.filter(xs).contiguous()
        xs = self.bn3(xs)
        xs = self.T_fc1(xs)
        xs = self.act(xs)
        xs = self.dropout(xs)
        xs = self.bn1(xs)
        xs = self.T_fc2(xs)
        xs = self.dropout(xs)
        xs = self.bn2(xs)
        if self.skip_connect:
            x = x + xs
        else:
            x = xs
        return x
class Featuremodule(nn.Module):
    def __init__(self, d_model,d_model1, mlp_ratio=0.25, act_layer=nn.GELU, skip_connect=False,dropout=0.):
        super().__init__()
        self.bn = nn.BatchNorm1d(d_model)
        self.bn1 = nn.BatchNorm1d(d_model)
        self.skip_connect = skip_connect
        hidden_features = int(d_model * mlp_ratio)
        self.act = act_layer()
        self.squeeze = nn.AdaptiveAvgPool1d(1)
        self.excitation = nn.Sequential(
            nn.Linear(d_model, hidden_features),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_features, d_model),
            nn.Sigmoid()
        )

    def forward(self, x):
        batch_size, num_channels, seq_length = x.size()
        squeezed = self.squeeze(x).view(batch_size, num_channels)
        excitation = self.excitation(squeezed)
        excitation = excitation.view(batch_size, num_channels, 1)
        xs = x * excitation
        if self.skip_connect:
            x = x + xs
        else:
            x = xs
        return x
class EncoderLayer(nn.Module):
    def __init__(self,c_in,context_window, dropout=0.):
        super().__init__()
        self.featuremodule = Featuremodule(c_in,context_window,skip_connect=False,dropout=dropout)
        self.temporalmodule = Temporalmodule(context_window,c_in,skip_connect=False,dropout=dropout)

        self.dropout_attn = nn.Dropout(dropout)

        self.dropout_ffn = nn.Dropout(dropout)
    def forward(self, src:Tensor) -> Tensor:

        x = self.featuremodule(src)
        x = self.temporalmodule(x)
        x = x + src
        return x

class ResDWC(nn.Module):
    def __init__(self, dim, kernel_size=3):
        super().__init__()
        self.dim = dim
        self.kernel_size = kernel_size

        self.act = nn.ReLU(inplace=True)

        # 定义不同尺度的卷积层
        self.conv1 = nn.Conv2d(dim, dim, kernel_size, padding=kernel_size // 2, groups=dim, bias=True)
        self.conv2 = nn.Conv2d(dim, dim, kernel_size, padding=kernel_size // 2, groups=dim, bias=True)
        self.conv3 = nn.Conv2d(dim, dim, kernel_size, padding=kernel_size // 2, groups=dim, bias=True)

    def forward(self, x):


        # 计算不同尺度的卷积
        x1 = self.act(self.conv1(x))
        x2 = self.act(self.conv2(nn.functional.avg_pool2d(x, kernel_size=2)))  # 下采样
        x2 = nn.functional.interpolate(x2, size=x.size()[2:], mode='bilinear', align_corners=False)  # 调整尺寸

        x3 = self.act(
            self.conv3(nn.functional.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)))  # 上采样
        x3 = nn.functional.avg_pool2d(x3, kernel_size=2)  # 调整尺寸
        # print(f"x shape: {x.shape}, x1 shape: {x1.shape}, x2 shape: {x2.shape}, x3 shape: {x3.shape}")
        # 融合不同尺度的特征
        return x + x1 + x2 + x3
class ResDWC11(nn.Module):
    def __init__(self, dim, kernel_size=3):
        super().__init__()
        self.dim = dim
        self.kernel_size = kernel_size
        #
        self.act = nn.ReLU(inplace=True)
        #
        self.conv = nn.Conv2d(dim, dim, kernel_size, 1, kernel_size // 2, groups=dim,bias=True)
    def forward(self, x):
        return x + self.act(self.conv(x))
class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features = out_features or in_features
        self.hidden_features = hidden_features = hidden_features or in_features
        # 使用1x1卷积进行全连接层的实现
        self.fc1 = nn.Conv2d(in_features, hidden_features, 1)
        self.act1 = act_layer()
        self.fc2 = nn.Conv2d(hidden_features, out_features, 1)
        self.drop = nn.Dropout(drop)
        # 添加 ResDWC 模块
        self.conv = ResDWC(hidden_features, 3)

    def forward(self, x):
        # 第一层卷积 + 激活 + Dropout
        x = self.fc1(x)
        x = self.act1(x)
        x = self.drop(x)

        # 通过 ResDWC 模块
        x = self.conv(x)
        # 第二层卷积 + Dropout
        x = self.fc2(x)
        x = self.drop(x)
        return x
class MY333(nn.Module):
    def __init__(self, input_shape, meta_shape, cross_shape, num_init_features=12, growth_rate=12,bn_size=4,
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
        self. segRNN = SegRNN(seq_len=25, enc_in=400, d_model=128, dropout=0.2, task_name='short_term_forecast', pred_len=25,
                      seg_len=25)
        self.decomp_module = series_decomp(25)
        self.CONglu =Mlp(25, 200)
    def forward(self, out, meta=None, cross=None):
        batch_size, seq_len, features, row_of_cells, col_of_cells = out.shape
        #   (b,s,f,20,20)
        out = out.squeeze(2)
        # out = self.xb(out)
        out = out.view(batch_size, seq_len, features, row_of_cells * col_of_cells).transpose(2, 3).squeeze(
            -1)  #       # (32, 25, 400)
        out = self.segRNN (out)
        out=out.view(batch_size, -1, row_of_cells, col_of_cells)
        # out = self.CONglu(out)
        return out
if __name__ == '__main__':
    device = torch.device('cpu')
    input = torch.randn(128, 25, 1, 20, 20).to(device)
    mete = torch.randn(32, 4).to(device)  #小时的特征
    cros = torch.randn(32, 4, 20, 20)  #其他特征
    net = MY333(input.shape, mete.shape, cros.shape).to(device)
    out = net(input)
    print(out.shape)
