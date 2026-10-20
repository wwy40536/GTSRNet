#  -*- coding: UTF-8 -*
import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np
from collections import OrderedDict
import math
from math import ceil
import torch
import lightning as L
from timm.models.layers import trunc_normal_
adaptive_filter=True
from torch import nn, einsum, Tensor
from pytorch_wavelets import DWTForward
from utils.layers.Embed import PatchEmbedding
from utils.layers.SelfAttention_Family import AttentionLayer, FullAttention
from utils.layers.Transformer_EncDec import Encoder, EncoderLayer
from einops import rearrange, reduce
from torch.nn import Conv2d
from torch.nn import  MultiheadAttention
from einops import rearrange
class EMA(nn.Module):
    def __init__(self, channels, factor=3):
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
import torch.nn.functional as F
class _ScaleModule(nn.Module):
    def __init__(self, dims, init_scale=1.0, init_bias=0):
        super(_ScaleModule, self).__init__()
        self.dims = dims
        self.weight = nn.Parameter(torch.ones(*dims) * init_scale)
        self.bias = None

    def forward(self, x):
        return torch.mul(self.weight, x)

class InceptionDWConv2d(nn.Module):
    """ Inception depthweise convolution
    """

    def __init__(self, in_channels, square_kernel_size=3, band_kernel_size=11, branch_ratio=0.125):
        super().__init__()

        gc = int(in_channels * branch_ratio)  # channel numbers of a convolution branch
        self.dwconv_hw = nn.Conv2d(gc, gc, square_kernel_size, padding=square_kernel_size // 2, groups=gc)
        self.dwconv_w = nn.Conv2d(gc, gc, kernel_size=(1, band_kernel_size), padding=(0, band_kernel_size // 2),
                                  groups=gc)
        self.dwconv_h = nn.Conv2d(gc, gc, kernel_size=(band_kernel_size, 1), padding=(band_kernel_size // 2, 0),
                                  groups=gc)
        self.split_indexes = (in_channels - 3 * gc, gc, gc, gc)

    def forward(self, x):
        x_id, x_hw, x_w, x_h = torch.split(x, self.split_indexes, dim=1)
        return torch.cat(
            (x_id, self.dwconv_hw(x_hw), self.dwconv_w(x_w), self.dwconv_h(x_h)),
            dim=1,
        )
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
class FTMLP_backbone(nn.Module):
    def __init__(self, c_in: int, context_window: int, target_window: int, patch_len: int, stride: int,
                 n_layers: int = 3, d_model=128, dropout: float = 0., fc_dropout: float = 0., head_dropout=0,
                 padding_patch=None,
                 pretrain_head: bool = False, head_type='flatten', individual=False, revin=True, affine=True,
                 subtract_last=False):

        super().__init__()

        # RevIn
        self.revin = revin
        if self.revin: self.revin_layer = RevIN(c_in, affine=affine, subtract_last=subtract_last)

        # Backbone
        self.backbone = FTblock(c_in=c_in, context_window=context_window, n_layers=n_layers, d_model=d_model,
                                dropout=dropout)

        self.n_vars = c_in
        self.pretrain_head = pretrain_head
        self.head_type = head_type
        self.individual = individual

        if self.pretrain_head:
            self.head = self.create_pretrain_head(self.head_nf, c_in, fc_dropout)
        elif head_type == 'flatten':
            self.head = Flatten_Head(self.individual, self.n_vars, target_window, context_window,
                                     head_dropout=head_dropout)
            # self.head = Flatten_Head(self.individual, self.n_vars, 25, context_window,
            #                          head_dropout=head_dropout)

    def forward(self, z):
        # norm
        if self.revin:
            z = z.permute(0, 2, 1)  # b,l,f 32 25 ,400
            z = self.revin_layer(z, 'norm')
        else:
            z = z.permute(0, 2, 1)

        z = self.backbone(z)#32 400 25
        # model
        z = self.head(z)

        # denorm
        if self.revin:
            z = z.permute(0, 2, 1)
            z = self.revin_layer(z, 'denorm')
            z = z.permute(0, 2, 1)
        return z

    def create_pretrain_head(self, head_nf, vars, dropout):
        return nn.Sequential(nn.Dropout(dropout),
                             nn.Conv1d(head_nf, vars, 1)
                             )
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

class RevIN(nn.Module):
    def __init__(self, num_features: int, eps=1e-5, affine=True, subtract_last=False):
        """
        :param num_features: the number of features or channels
        :param eps: a value added for numerical stability
        :param affine: if True, RevIN has learnable affine parameters
        """
        super(RevIN, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        self.subtract_last = subtract_last
        if self.affine:
            self._init_params()

    def forward(self, x, mode:str):
        if mode == 'norm':
            self._get_statistics(x)
            x = self._normalize(x)
        elif mode == 'denorm':
            x = self._denormalize(x)
        else: raise NotImplementedError
        return x

    def _init_params(self):
        # initialize RevIN params: (C,)
        self.affine_weight = nn.Parameter(torch.ones(self.num_features))
        self.affine_bias = nn.Parameter(torch.zeros(self.num_features))

    def _get_statistics(self, x):
        dim2reduce = tuple(range(1, x.ndim-1))
        if self.subtract_last:
            self.last = x[:,-1,:].unsqueeze(1)
        else:
            self.mean = torch.mean(x, dim=dim2reduce, keepdim=True).detach()
        self.stdev = torch.sqrt(torch.var(x, dim=dim2reduce, keepdim=True, unbiased=False) + self.eps).detach()

    def _normalize(self, x):
        if self.subtract_last:
            x = x - self.last
        else:
            x = x - self.mean
        x = x / self.stdev
        if self.affine:
            x = x * self.affine_weight
            x = x + self.affine_bias
        return x

    def _denormalize(self, x):
        if self.affine:
            x = x - self.affine_bias
            x = x / (self.affine_weight + self.eps*self.eps)
        x = x * self.stdev
        if self.subtract_last:
            x = x + self.last
        else:
            x = x + self.mean
        return x
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
class FreTS(nn.Module):
    def __init__(self, pred_len, enc_in, seq_len, channel_independence):
        super(FreTS, self).__init__()
        self.embed_size =128  #embed_size
        self.hidden_size =256  #hidden_size
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
        self.sd = DFT_series_decomp(5)
        self.revin_layer = RevIN(128, affine=True, subtract_last=False)
        self.decomp_block = DFT_series_decomp(top_k=5)
        self.fc = nn.Sequential(
            nn.Linear(self.seq_length * self.embed_size, self.hidden_size),
            nn.LeakyReLU(),
            nn.Linear(self.hidden_size, self.hidden_size),  # 增加一个隐藏层
            nn.LeakyReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(self.hidden_size, self.pre_length)
        )
        self. EMA = EMA(400)
        self.EMA11 = EMA(3)

    # dimension extension
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
        y = self.FreMLP(B, L, N, x, self.r1, self.i1, self.rb1, self.ib1)#32,25,201,128
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
        #
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

    def forward(self, x):
        x_season, x_trend = self.sd(x)
        # 获取输入形状
        B, T, N = x.shape
        # # x: [Batch, Input length, Channel]
        # B, T, N = x.shape
        # # embedding x: [B, N, T, D]
        # x = self.tokenEmb(x)
        # bias = x
        # # [B, N, T, D]
        # if self.channel_independence == '1':
        #     x = self.MLP_channel(x, B, N, T)
        # # [B, N, T, D]
        # x = self.MLP_temporal(x, B, N, T)
        # x = x + bias
        # x = self.fc(x.reshape(B, N, -1)).permute(0, 2, 1)
        # 对 x_season 和 x_trend 进行特征提取
        # 嵌入特征：[B, N, T, D]
        x_season = self.tokenEmb(x_season)
        x_trend = self.tokenEmb(x_trend)
        # x_season = self.revin_layer(x_season, 'norm')
        # x_trend = self.revin_layer( x_trend, 'norm')
        # # 记录偏置
        bias_season = x_season
        bias_trend = x_trend
        # 分别处理 x_season 和 x_trend
        # 通道独立处理
        if self.channel_independence == '1':
            x_season = self.MLP_channel(x_season, B, N, T)
            x_trend = self.MLP_channel(x_trend, B, N, T)
        # 时间处理
        x_season = self.MLP_temporal(x_season, B, N, T)
        x_trend = self.MLP_temporal(x_trend, B, N, T)
        # x_season = self.revin_layer(x_season, 'denorm')
        # x_trend = self.revin_layer(x_trend, 'denorm')
        # 将处理后的结果相加
        x_season = x_season + bias_season
        x_trend = x_trend + bias_trend

        x_season = self.EMA( x_season)
        x_trend  = self.EMA( x_trend )
        # 将分解后的部分合并为最终的 x，使用加权平均
        alpha = 0.8  # 权重参数
        x = alpha * x_season + (1 - alpha) * x_trend
        # 输出形状调整
        x = self.fc(x.reshape(B, N, -1)).permute(0, 2, 1)
        return x
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




class ResDWC11(nn.Module):
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


class ResDWC(nn.Module):
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

class MY3(nn.Module):
    def __init__(self, input_shape, meta_shape, cross_shape, num_init_features=12, growth_rate=12,bn_size=4,
                 drop_rate=0.1, nb_flows=25, fusion=1, maps=1, d_model=25, pred_len=3, enc_in=400, seq_len=3,
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
        self.model_res = FTMLP_backbone(c_in=400, context_window=25, target_window=25,
                                        patch_len=16, stride=8,
                                        n_layers=3, d_model=128, dropout=0.1,
                                        fc_dropout=0.2, head_dropout=0, padding_patch=None,
                                        head_type='flatten', individual=False, revin=True, affine=True,
                                        subtract_last=False)
        self.h, self.w = self.input_shape[-2], self.input_shape[-1]
        self.inner_shape = self.input_shape[:2] + (self.filters,) + self.input_shape[-2:]
        self.seq_len = self.input_shape[1]
        self.d_model = d_model
        self.model4 = FreTS(pred_len=pred_len, enc_in=enc_in, seq_len=seq_len,channel_independence=channel_independence)
        self.trg_word_prj11 = nn.Linear(1, 25, bias=False)
        self.decomp_module = series_decomp(25)
        self.CONglu =Mlp(3, 128)
    def forward(self, out, meta=None, cross=None):
        batch_size, seq_len, features, row_of_cells, col_of_cells = out.shape
        #   (b,s,f,20,20)
        out = out.squeeze(2)
        # out = self.xb(out)
        out = out.view(batch_size, seq_len, features, row_of_cells * col_of_cells).transpose(2, 3).squeeze(
            -1)  #       # (32, 25, 400)
        out2 = out.view(batch_size, seq_len, features, row_of_cells * col_of_cells).transpose(2, 3)
        out = self.model4(out)

        out=out.view(batch_size, -1, row_of_cells, col_of_cells)
        out = self.CONglu(out)
        return out
if __name__ == '__main__':
    device = torch.device('cpu')
    input = torch.randn(128, 3, 1, 20, 20).to(device)
    mete = torch.randn(32, 4).to(device)  #小时的特征
    cros = torch.randn(32, 4, 20, 20)  #其他特征
    net = MY3(input.shape, mete.shape, cros.shape).to(device)
    out = net(input)
    print(out.shape)  #
