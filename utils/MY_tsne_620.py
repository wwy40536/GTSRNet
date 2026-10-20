#  -*- coding: UTF-8 -*
import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np
import torch

adaptive_filter = True
from torch import nn, einsum, Tensor


# class EMA1(nn.Module):
#     def __init__(self, channels, factor=16):
#         super(EMA, self).__init__()
#         self.groups = factor
#         assert channels // self.groups > 0
#         self.softmax = nn.Softmax(-1)
#         self.agp = nn.AdaptiveAvgPool2d((1, 1))
#         self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
#         self.pool_w = nn.AdaptiveAvgPool2d((1, None))
#         self.gn = nn.GroupNorm(channels // self.groups, channels // self.groups)
#         self.conv1x1 = nn.Conv2d(channels // self.groups, channels // self.groups, kernel_size=1, stride=1, padding=0)
#         self.conv3x3 = nn.Conv2d(channels // self.groups, channels // self.groups, kernel_size=3, stride=1, padding=1)
#
#     def forward(self, x):
#         b, c, h, w = x.size()
#         # 分组操作
#         group_x = x.reshape(b * self.groups, -1, h, w)  # b*g,c//g,h,w
#         x_h = self.pool_h(group_x)
#         x_w = self.pool_w(group_x).permute(0, 1, 3, 2)
#         hw = self.conv1x1(torch.cat([x_h, x_w], dim=2))
#         x_h, x_w = torch.split(hw, [h, w], dim=2)
#         x1 = self.gn(group_x * x_h.sigmoid() * x_w.permute(0, 1, 3, 2).sigmoid())
#         x2 = self.conv3x3(group_x)
#         x11 = self.softmax(self.agp(x1).reshape(b * self.groups, -1, 1).permute(0, 2, 1))
#         x12 = x2.reshape(b * self.groups, c // self.groups, -1)  # b*g, c//g, hw
#         x21 = self.softmax(self.agp(x2).reshape(b * self.groups, -1, 1).permute(0, 2, 1))
#         x22 = x1.reshape(b * self.groups, c // self.groups, -1)  # b*g, c//g, hw
#         weights = (torch.matmul(x11, x12) + torch.matmul(x21, x22)).reshape(b * self.groups, 1, h, w)
#         return (group_x * weights.sigmoid()).reshape(b, c, h, w)


class AdaptiveMixPool2d(nn.Module):
    def __init__(self, output_size=(1, 1), alpha=0.5):
        super(AdaptiveMixPool2d, self).__init__()
        self.adaptive_maxpool = nn.AdaptiveMaxPool2d(output_size)
        self.adaptive_avgpool = nn.AdaptiveAvgPool2d(output_size)
        self.alpha = alpha

    def forward(self, x):
        max_pooled = self.adaptive_maxpool(x)
        avg_pooled = self.adaptive_avgpool(x)
        return self.alpha * max_pooled + (1 - self.alpha) * avg_pooled


class EMA(nn.Module):
    def __init__(self, channels, factor=16):
        super(EMA, self).__init__()
        self.groups = factor
        assert channels // self.groups > 0
        self.softmax = nn.Softmax(-1)
        # 使用 AdaptiveMixPool2d 替换 AdaptiveAvgPool2d
        self.agp = AdaptiveMixPool2d(output_size=(1, 1), alpha=0.3)
        # self.agp = nn.AdaptiveAvgPool2d((1, 1))
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        self.gn = nn.GroupNorm(channels // self.groups, channels // self.groups)
        self.conv1x1 = nn.Conv2d(channels // self.groups, channels // self.groups, kernel_size=1, stride=1, padding=0)
        self.conv3x3 = nn.Conv2d(channels // self.groups, channels // self.groups, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        b, c, h, w = x.size()
        # 分组操作
        group_x = x.reshape(b * self.groups, -1, h, w)  # b*g,c//g,h,w
        # 提取 x_h 和 x_w
        x_h = self.pool_h(group_x)
        x_w = self.pool_w(group_x).permute(0, 1, 3, 2)
        hw = self.conv1x1(torch.cat([x_h, x_w], dim=2))
        x_h, x_w = torch.split(hw, [h, w], dim=2)
        x1 = self.gn(group_x * x_h.relu() * x_w.permute(0, 1, 3, 2).relu())
        x2 = self.conv3x3(group_x)
        x11 = self.softmax(self.agp(x1).reshape(b * self.groups, -1, 1).permute(0, 2, 1))
        # print( x11.shape)
        x12 = x2.reshape(b * self.groups, c // self.groups, -1)  # b*g, c//g, hw
        # print(x12.shape)
        x21 = self.softmax(self.agp(x2).reshape(b * self.groups, -1, 1).permute(0, 2, 1))
        # print(x21.shape)
        x22 = x1.reshape(b * self.groups, c // self.groups, -1)  # b*g, c//g, hw
        # print(x22.shape)
        weights = (torch.matmul(x11, x12) + torch.matmul(x21, x22)).reshape(b * self.groups, 1, h, w)
        # print(weights.shape)
        return (group_x * weights.relu()).reshape(b, c, h, w)


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


class FreTS(nn.Module):
    def __init__(self, pred_len, enc_in, seq_len, channel_independence, alpha):
        super(FreTS, self).__init__()
        self.embed_size = 128  #embed_size
        self.hidden_size = 256  #hidden_size
        self.pre_length = pred_len
        self.alpha = alpha
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
        self.sd = series_decomp(3)

        self.fc = nn.Sequential(
            nn.Linear(self.seq_length * self.embed_size, self.hidden_size),
            nn.LeakyReLU(),
            nn.Linear(self.hidden_size, self.hidden_size),  # 增加一个隐藏层
            nn.LeakyReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(self.hidden_size, self.pre_length)
        )

        self.EMA = EMA(400)
        self.mlp = Mlp(self.pre_length)

    def tokenEmb(self, x):
        # x: [Batch, Input length, Channel]
        x = x.permute(0, 2, 1)
        x = x.unsqueeze(3)
        # N*T*1 x 1*D = N*T*D
        y = self.embeddings
        return x * y

    # frequency temporal learner
    def MLP_temporal(self, x, B, N, L):
        #[B, N, T, D]
        x = torch.fft.rfft(x, dim=2, norm='ortho')  # FFT on L dimension
        y = self.FreMLP(B, N, L, x, self.r2, self.i2, self.rb2, self.ib2)
        x = torch.fft.irfft(y, n=self.seq_length, dim=2, norm="ortho")

        # B, N, T, D = x.shape  # 获取输入的维度
        # x = x.view(B * N * T, D)  # 展平为 [B * N * T, D] 与 mlp_temporal 的输入维度匹配
        # y = self.mlp_temporal(x)
        # x = y.view(B, N, T, D)  # 恢复为 [B, N, T, D]
        return x


    # frequency channel learner
    def MLP_channel(self, x, B, N, L):
        #[B, N, T, D]
        x = x.permute(0, 2, 1, 3)
        # [B, T, N, D]
        x = torch.fft.rfft(x, dim=2, norm='ortho')  # FFT on N dimension

        y = self.FreMLP(B, L, N, x, self.r1, self.i1, self.rb1, self.ib1)  #32,25,201,128
        x = torch.fft.irfft(y, n=self.feature_size, dim=2, norm="ortho")
        x = x.permute(0, 2, 1, 3)
        # [B, N, T, D]

        # B, N, T, D = x.shape
        # x = x.permute(0, 2, 1, 3).contiguous()  # [B, T, N, D]
        # x = x.view(B * T * D, N)  # 调整为 [B * T * D, N] 以匹配 mlp_channel 的输入
        # y = self.mlp_channel(x)
        # x = y.view(B, T, N, D).permute(0, 2, 1, 3).contiguous()  # 恢复为 [B, N, T, D]
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
        # o1_real = F.relu(
        #     torch.einsum('bijd,dd->bijd', x.real, r) - \
        #     torch.einsum('bijd,dd->bijd', x.imag, i) + \
        #     rb
        # )
        # o1_imag = F.relu(
        #     torch.einsum('bijd,dd->bijd', x.imag, r) + \
        #     torch.einsum('bijd,dd->bijd', x.real, i) + \
        #     ib
        # )

        o1_real = F.relu(
            torch.einsum('bijd,dd->bijd', x.real, r) + \
 \
            rb
        )
        o1_imag = F.relu(
            torch.einsum('bijd,dd->bijd', x.imag, i) + \
            ib
        )

        y = torch.stack([o1_real, o1_imag], dim=-1)
        y = F.softshrink(y, lambd=self.sparsity_threshold)
        y = torch.view_as_complex(y)
        return y



    def forward(self, x, return_repr=False):
        #测试没有分解
        # x_season = x
        # x_trend = x
        #  我的分解
        x_season, x_trend = self.sd(x)
        # 获取输入形状
        B, T, N = x.shape
        # x: [Batch, Input length, Channel]
        # B, T, N = x.shape
        # # embedding x: [B, N, T, D]
        # x = self.tokenEmb(x)
        # bias = x
        # [B, N, T, D]
        # if self.channel_independence == '1':
        #     x = self.MLP_channel(x, B, N, T)
        # # [B, N, T, D]
        # x = self.MLP_temporal(x, B, N, T)
        # residual3 = self.residual_block(x.permute(0, 2, 1, 3))
        # x_season = x + residual3.permute(0, 2, 1, 3)
        # x = x + bias
        # x = self.fc(x.reshape(B, N, -1)).permute(0, 2, 1)
        # 对 x_season 和 x_trend 进行特征提取
        # 嵌入特征：[B, N, T, D]
        x_season = self.tokenEmb(x_season)
        x_trend = self.tokenEmb(x_trend)  # 128 400 25 128
        # 记录偏置
        bias_season = x_season
        bias_trend = x_trend
        # 分别处理 x_season 和 x_trend
        # 通道独立处理
        if self.channel_independence == '1':
            x_season = self.MLP_channel(x_season, B, N, T)  #128 400 25 128
            x_trend = self.MLP_channel(x_trend, B, N, T)
            x_season = x_season + bias_season
            x_trend = x_trend + bias_trend
        # #时间处理
        # x_season = self.MLP_temporal(x_season, B, N, T)  #128 400 25 128
        # x_trend = self.MLP_temporal(x_trend, B, N, T)

        # 将处理后的结果相加
        x_season = x_season + bias_season
        x_trend = x_trend + bias_trend

        #
        # x_season = self.EMA(x_season)  #2024 400 25 128
        # x_trend = self.EMA(x_trend)

        #将分解后的部分合并为最终的 x，使用加权平均
        x = x_season * self.alpha + x_trend * (1 - self.alpha)#N T D

        if return_repr:
            return x  # ✅ 返回 repr, shape: [B, N, T, D]

        # 输出形状调整
        x=x.reshape(B, N, -1)# B N  (T*D)
        x = self.fc(x)#B T D
        x = x.permute(0, 2, 1)  # (T*D)xN
        # x = self.fc(x.reshape(B, N, -1)).permute(0, 2, 1)
        # x = self.fc(x.reshape(B, N, -1)).permute(0, 2, 1)
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
        x1 = self.act(self.conv1(x))#200 20 20-->200 20 20
        x2 = self.act(self.conv2(nn.functional.avg_pool2d(x, kernel_size=2)))  # 下采样
        x2 = nn.functional.interpolate(x2, size=x.size()[2:], mode='bilinear', align_corners=False)  # 调整尺寸
        x3 = self.act(self.conv3(nn.functional.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)))  # 上采样
        x3 = nn.functional.avg_pool2d(x3, kernel_size=2)  # 调整尺寸
        # print(f"x shape: {x.shape}, x1 shape: {x1.shape}, x2 shape: {x2.shape}, x3 shape: {x3.shape}")
        # 融合不同尺度的特征
        return x + x1 + x2 + x3


# class ResDWC1(nn.Module):
#     def __init__(self, dim, kernel_size=3):
#         super().__init__()
#         self.dim = dim
#         self.kernel_size = kernel_size
#         #
#         self.act = nn.ReLU(inplace=True)
#         #
#         self.conv = nn.Conv2d(dim, dim, kernel_size, 1, kernel_size // 2, groups=dim,bias=True)
#     def forward(self, x):
#         return x + self.act(self.conv(x))
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
        x = self.fc1(x)# T 20 20-->200 20 20
        x = self.act1(x)
        x = self.drop(x)
        # # 通过 ResDWC 模块
        x = self.conv(x)
        # 第二层卷积 + Dropout
        x = self.fc2(x)
        x = self.drop(x)
        return x


class STMLP(nn.Module):
    def __init__(self, input_shape, meta_shape, cross_shape, alpha, num_init_features=12, growth_rate=12, bn_size=4,
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
        self.alpha = alpha
        self.h, self.w = self.input_shape[-2], self.input_shape[-1]
        self.inner_shape = self.input_shape[:2] + (self.filters,) + self.input_shape[-2:]
        self.seq_len = self.input_shape[1]
        self.d_model = d_model
        self.model4 = FreTS(pred_len=pred_len, enc_in=enc_in, seq_len=pred_len,
                            channel_independence=channel_independence, alpha=alpha)

        # self.segRNN = SegRNN(seq_len=25, enc_in=400, d_model=256, dropout=0.1, task_name='short_term_forecast',
        #                      pred_len=25,
        #                      seg_len=5)
        self.CONglu = Mlp(pred_len, 200)

    def forward(self, out, meta=None, cross=None, return_repr=False):
        batch_size, seq_len, features, row_of_cells, col_of_cells = out.shape
        #   (b,s,f,20,20)
        out = out.squeeze(2)
        out = out.view(batch_size, seq_len, features, row_of_cells * col_of_cells).transpose(2, 3).squeeze(
            -1)  #       # (32, 25, 400)
        out = self.model4(out)
        if return_repr:
            return self.model4(out, return_repr=True)  # 返回嵌入表示 [B, N, T, D]

        out = out.view(batch_size, -1, row_of_cells, col_of_cells)
        # out = self.CONglu(out)
        return out


if __name__ == '__main__':
    device = torch.device('cpu')
    input = torch.randn(128, 25, 1, 20, 20).to(device)
    mete = torch.randn(32, 4).to(device)  #小时的特征
    cros = torch.randn(32, 4, 20, 20)  #其他特征
    net = STMLP(input.shape, mete.shape, cros.shape,0.8).to(device)
    out = net(input)
    print(out.shape)
