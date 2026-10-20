# utils/densenet_convlstm_cross_ms.py
# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict


class ConvLSTMCell(nn.Module):
    """Convolutional LSTM cell"""

    def __init__(self, input_size, input_dim, hidden_dim, kernel_size):
        super().__init__()
        self.height, self.width = input_size
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.padding = kernel_size // 2

        self.Gates = nn.Conv2d(
            in_channels=self.input_dim + self.hidden_dim,
            out_channels=4 * self.hidden_dim,
            kernel_size=self.kernel_size,
            padding=self.padding,
            bias=True
        )

    def forward(self, input_tensor, cur_state):
        # input_tensor: (B, Cin, H, W)
        h_cur, c_cur = cur_state
        # 保证 hidden/cell 在同一个 device 上（不要用全局 device）
        h_cur = h_cur.to(input_tensor.device)
        c_cur = c_cur.to(input_tensor.device)

        combined = torch.cat([input_tensor, h_cur], dim=1)
        gates = self.Gates(combined)

        in_gate, remember_gate, out_gate, cell_gate = gates.chunk(4, 1)

        in_gate = self.hard_sigmoid(in_gate)
        remember_gate = self.hard_sigmoid(remember_gate)
        out_gate = self.hard_sigmoid(out_gate)
        cell_gate = torch.tanh(cell_gate)

        c_next = (remember_gate * c_cur) + (in_gate * cell_gate)
        h_next = out_gate * torch.tanh(c_next)
        return h_next, c_next

    def init_hidden(self, batch_size, device):
        return (
            torch.zeros(batch_size, self.hidden_dim, self.height, self.width, device=device),
            torch.zeros(batch_size, self.hidden_dim, self.height, self.width, device=device)
        )

    @staticmethod
    def hard_sigmoid(x):
        x = (0.2 * x) + 0.5
        x = F.threshold(-x, -1, -1)
        x = F.threshold(-x, 0, 0)
        return x


class ConvLSTMLayer(nn.Module):
    def __init__(self, filters, kernel_size, input_shape, return_sequences=False):
        super().__init__()
        # input_shape: (B, T, C, H, W)
        self.filters = filters
        self.kernel_size = kernel_size
        self.input_shape = input_shape
        self.height = input_shape[3]
        self.width = input_shape[4]
        self.channels = input_shape[2]
        self.sequences = input_shape[1]
        self.return_sequences = return_sequences

        self.CLCell = ConvLSTMCell(
            input_size=(self.height, self.width),
            input_dim=self.channels,
            hidden_dim=self.filters,
            kernel_size=self.kernel_size
        )

    def forward(self, x, hidden_state=None):
        # x: (B, T, C, H, W)
        B, T, C, H, W = x.shape
        device = x.device

        if hidden_state is None:
            h, c = self.CLCell.init_hidden(B, device=device)
        else:
            h, c = hidden_state
            h = h.to(device)
            c = c.to(device)

        outputs = []
        for t in range(T):
            h, c = self.CLCell(x[:, t], cur_state=(h, c))
            outputs.append(h)

        layer_output = torch.stack(outputs, dim=1)  # (B, T, filters, H, W)
        if self.return_sequences:
            return layer_output
        else:
            return layer_output[:, -1]  # (B, filters, H, W)


class _DenseLayer(nn.Sequential):
    def __init__(self, num_input_features, growth_rate, bn_size, drop_rate):
        super().__init__()
        self.add_module('norm1', nn.BatchNorm2d(num_input_features))
        self.add_module('relu1', nn.ReLU(inplace=True))
        self.add_module('conv1', nn.Conv2d(num_input_features, bn_size * growth_rate,
                                           kernel_size=1, stride=1, bias=False))
        self.add_module('norm2', nn.BatchNorm2d(bn_size * growth_rate))
        self.add_module('relu2', nn.ReLU(inplace=True))
        self.add_module('conv2', nn.Conv2d(bn_size * growth_rate, growth_rate,
                                           kernel_size=3, stride=1, padding=1, bias=False))
        self.drop_rate = drop_rate

    def forward(self, x):
        new_features = super().forward(x.contiguous())
        if self.drop_rate > 0:
            new_features = F.dropout(new_features, p=self.drop_rate, training=self.training)
        return torch.cat([x, new_features], 1)


class _Transition(nn.Sequential):
    def __init__(self, num_input_features, num_output_features):
        super().__init__()
        self.add_module('norm', nn.BatchNorm2d(num_input_features))
        self.add_module('relu', nn.ReLU(inplace=True))
        self.add_module('conv', nn.Conv2d(num_input_features, num_output_features,
                                          kernel_size=1, stride=1, bias=False))


class _DenseBlock(nn.Sequential):
    def __init__(self, num_layers, num_input_features, bn_size, growth_rate, drop_rate):
        super().__init__()
        for i in range(num_layers):
            layer = _DenseLayer(num_input_features + i * growth_rate,
                                growth_rate, bn_size, drop_rate)
            self.add_module(f'denselayer{i+1}', layer)


class DenseNetConvLSTM_Cross_MultiStep(nn.Module):
    """
    兼容你现在的数据和训练逻辑：
      x:     (B, Tin, 1, H, W)
      cross: (B, 4, H, W)  # 原样输入，不额外处理
    输出：
      yhat:  (B, Tout, 1, H, W)
    """
    def __init__(
        self,
        input_shape,
        meta_shape=None,
        cross_shape=None,
        tin=25,
        tout=8,
        growth_rate=12,
        num_init_features=16,
        bn_size=4,
        drop_rate=0.2,
        nb_flows=1,
        maps=2,          # meta+cross+1；你现在 meta=0,cross=1 => maps=2
        cross_dim=4
    ):
        super().__init__()
        self.input_shape = input_shape
        self.meta_shape = meta_shape
        self.cross_shape = cross_shape

        self.tin = tin
        self.tout = tout
        self.filters = num_init_features
        self.nb_flows = nb_flows
        self.maps = maps

        self.h, self.w = input_shape[-2], input_shape[-1]

        # ConvLSTM temporal encoder
        inner_shape = (input_shape[0], input_shape[1], self.filters, self.h, self.w)
        self.temporal = nn.Sequential(OrderedDict([
            ('LSTM0', ConvLSTMLayer(self.filters, 3, input_shape=input_shape, return_sequences=True)),
            ('LSTM1', ConvLSTMLayer(self.filters, 3, input_shape=inner_shape, return_sequences=True)),
            ('LSTM2', ConvLSTMLayer(self.filters, 3, input_shape=inner_shape, return_sequences=False)),
        ]))

        # cross branch（不做额外预处理，直接进这套 conv）
        self.cross = nn.Sequential(OrderedDict([
            ('conv0', nn.Conv2d(cross_dim, self.filters, kernel_size=3, padding=1, bias=False)),
            ('bn0', nn.BatchNorm2d(self.filters)),
            ('relu0', nn.ReLU(inplace=True)),
            ('conv1', nn.Conv2d(self.filters, self.filters, kernel_size=3, padding=1, bias=False)),
        ]))

        # DenseNet trunk
        self.features = nn.Sequential()
        num_features = self.filters * maps  # maps=2 => temporal(filters) + cross(filters)

        block_config = [6, 6, 6]  # 原配置
        for i, num_layers in enumerate(block_config):
            block = _DenseBlock(num_layers=num_layers,
                                num_input_features=num_features,
                                bn_size=bn_size,
                                growth_rate=growth_rate,
                                drop_rate=drop_rate)
            self.features.add_module(f'denseblock{i+1}', block)
            num_features = num_features + num_layers * growth_rate
            if i != len(block_config) - 1:
                trans = _Transition(num_input_features=num_features,
                                    num_output_features=num_features // 2)
                self.features.add_module(f'transition{i+1}', trans)
                num_features = num_features // 2

        self.features.add_module('norm_last', nn.BatchNorm2d(num_features))
        self.features.add_module('relu_last', nn.ReLU(inplace=True))

        # 关键改动：输出 Tout * nb_flows 个通道，然后 reshape 成 (B,Tout,nb_flows,H,W)
        self.head = nn.Conv2d(num_features, tout * nb_flows, kernel_size=1, padding=0, bias=True)

        # init
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm3d)):
                m.weight.data.fill_(1.0)
                m.bias.data.zero_()

    def forward(self, x, meta=None, cross=None):
        # x: (B,Tin,1,H,W)
        out_t = self.temporal(x)  # (B, filters, H, W)

        if cross is None:
            raise ValueError("cross is required (B,4,H,W). You said you want to directly use cross.")

        out_c = self.cross(cross)  # (B, filters, H, W)

        # 拼接（maps=2 对应 temporal+cross）
        out = torch.cat([out_t, out_c], dim=1)  # (B, 2*filters, H, W)

        out = self.features(out)
        out = self.head(out)  # (B, Tout*nb_flows, H, W)

        B, _, H, W = out.shape
        out = out.view(B, self.tout, self.nb_flows, H, W)  # (B,Tout,1,H,W)

        return out
