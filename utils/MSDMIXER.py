#  -*- coding: UTF-8 -*
import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np
import torch

adaptive_filter = True


from einops import rearrange, reduce
from einops.layers.torch import Rearrange
from timm.models.layers import DropPath

def get_activation(activ: str):
    if activ == "gelu":
        return nn.GELU()
    elif activ == "sigmoid":
        return nn.Sigmoid()
    elif activ == "tanh":
        return nn.Tanh()
    elif activ == "relu":
        return nn.ReLU()

    raise RuntimeError("activation should not be {}".format(activ))

class MLPBlock(nn.Module):

    def __init__(
        self,
        dim,
        in_features: int,
        hid_features: int,
        out_features: int,
        activ="gelu",
        drop: float = 0.0,
        jump_conn='trunc',
    ):
        super().__init__()
        self.dim = dim
        self.out_features = out_features
        self.net = nn.Sequential(
            nn.Linear(in_features, hid_features),
            get_activation(activ),
            nn.Linear(hid_features, out_features),
            DropPath(drop))
        if jump_conn == "trunc":
            self.jump_net = nn.Identity()
        elif jump_conn == 'proj':
            self.jump_net = nn.Linear(in_features, out_features)
        else:
            raise ValueError(f"jump_conn:{jump_conn}")

    def forward(self, x):
        x = torch.transpose(x, self.dim, -1)
        x = self.jump_net(x)[..., :self.out_features] + self.net(x)
        x = torch.transpose(x, self.dim, -1)
        return x


class PatchEncoder(nn.Module):

    def __init__(
        self,
        in_len: int,
        hid_len: int,
        in_chn: int,
        hid_chn: int,
        out_chn,
        patch_size: int,
        hid_pch: int,
        norm=None,
        activ="gelu",
        drop: float = 0.0,
    ) -> None:
        super().__init__()
        self.net = nn.Sequential()
        channel_wise_mlp = MLPBlock(1, in_chn, hid_chn, out_chn, activ, drop)
        inter_patch_mlp = MLPBlock(2, in_len // patch_size, hid_len, in_len // patch_size, activ,
                         drop)

        if norm == 'bn':
            norm_class = nn.BatchNorm2d
        elif norm == 'in':
            norm_class = nn.InstanceNorm2d
        else:
            norm_class = nn.Identity
        linear = nn.Linear(patch_size, 1)
        intra_patch_mlp = MLPBlock(3, patch_size, hid_pch, patch_size, activ, drop)
        self.net.append(Rearrange("b c (l1 l2) -> b c l1 l2", l2=patch_size))
        self.net.append(norm_class(in_chn))
        self.net.append(channel_wise_mlp)
        self.net.append(norm_class(out_chn))
        self.net.append(inter_patch_mlp)
        self.net.append(norm_class(out_chn))
        self.net.append(intra_patch_mlp)
        self.net.append(linear)
        self.net.append(Rearrange("b c l1 1 -> b c l1"))

    def forward(self, x):
        # b,c,l
        return self.net(x)



class PatchDecoder(nn.Module):

    def __init__(
        self,
        in_len: int,
        hid_len: int,
        in_chn: int,
        hid_chn: int,
        out_chn,
        patch_size: int,
        hid_pch: int,
        norm=None,
        activ="gelu",
        drop: float = 0.0,
    ) -> None:
        super().__init__()
        self.net = nn.Sequential()
        inter_patch_mlp = MLPBlock(2, in_len // patch_size, hid_len, in_len // patch_size, activ,
                         drop)
        channel_wise_mlp = MLPBlock(1, in_chn, hid_chn, out_chn, activ, drop)
        if norm == 'bn':
            norm_class = nn.BatchNorm2d
        elif norm == 'in':
            norm_class = nn.InstanceNorm2d
        else:
            norm_class = nn.Identity
        linear = nn.Linear(1, patch_size)
        intra_patch_mlp = MLPBlock(3, patch_size, hid_pch, patch_size, activ, drop)
        self.net.append(Rearrange("b c l1 -> b c l1 1"))
        self.net.append(linear)
        self.net.append(norm_class(in_chn))
        self.net.append(intra_patch_mlp)
        self.net.append(norm_class(in_chn))
        self.net.append(inter_patch_mlp)
        self.net.append(norm_class(in_chn))
        self.net.append(channel_wise_mlp)
        self.net.append(Rearrange("b c l1 l2 -> b c (l1 l2)"))

    def forward(self, x):
        # b,c,l
        return self.net(x)


class PredictionHead(nn.Module):

    def __init__(self,
                 in_len,
                 out_len,
                 hid_len,
                 in_chn,
                 out_chn,
                 hid_chn,
                 activ,
                 drop=0.0) -> None:
        super().__init__()
        self.net = nn.Sequential()
        if in_chn != out_chn:
            c_jump_conn = "proj"
        else:
            c_jump_conn = "trunc"
        self.net.append(
            MLPBlock(1,
                in_chn,
                hid_chn,
                out_chn,
                activ=activ,
                drop=drop,
                jump_conn=c_jump_conn))
        self.net.append(
            MLPBlock(2,
                in_len,
                hid_len,
                out_len,
                activ=activ,
                drop=drop,
                jump_conn='proj'))

    def forward(self, x):
        return self.net(x)


class MSDMixer(nn.Module):

    def __init__(self,
                 in_len,
                 out_len,
                 in_chn,
                 ex_chn,
                 out_chn,
                 patch_sizes,
                 hid_len,
                 hid_chn,
                 hid_pch,
                 hid_pred,
                 norm,
                 last_norm,
                 activ,
                 drop,
                 reduction="sum") -> None:
        super().__init__()
        self.in_len = in_len
        self.out_len = out_len
        self.in_chn = in_chn
        self.out_chn = out_chn
        self.last_norm = last_norm
        self.reduction = reduction
        self.patch_encoders = nn.ModuleList()
        self.patch_decoders = nn.ModuleList()
        self.pred_heads = nn.ModuleList()
        self.patch_sizes = patch_sizes
        self.paddings = []
        all_chn = in_chn + ex_chn
        for i, patch_size in enumerate(patch_sizes):
            res = in_len % patch_size
            padding = (patch_size - res) % patch_size
            self.paddings.append(padding)
            padded_len = in_len + padding
            self.patch_encoders.append(
                PatchEncoder(padded_len, hid_len, all_chn, hid_chn,
                          in_chn, patch_size, hid_pch, norm, activ, drop))
            self.patch_decoders.append(
                PatchDecoder(padded_len, hid_len, in_chn, hid_chn, in_chn,
                        patch_size, hid_pch, norm, activ, drop))
            if out_len != 0 and out_chn != 0:
                self.pred_heads.append(
                    PredictionHead(padded_len // patch_size, out_len, hid_pred,
                                    in_chn, out_chn, hid_chn, activ, drop))
            else:
                self.pred_heads.append(nn.Identity())

    def forward(self, x, x_mark=None, x_mask=None):
        x = rearrange(x, "b l c -> b c l")
        if x_mark is not None:
            x_mark = rearrange(x_mark, "b l c -> b c l")
        if x_mask is not None:
            x_mask = rearrange(x_mask, "b l c -> b c l")
        if self.last_norm:
            x_last = x[:, :, [-1]].detach()
            x = x - x_last
            if x_mark is not None:
                x_mark_last = x_mark[:, :, [-1]].detach()
                x_mark = x_mark - x_mark_last
        y_pred = []
        for i in range(len(self.patch_sizes)):
            x_in = x
            if x_mark is not None:
                x_in = torch.cat((x, x_mark), 1)
            x_in = F.pad(x_in, (self.paddings[i], 0), "constant", 0)
            emb = self.patch_encoders[i](x_in)
            comp = self.patch_decoders[i](emb)[:, :, self.paddings[i]:]
            pred = self.pred_heads[i](emb)
            if x_mask is not None:
                comp = comp * x_mask
            x = x - comp
            if self.out_len != 0 and self.out_chn != 0:
                y_pred.append(pred)

        if self.out_len != 0 and self.out_chn != 0:
            y_pred = reduce(torch.stack(y_pred, 0), "h b c l -> b c l",
                            self.reduction)
            if self.last_norm and self.out_chn == self.in_chn:
                y_pred += x_last
            y_pred = rearrange(y_pred, "b c l -> b l c")
            return y_pred, x
        else:
            return None, x


import torch

class  MIX(nn.Module):
    def __init__(self, input_shape, meta_shape, cross_shape, num_init_features=12, growth_rate=12, bn_size=4,
                 drop_rate=0.1, nb_flows=25, fusion=1, d_model=25, pred_len=25, enc_in=400, seq_len=25,
                 channel_independence='1',
                 flags_meta=0, flags_cross=0):
        super().__init__()
        self.input_shape = input_shape
        self.meta_shape = meta_shape
        self.cross_shape = cross_shape
        self.filters = num_init_features
        self.channels = nb_flows
        self.fusion = fusion
        self.h, self.w = self.input_shape[-2], self.input_shape[-1]
        self.inner_shape = self.input_shape[:2] + (self.filters,) + self.input_shape[-2:]
        self.seq_len = self.input_shape[1]
        self.d_model = d_model
        self.model4 = MSDMixer(out_len=pred_len, in_len=pred_len,in_chn=400,out_chn=400,patch_sizes=[12, 6, 6, 1, 1],ex_chn=0,
                               hid_len=64,hid_chn=32,hid_pch=32,hid_pred=64,norm='bn',last_norm=True,activ="gelu",drop= 0.01,reduction="sum")


    def forward(self, out, meta=None, cross=None):
        batch_size, seq_len, features, row_of_cells, col_of_cells = out.shape
        #   (b,s,f,20,20)
        out = out.squeeze(2)
        out = out.view(batch_size, seq_len, features, row_of_cells * col_of_cells).transpose(2, 3).squeeze(
            -1)  #       # (32, 25, 400)
        out,res= self.model4(out)
        out = out.view(batch_size, -1, row_of_cells, col_of_cells)
        return out,res


if __name__ == '__main__':
    device = torch.device('cpu')
    input = torch.randn(128, 25, 1, 20, 20).to(device)
    mete = torch.randn(32, 4).to(device)  #小时的特征
    cros = torch.randn(32, 4, 20, 20)  #其他特征
    net = MIX(input.shape, mete.shape, cros.shape).to(device)
    out,res = net(input)
    print(out.shape)
    print(res.shape)

