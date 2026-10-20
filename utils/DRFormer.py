import torch
from torch import nn
from torch import Tensor
from layers.Transformer_EncDec import Encoder, EncoderLayer
from layers.SelfAttention_Family import FullAttention
from layers.ROPEAttentionLayer import ROPEAttentionGroupSequenceIngroupLayer
from layers.Embed import PatchEmbedding, Patching
from torch.nn.parameter import Parameter
import torch.nn.functional as F
import torch.nn.init as init
import math


class FlattenHead(nn.Module):
    def __init__(self, n_vars, nf, target_window, head_dropout=0):
        super().__init__()
        self.n_vars = n_vars
        self.flatten = nn.Flatten(start_dim=-2)
        self.linear = nn.Linear(nf, target_window)
        self.dropout = nn.Dropout(head_dropout)

    def forward(self, x):  # x: [bs x nvars x d_model x patch_num]
        x = self.flatten(x)
        x = self.linear(x)
        x = self.dropout(x)
        return x


class FixMaskLinear(nn.Module):
    __constants__ = ['in_features', 'out_features']
    in_features: int
    out_features: int
    weight: Tensor
    group: int

    def __init__(self, group: int, in_features: int, out_features: int, bias: bool = True,
                 device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.empty((out_features, in_features), **factory_kwargs))

        if bias:
            self.bias = Parameter(torch.empty(out_features, **factory_kwargs))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            init.uniform_(self.bias, -bound, bound)

    def forward(self, input: Tensor) -> Tensor:
        return F.linear(input, self.weight, self.bias)

    def extra_repr(self) -> str:
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_features, self.out_features, self.bias is not None
        )


class Model(nn.Module):
    """
    Paper link: https://arxiv.org/pdf/2211.14730.pdf
    """

    def __init__(self, seq_len, pred_len,sequence_num,enc_in,dropout,d_model,patch_len=48, stride=2,e_layers=2,d_ff=2048,n_heads=8,factor=1,activation='relu',output_attention=False):
        """
        patch_len: int, patch len for patch_embedding
        stride: int, stride for patch_embedding
        """
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.patch_len = patch_len
        self. stride = stride
        self.d_model = d_model
        padding = stride
        self.sequence_num =sequence_num
        self.enc_in = enc_in
        self.dropout = dropout
        self.e_layers= e_layers
        # patching and embedding
        self.patching = Patching(d_model, patch_len, stride, padding, dropout)
        self.masked_linear = FixMaskLinear(group=8, in_features=patch_len, out_features=d_model)

        self.patch_num = int((seq_len - patch_len) / stride + 2)

        self.stride_list = [2 ** i for i in range(1, sequence_num)]
        self.total_len = int((seq_len - patch_len) / stride + 2)
        group_pos = [[0] * self.total_len]

        for i in range(0, sequence_num - 1):
            self.total_len = self.total_len + math.ceil(self.patch_num / self.stride_list[i])
            group_pos.append([i + 1] * math.ceil(self.patch_num / self.stride_list[i]))

        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    ROPEAttentionGroupSequenceIngroupLayer(
                        FullAttention(False, factor, attention_dropout=dropout,
                                      output_attention=output_attention), d_model, n_heads,
                        group_pos),
                    d_model,
                    d_ff,
                    dropout=dropout,
                    activation=activation
                ) for l in range(e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(d_model)
        )

        # Prediction Head
        self.head_nf = self.d_model * int((seq_len - patch_len) / stride + 2)
        self.head = FlattenHead(self.enc_in, self.head_nf, self.pred_len,
                                head_dropout=self.dropout)
        padding_list = []
        for i in range(sequence_num - 1):
            if self.patch_num % self.stride_list[i] == 0:
                padding_len = 0
                padding_list.append(padding_len)
                continue

            padding_len = math.ceil((self.stride_list[i] - self.patch_num % self.stride_list[i]) / 2)
            padding_list.append(padding_len)

        self.max_pooling_list = nn.ModuleList(
            [nn.MaxPool1d(self.stride_list[i], stride=self.stride_list[i], padding=padding_list[i]) for i in
             range(sequence_num - 1)])
        self.sep = Parameter(torch.randn(1, d_model))
        self.conv_transpose_list = nn.ModuleList([nn.ConvTranspose1d(in_channels=self.d_model,
                                                                     out_channels=self.d_model,
                                                                     stride=self.stride_list[i],
                                                                     kernel_size=self.stride_list[i],
                                                                     padding=padding_list[i], output_padding=0,
                                                                     dilation=1, padding_mode="zeros", bias=False) for i
                                                  in range(sequence_num - 1)])

    def multi_scale(self, x):
        # z: [bs * nvars x d_model x patch_num]
        x_permute = x.permute(0, 2, 1)
        # z: [bs * nvars x sequence_len x d_model]
        sequence_list = [self.max_pooling_list[i](x_permute).permute(0, 2, 1) for i in range(self.sequence_num - 1)]
        self.sequence_sep_len = [x.shape[1]]
        self.sequence_sep_len.extend([sequence_list[i].shape[1] for i in range(self.sequence_num - 1)])
        x_embedding = torch.cat((x, *sequence_list), dim=1)

        return x_embedding

    def conv_transpose(self, enc_out):
        enc_sequence_list = []
        index = 0
        for i in range(len(self.sequence_sep_len)):
            end = index + self.sequence_sep_len[i]
            enc_sequence_list.append(enc_out[:, index:end, :])
            index = end
        sequence = enc_sequence_list[0]
        for i in range(1, len(enc_sequence_list)):
            conv = self.conv_transpose_list[i - 1](enc_sequence_list[i].permute(0, 2, 1)).permute(0, 2, 1)
            sequence = sequence + conv
        return sequence

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        # x: [Batch, Input length, Channel]
        # Normalization from Non-stationary Transformer
        print(x_enc.shape)

        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(
            torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc /= stdev

        # do patching and embedding
        x_enc = x_enc.permute(0, 2, 1)
        # u: [bs * nvars x patch_num x d_model]
        enc_out, n_vars = self.patching(x_enc)
        print(enc_out.shape)

        enc_out = self.masked_linear(enc_out)

        enc_out = self.multi_scale(enc_out)

        # Encoder
        # z: [bs * nvars x patch_num x d_model]
        enc_out, attns = self.encoder(enc_out)

        enc_out = self.conv_transpose(enc_out)
        print(f"Shape of enc_out before reshape: {enc_out.shape}")

        # z: [bs x nvars x patch_num x d_model]
        enc_out = torch.reshape(
            enc_out, (-1, n_vars, enc_out.shape[-2], enc_out.shape[-1]))

        # z: [bs x nvars x d_model x patch_num]
        enc_out = enc_out.permute(0, 1, 3, 2)

        # Decoder
        dec_out = self.head(enc_out)  # z: [bs x nvars x target_window]
        dec_out = dec_out.permute(0, 2, 1)

        # De-Normalization from Non-stationary Transformer
        dec_out = dec_out * \
                  (stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
        dec_out = dec_out + \
                  (means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
        return dec_out[:, -self.pred_len:, :]  # [B, L, D]
if __name__ == "__main__":
    # Define model hyperparameters
    seq_len = 25 # Sequence length
    pred_len =25 # Prediction length
    d_model =128  # Embedding dimension
    patch_len =16 # Patch length
    stride = 2  # Stride for patching
    sequence_num = 1  # Number of sequences
    enc_in = 400  # Number of input features
    dropout = 0.01  # Dropout rate
    e_layers = 1  # Number of encoder layers
    d_ff = 2048  # Feed-forward network dimension
    n_heads = 8  # Number of attention heads
    factor = 3  # Attention factor
    activation = 'relu'  # Activation function
    output_attention = False  # Whether to output attention weights

    # Initialize the model with the defined hyperparameters
    model = Model(
        seq_len=seq_len,
        pred_len=pred_len,
        d_model=d_model,
        patch_len=patch_len,
        stride=stride,
        sequence_num=sequence_num,
        enc_in=enc_in,
        dropout=dropout,
        e_layers=e_layers,
        d_ff=d_ff,
        n_heads=n_heads,
        factor=factor,
        activation=activation,
        output_attention=output_attention
    )

    # Generate sample input data
    batch_size = 32
    num_vars = 400 # For simplicity, using 1 variable
    x_enc = torch.randn(batch_size,seq_len,  num_vars)
    x_mark_enc = torch.randn(batch_size, num_vars, seq_len)  # Placeholder for time features
    x_dec = torch.randn(batch_size, num_vars, pred_len)
    x_mark_dec = torch.randn(batch_size, num_vars, pred_len)  # Placeholder for time features

    # Perform a forward pass
    output = model(x_enc, x_mark_enc, x_dec, x_mark_dec)

    # Print the output shape
    print(f"Output shape: {output.shape}")
