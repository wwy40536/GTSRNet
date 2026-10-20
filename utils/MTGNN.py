from utils. layer import *





class gtnet(nn.Module):
    def __init__(self, gcn_true, buildA_true, gcn_depth, num_nodes, device, predefined_A=None, static_feat=None, dropout=0.3, subgraph_size=20, node_dim=40, dilation_exponential=1, conv_channels=32, residual_channels=32, skip_channels=64, end_channels=128, seq_length=12, in_dim=2, out_dim=12, layers=3, propalpha=0.05, tanhalpha=3, layer_norm_affline=True):
        super(gtnet, self).__init__()
        self.gcn_true = gcn_true
        self.buildA_true = buildA_true
        self.num_nodes = num_nodes
        self.dropout = dropout
        self.predefined_A = predefined_A
        self.filter_convs = nn.ModuleList()
        self.gate_convs = nn.ModuleList()
        self.residual_convs = nn.ModuleList()
        self.skip_convs = nn.ModuleList()
        self.gconv1 = nn.ModuleList()
        self.gconv2 = nn.ModuleList()
        self.norm = nn.ModuleList()
        self.start_conv = nn.Conv2d(in_channels=in_dim,
                                    out_channels=residual_channels,
                                    kernel_size=(1, 1))
        self.gc = graph_constructor(num_nodes, subgraph_size, node_dim, device, alpha=tanhalpha, static_feat=static_feat)

        self.seq_length = seq_length
        kernel_size = 7
        if dilation_exponential>1:
            self.receptive_field = int(1+(kernel_size-1)*(dilation_exponential**layers-1)/(dilation_exponential-1))
        else:
            self.receptive_field = layers*(kernel_size-1) + 1

        for i in range(1):
            if dilation_exponential>1:
                rf_size_i = int(1 + i*(kernel_size-1)*(dilation_exponential**layers-1)/(dilation_exponential-1))
            else:
                rf_size_i = i*layers*(kernel_size-1)+1
            new_dilation = 1
            for j in range(1,layers+1):
                if dilation_exponential > 1:
                    rf_size_j = int(rf_size_i + (kernel_size-1)*(dilation_exponential**j-1)/(dilation_exponential-1))
                else:
                    rf_size_j = rf_size_i+j*(kernel_size-1)

                self.filter_convs.append(dilated_inception(residual_channels, conv_channels, dilation_factor=new_dilation))
                self.gate_convs.append(dilated_inception(residual_channels, conv_channels, dilation_factor=new_dilation))
                self.residual_convs.append(nn.Conv2d(in_channels=conv_channels,
                                                    out_channels=residual_channels,
                                                 kernel_size=(1, 1)))
                if self.seq_length>self.receptive_field:
                    self.skip_convs.append(nn.Conv2d(in_channels=conv_channels,
                                                    out_channels=skip_channels,
                                                    kernel_size=(1, self.seq_length-rf_size_j+1)))
                else:
                    self.skip_convs.append(nn.Conv2d(in_channels=conv_channels,
                                                    out_channels=skip_channels,
                                                    kernel_size=(1, self.receptive_field-rf_size_j+1)))

                if self.gcn_true:
                    self.gconv1.append(mixprop(conv_channels, residual_channels, gcn_depth, dropout, propalpha))
                    self.gconv2.append(mixprop(conv_channels, residual_channels, gcn_depth, dropout, propalpha))

                if self.seq_length>self.receptive_field:
                    self.norm.append(LayerNorm((residual_channels, num_nodes, self.seq_length - rf_size_j + 1),elementwise_affine=layer_norm_affline))
                else:
                    self.norm.append(LayerNorm((residual_channels, num_nodes, self.receptive_field - rf_size_j + 1),elementwise_affine=layer_norm_affline))

                new_dilation *= dilation_exponential

        self.layers = layers
        self.end_conv_1 = nn.Conv2d(in_channels=skip_channels,
                                             out_channels=end_channels,
                                             kernel_size=(1,1),
                                             bias=True)
        self.end_conv_2 = nn.Conv2d(in_channels=end_channels,
                                             out_channels=out_dim,
                                             kernel_size=(1,1),
                                             bias=True)
        if self.seq_length > self.receptive_field:
            self.skip0 = nn.Conv2d(in_channels=in_dim, out_channels=skip_channels, kernel_size=(1, self.seq_length), bias=True)
            self.skipE = nn.Conv2d(in_channels=residual_channels, out_channels=skip_channels, kernel_size=(1, self.seq_length-self.receptive_field+1), bias=True)

        else:
            self.skip0 = nn.Conv2d(in_channels=in_dim, out_channels=skip_channels, kernel_size=(1, self.receptive_field), bias=True)
            self.skipE = nn.Conv2d(in_channels=residual_channels, out_channels=skip_channels, kernel_size=(1, 1), bias=True)


        self.idx = torch.arange(self.num_nodes).to(device)


    def forward(self, input, idx=None):
        seq_len = input.size(3)
        assert seq_len==self.seq_length, 'input sequence length not equal to preset sequence length'

        if self.seq_length<self.receptive_field:
            input = nn.functional.pad(input,(self.receptive_field-self.seq_length,0,0,0))



        if self.gcn_true:
            if self.buildA_true:
                if idx is None:
                    adp = self.gc(self.idx)
                else:
                    adp = self.gc(idx)
            else:
                adp = self.predefined_A

        x = self.start_conv(input)
        skip = self.skip0(F.dropout(input, self.dropout, training=self.training))
        for i in range(self.layers):
            residual = x
            filter = self.filter_convs[i](x)
            filter = torch.tanh(filter)
            gate = self.gate_convs[i](x)
            gate = torch.sigmoid(gate)
            x = filter * gate
            x = F.dropout(x, self.dropout, training=self.training)
            s = x
            s = self.skip_convs[i](s)
            skip = s + skip
            if self.gcn_true:
                x = self.gconv1[i](x, adp)+self.gconv2[i](x, adp.transpose(1,0))
            else:
                x = self.residual_convs[i](x)

            x = x + residual[:, :, :, -x.size(3):]
            if idx is None:
                x = self.norm[i](x,self.idx)
            else:
                x = self.norm[i](x,idx)

        skip = self.skipE(x) + skip
        x = F.relu(skip)
        x = F.relu(self.end_conv_1(x))
        x = self.end_conv_2(x)
        return x


class Mtgnn(nn.Module):
        def __init__(self, input_shape, meta_shape, cross_shape, device,num_init_features=12, growth_rate=12, bn_size=4,
                     e_layers=2, d_ff=2048, n_heads=8, factor=1, activation='relu', output_attention=False,
                     drop_rate=0.01, nb_flows=25, fusion=1, maps=1, pred_len=25, enc_in=400, seq_len=25,
                     patch_len=8, stride=2, sequence_num=3,
                     ):
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
            self.model4 = gtnet(
                gcn_true=True,
                buildA_true=True,
                gcn_depth=2,
                num_nodes=400,  # 对应输入中的D=400
                device=device,
                in_dim=1,  # 输入特征维度（调整为1）
                out_dim=2,  # 预测25个时间步
                seq_length=24,  # 输入序列长度
                # 其余参数保持默认或按需设置
                conv_channels=8,
                residual_channels=16,
                skip_channels=32,
                end_channels=64,
                layers=3,
                dilation_exponential=2
            )

        def forward(self, out, meta=None, cross=None):
            batch_size, seq_len, features, row_of_cells, col_of_cells = out.shape
            #   (b,s,f,20,20)
            out = out.squeeze(2)
            out = out.view(batch_size, seq_len, features, row_of_cells * col_of_cells).transpose(2, 3).squeeze(
                -1)  # # (32, 25, 400)
            out = out.permute(0, 2, 1).unsqueeze(1)  # 形状变为 (32, 1, 400, 25)
            out = self.model4(out)
            out = out.squeeze(-1)  # 形状变为 (32, 25, 400)
            out = out.view(batch_size, -1, row_of_cells, col_of_cells)
            return out
if __name__ == '__main__':
    import torch

    # 假设输入数据形状为 (32, 25, 400)
    input_data = torch.randn(32, 24, 400)  # B, L, D

    # 调整输入维度为 (B, in_dim, num_nodes, L)
    input_reshaped = input_data.permute(0, 2, 1).unsqueeze(1)  # 形状变为 (32, 1, 400, 25)

    # 创建模型实例（关键参数配置）
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = gtnet(
        gcn_true=True,
        buildA_true=True,
        gcn_depth=2,
        num_nodes=400,  # 对应输入中的D=400
        device=device,
        in_dim=1,  # 输入特征维度（调整为1）
        out_dim=24,  # 预测25个时间步
        seq_length=24,  # 输入序列长度
        # 其余参数保持默认或按需设置
        conv_channels=32,
        residual_channels=32,
        skip_channels=64,
        end_channels=128,
        layers=3,
        dilation_exponential=1
    ).to(device)

    # 将数据移至相同设备
    input_reshaped = input_reshaped.to(device)

    # 前向传播
    model.eval()  # 设置为评估模式
    with torch.no_grad():
        output = model(input_reshaped)  # 输出形状 (32, 25, 400, 1)

    # 调整输出维度，移除末尾的维度1
    output = output.squeeze(-1)  # 形状变为 (32, 25, 400)

    print("预测结果形状:", output.shape)  # 应为 (32, 25, 400)