import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicGCN(nn.Module):
    """
    最基础的图卷积层: Z = (A + I) * X * W
    """

    def __init__(self, in_channels, out_channels):
        super(BasicGCN, self).__init__()
        self.linear = nn.Linear(in_channels, out_channels)

    def forward(self, x, adj):
        """
        x: (B, N, C)
        adj: (N, N)
        """
        # 1. 图聚合: AX (B, N, C)
        # using einsum for matrix multiplication between (N,N) and (B,N,C)
        x_agg = torch.einsum('vw, bwc -> bvc', adj, x)

        # 2. 特征变换: W
        x_out = self.linear(x_agg)

        return F.relu(x_out)


class STBlock(nn.Module):
    """
    STGCN 核心模块: Temporal -> Spatial -> Temporal (三明治结构)
    """

    def __init__(self, channels, num_nodes, dropout=0.1):
        super(STBlock, self).__init__()

        # 1. Temporal Conv 1 (使用 1D 卷积模拟时序处理)
        # Input: (B, C, N, T)
        self.temp_conv1 = nn.Conv2d(channels, channels, kernel_size=(1, 3), padding=(0, 1))

        # 2. Spatial GCN
        self.gcn = BasicGCN(channels, channels)

        # 3. Temporal Conv 2
        self.temp_conv2 = nn.Conv2d(channels, channels, kernel_size=(1, 3), padding=(0, 1))

        self.dropout = nn.Dropout(dropout)
        self.bn = nn.BatchNorm2d(channels)

    def forward(self, x, adj):
        """
        x: (B, C, N, T)
        adj: (N, N)
        """
        residual = x

        # --- Temporal 1 ---
        x = self.temp_conv1(x)
        x = F.relu(x)

        # --- Spatial GCN ---
        # 转换维度适配 GCN: (B, C, N, T) -> (B*T, N, C)
        B, C, N, T = x.shape
        x_gcn = x.permute(0, 3, 2, 1).reshape(B * T, N, C)

        x_gcn = self.gcn(x_gcn, adj)  # (B*T, N, C)

        # 还原维度: (B*T, N, C) -> (B, C, N, T)
        x = x_gcn.reshape(B, T, N, C).permute(0, 3, 2, 1)

        # --- Temporal 2 ---
        x = self.temp_conv2(x)
        x = self.bn(x)

        # --- Residual + Dropout ---
        x = x + residual
        x = self.dropout(x)

        return x


class SimpleSTGCN(nn.Module):
    def __init__(self, T_in, T_out, H, W, d_model=64):
        super(SimpleSTGCN, self).__init__()
        self.H = H
        self.W = W
        self.N = H * W
        self.T_in = T_in
        self.T_out = T_out

        # 1. 自适应邻接矩阵 (Learnable Adjacency Matrix)
        # 不需要外部传入 adj，让模型自己学习节点间的关系
        # 初始化为随机，训练中自动优化
        self.node_embeddings = nn.Parameter(torch.randn(self.N, 10), requires_grad=True)

        # 2. 输入层: 将 1 维流量映射到 d_model
        self.input_proj = nn.Linear(1, d_model)

        # 3. ST-Blocks (堆叠 2 层)
        self.st_blocks = nn.ModuleList([
            STBlock(d_model, self.N),
            STBlock(d_model, self.N)
        ])

        # 4. 输出层: 将 T_in 维度 映射到 T_out
        # 我们在这里做一个简单的时间轴投影
        self.output_proj1 = nn.Linear(d_model, d_model)
        self.output_proj2 = nn.Linear(T_in, T_out)
        self.final_proj = nn.Linear(d_model, 1)  # 映射回 1 维流量

    def forward(self, x, *args, **kwargs):
        """
        Input: (B, T_in, 1, H, W)
        Output: (B, T_out, 1, H, W)
        """
        # ==========================
        # 1. 预处理
        # ==========================
        if x.dim() == 5:
            x = x.squeeze(2)  # (B, T, H, W)

        B, T, H, W = x.shape
        # Flatten H, W -> N
        x = x.reshape(B, T, -1).unsqueeze(-1)  # (B, T, N, 1)

        # ==========================
        # 2. 构建邻接矩阵 (Adaptive)
        # ==========================
        # A = Softmax(ReLU(E * E^T))
        adj = F.relu(torch.mm(self.node_embeddings, self.node_embeddings.transpose(0, 1)))
        adj = F.softmax(adj, dim=1)  # 归一化

        # ==========================
        # 3. 特征嵌入
        # ==========================
        x = self.input_proj(x)  # (B, T, N, d_model)

        # 转换为 (B, C, N, T) 以适配 Conv2d
        x = x.permute(0, 3, 2, 1)

        # ==========================
        # 4. ST-Blocks 前向传播
        # ==========================
        for block in self.st_blocks:
            x = block(x, adj)

        # x shape: (B, d_model, N, T_in)

        # ==========================
        # 5. 输出预测
        # ==========================
        # 变换回 (B, N, T_in, d_model)
        x = x.permute(0, 2, 3, 1)

        # 特征投影
        x = F.relu(self.output_proj1(x))  # (B, N, T_in, d_model)

        # 时间维度投影: T_in -> T_out
        x = x.permute(0, 1, 3, 2)  # (B, N, d_model, T_in)
        x = self.output_proj2(x)  # (B, N, d_model, T_out)

        # 映射回数值: d_model -> 1
        x = x.permute(0, 3, 1, 2)  # (B, T_out, N, d_model)
        x = self.final_proj(x)  # (B, T_out, N, 1)

        # ==========================
        # 6. 还原形状
        # ==========================
        x = x.reshape(B, self.T_out, H, W).unsqueeze(2)  # (B, T_out, 1, H, W)

        return x