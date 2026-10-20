import torch
from torch import nn


# ================================================================
# 1. 基础组件: MultiLayerPerceptron
# ================================================================
class MultiLayerPerceptron(nn.Module):
    """Multi-Layer Perceptron with residual links."""

    def __init__(self, input_dim, hidden_dim) -> None:
        super().__init__()
        self.fc1 = nn.Conv2d(
            in_channels=input_dim, out_channels=hidden_dim, kernel_size=(1, 1), bias=True)
        self.fc2 = nn.Conv2d(
            in_channels=hidden_dim, out_channels=hidden_dim, kernel_size=(1, 1), bias=True)
        self.act = nn.ReLU()
        self.drop = nn.Dropout(p=0.15)

    def forward(self, input_data: torch.Tensor) -> torch.Tensor:
        """Feed forward of MLP.
        Args:
            input_data (torch.Tensor): input data with shape [B, D, N, 1]
        Returns:
            torch.Tensor: latent repr
        """
        hidden = self.fc2(self.drop(self.act(self.fc1(input_data))))  # MLP
        hidden = hidden + input_data  # residual
        return hidden


# ================================================================
# 2. STID 主模型
# ================================================================
class STID(nn.Module):
    """
    Paper: Spatial-Temporal Identity: A Simple yet Effective Baseline for Multivariate Time Series Forecasting
    Link: https://arxiv.org/abs/2208.05233
    """

    def __init__(self, T_in, T_out, H, W,
                 input_dim=1,
                 embed_dim=32,
                 num_layer=3,
                 if_node=True,
                 node_dim=32,
                 if_T_i_D=True,
                 if_D_i_W=True,
                 temp_dim_tid=32,
                 temp_dim_diw=32,
                 time_of_day_size=288,
                 day_of_week_size=7):
        super().__init__()

        # 基本参数
        self.num_nodes = H * H
        self.H = H
        self.W = W
        self.input_len = T_in
        self.output_len = T_out
        self.input_dim = input_dim
        self.embed_dim = embed_dim
        self.num_layer = num_layer
        self.node_dim = node_dim

        # 时间嵌入配置
        self.if_time_in_day = if_T_i_D
        self.if_day_in_week = if_D_i_W
        self.temp_dim_tid = temp_dim_tid
        self.temp_dim_diw = temp_dim_diw
        self.time_of_day_size = time_of_day_size
        self.day_of_week_size = day_of_week_size
        self.if_spatial = if_node

        # ---------------------------------------------
        # 1. Spatial Embeddings
        # ---------------------------------------------
        if self.if_spatial:
            self.node_emb = nn.Parameter(
                torch.empty(self.num_nodes, self.node_dim))
            nn.init.xavier_uniform_(self.node_emb)

        # ---------------------------------------------
        # 2. Temporal Embeddings
        # ---------------------------------------------
        if self.if_time_in_day:
            self.time_in_day_emb = nn.Parameter(
                torch.empty(self.time_of_day_size, self.temp_dim_tid))
            nn.init.xavier_uniform_(self.time_in_day_emb)

        if self.if_day_in_week:
            self.day_in_week_emb = nn.Parameter(
                torch.empty(self.day_of_week_size, self.temp_dim_diw))
            nn.init.xavier_uniform_(self.day_in_week_emb)

        # ---------------------------------------------
        # 3. Structure
        # ---------------------------------------------
        self.time_series_emb_layer = nn.Conv2d(
            in_channels=self.input_dim * self.input_len, out_channels=self.embed_dim, kernel_size=(1, 1), bias=True)

        self.hidden_dim = self.embed_dim + \
                          self.node_dim * int(self.if_spatial) + \
                          self.temp_dim_tid * int(self.if_time_in_day) + \
                          self.temp_dim_diw * int(self.if_day_in_week)

        self.encoder = nn.Sequential(
            *[MultiLayerPerceptron(self.hidden_dim, self.hidden_dim) for _ in range(self.num_layer)])

        self.regression_layer = nn.Conv2d(
            in_channels=self.hidden_dim, out_channels=self.output_len, kernel_size=(1, 1), bias=True)

    def forward(self, x, x_mark_enc=None, *args, **kwargs):
        # 1. Input Adapter
        if x.dim() == 5:
            x = x.squeeze(2)  # (B, T, H, W)
        B, T, H, W = x.shape
        N = H * W
        input_data = x.reshape(B, T, N, 1)  # (B, T, N, 1)

        # 2. Get Temporal Embeddings
        time_in_day_emb = None
        day_in_week_emb = None

        if self.if_time_in_day:
            if x_mark_enc is not None:
                t_i_d_data = x_mark_enc[:, -1, -1]
                tid_index = (t_i_d_data * self.time_of_day_size).type(torch.LongTensor).to(x.device)
                tid_index = torch.clamp(tid_index, 0, self.time_of_day_size - 1)
            else:
                tid_index = torch.zeros(B, dtype=torch.long).to(x.device)
            time_in_day_emb = self.time_in_day_emb[tid_index]  # (B, D)

        if self.if_day_in_week:
            if x_mark_enc is not None:
                d_i_w_data = x_mark_enc[:, -1, 2]
                diw_index = (d_i_w_data * self.day_of_week_size).type(torch.LongTensor).to(x.device)
                diw_index = torch.clamp(diw_index, 0, self.day_of_week_size - 1)
            else:
                diw_index = torch.zeros(B, dtype=torch.long).to(x.device)
            day_in_week_emb = self.day_in_week_emb[diw_index]  # (B, D)

        # 3. Time Series Embedding
        batch_size, _, num_nodes, _ = input_data.shape
        input_data = input_data.transpose(1, 2).contiguous()  # (B, N, T, 1)
        input_data = input_data.view(batch_size, num_nodes, -1).transpose(1, 2).unsqueeze(-1)  # (B, T*1, N, 1)
        time_series_emb = self.time_series_emb_layer(input_data)  # (B, D, N, 1)

        # 4. Concatenate Embeddings
        node_emb = []
        if self.if_spatial:
            # (N, D) -> (1, D, N, 1) -> (B, D, N, 1)
            node_emb.append(self.node_emb.transpose(0, 1).unsqueeze(0).unsqueeze(-1).expand(batch_size, -1, -1, -1))

        tem_emb = []
        if time_in_day_emb is not None:
            # (B, D) -> (B, D, 1, 1) -> (B, D, N, 1)
            tem_emb.append(time_in_day_emb.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, num_nodes, -1))

        if day_in_week_emb is not None:
            # (B, D) -> (B, D, 1, 1) -> (B, D, N, 1)
            tem_emb.append(day_in_week_emb.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, num_nodes, -1))

        hidden = torch.cat([time_series_emb] + node_emb + tem_emb, dim=1)

        # 5. Encoder
        hidden = self.encoder(hidden)

        # 6. Regression
        prediction = self.regression_layer(hidden)  # (B, T_out, N, 1)

        # 7. Output Adapter
        prediction = prediction.squeeze(-1).transpose(1, 2)  # (B, N, T_out)
        prediction = prediction.transpose(1, 2).reshape(B, self.output_len, self.H, self.W).unsqueeze(2)

        return prediction