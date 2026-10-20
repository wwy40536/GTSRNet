class SegRNNBackbone(SegRNNGrid):
    """
    修改版 SegRNN：
    1. 不再输出最终预测值 (B, Tout, H, W)
    2. 而是输出时间特征 (B, Tout, N, D)
    3. 去掉了最后的 denorm (反归一化) 步骤，因为现在处理的是特征
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # 这里不需要改动 __init__，复用父类的组件即可
        # self.predict 层虽然还在，但我们在 forward 里不使用它

    def forward(self, x: torch.Tensor):
        """
        输入: x (B, Tin, 1, H, W)
        输出: time_feats (B, Tout, N, D)
        """
        B, Tin, C, H, W = x.shape

        # 1. 预处理 (与原版一致)
        x = x.squeeze(2).reshape(B, Tin, self.N)

        # === 注意：保留 Last Value 用于后续最终模型输出时的反归一化，但这里不加 ===
        seq_last = x[:, -1:, :].detach()
        x = x - seq_last  # 归一化输入

        x = x.permute(0, 2, 1).contiguous()  # (B,N,Tin)

        # 2. 分段嵌入 (这就是你想要的 "增加维度成为嵌入层")
        # 输入 (B*N, n, seg_len) -> 输出 (B*N, n, D)
        x_seg = x.view(B * self.N, self.seg_num_x, self.seg_len)
        x_emb = self.valueEmbedding(x_seg)

        # 3. GRU Encoder (与原版一致)
        enc_out, hn = self.rnn(x_emb)

        # 4. 构造 Decoder Token (与原版一致)
        pos = self.pos_emb.unsqueeze(0).repeat(self.N, 1, 1)
        cell = self._build_cell_emb(x.device, x.dtype)
        chn = cell.unsqueeze(1).repeat(1, self.seg_num_y, 1)
        dec_tok = torch.cat([pos, chn], dim=-1).view(-1, 1, self.d_model)
        dec_tok = dec_tok.repeat(B, 1, 1)
        hn_rep = hn.repeat(1, 1, self.seg_num_y).view(1, -1, self.d_model)

        # 5. GRU Decoder (得到隐状态)
        _, hy = self.rnn(dec_tok, hn_rep)  # hy: (1, B*N*m, D)

        # 6. Cross-Attention (增强特征)
        if self.use_cross_attn:
            BN = B * self.N
            m = self.seg_num_y
            d = self.d_model
            hy_q = hy.squeeze(0).view(BN, m, d)

            q = self.q_proj(hy_q)
            k = self.k_proj(enc_out)
            v = self.v_proj(enc_out)

            attn = torch.matmul(q, k.transpose(1, 2)) / math.sqrt(d)
            attn = F.softmax(attn, dim=-1)
            attn = self.attn_drop(attn)

            ctx = torch.matmul(attn, v)
            ctx = self.ctx_proj(ctx)

            hy_fuse = hy_q + ctx  # (BN, m, D)
        else:
            BN = B * self.N
            m = self.seg_num_y
            d = self.d_model
            hy_fuse = hy.squeeze(0).view(BN, m, d)

        # ==================== 修改核心区 ====================

        # 目前 hy_fuse 形状: (B*N, m, D)
        # 其中 m 是段数 (m = Tout // seg_len)
        # D 是特征维度

        # 还原 Batch 和 N
        temporal_feats = hy_fuse.view(B, self.N, m, d)  # (B, N, m, D)

        # 【关键步骤】:
        # 因为 SegRNN 是对每个 segment (长度 seg_len) 生成一个特征向量
        # 但我们需要对齐到 (Tout) 时间步，以便和空间模块融合
        # 方法：将每个 segment 的特征复制 seg_len 次

        if self.seg_len > 1:
            # (B, N, m, 1, D) -> repeat -> (B, N, m, seg_len, D)
            temporal_feats = temporal_feats.unsqueeze(3).repeat(1, 1, 1, self.seg_len, 1)
            # Flatten -> (B, N, Tout, D)
            temporal_feats = temporal_feats.view(B, self.N, self.Tout, d)
        else:
            # 如果 seg_len=1，直接 reshape
            temporal_feats = temporal_feats.view(B, self.N, self.Tout, d)

        # 调整为你的目标格式: (B, Tout, N, D)
        temporal_feats = temporal_feats.permute(0, 2, 1, 3).contiguous()

        return temporal_feats

    import torch

    # 假设 SegRNNBackbone 和 SegRNNGrid 已经在同一个文件中定义
    # 如果是在单独的文件中测试，请反注释下面的 import
    # from utils.segrnn_grid import SegRNNBackbone

    def test_segrnn_backbone():
        print("===== 开始测试 SegRNNBackbone =====")

        # 1. 基础参数设置
        B = 2  # Batch size
        Tin = 24  # 输入时间步
        Tout = 12  # 输出时间步
        H, W = 5, 5  # 网格大小
        N = H * W  # 节点数 = 25
        C = 1  # 输入通道
        d_model = 64  # 隐层维度

        # ---------------------------------------------------------
        # 测试案例 A: seg_len > 1 (测试维度扩展/Repeat逻辑)
        # ---------------------------------------------------------
        seg_len_a = 2
        print(f"\n[Case A] 测试 seg_len={seg_len_a} (存在分段)...")

        model_a = SegRNNBackbone(
            Tin=Tin, Tout=Tout, H=H, W=W,
            d_model=d_model, seg_len=seg_len_a,
            use_cross_attn=1
        )

        # 构造假数据 (B, Tin, 1, H, W)
        x_input = torch.randn(B, Tin, C, H, W)

        # 前向传播
        try:
            output_a = model_a(x_input)

            # 验证形状
            expected_shape = (B, Tout, N, d_model)  # (2, 12, 25, 64)

            print(f"输入形状: {x_input.shape}")
            print(f"输出形状: {output_a.shape}")
            print(f"期望形状: {expected_shape}")

            assert output_a.shape == expected_shape, \
                f"Case A 失败: 形状不匹配! Got {output_a.shape}, Expected {expected_shape}"

            print(">> Case A 通过: 维度扩展逻辑 (repeat) 正常。")

        except Exception as e:
            print(f">> Case A 发生错误: {e}")
            import traceback
            traceback.print_exc()

        # ---------------------------------------------------------
        # 测试案例 B: seg_len = 1 (测试不分段的 Reshape 逻辑)
        # ---------------------------------------------------------
        seg_len_b = 1
        print(f"\n[Case B] 测试 seg_len={seg_len_b} (逐点预测)...")

        model_b = SegRNNBackbone(
            Tin=Tin, Tout=Tout, H=H, W=W,
            d_model=d_model, seg_len=seg_len_b,
            use_cross_attn=1
        )

        try:
            output_b = model_b(x_input)

            # 验证形状
            assert output_b.shape == expected_shape, \
                f"Case B 失败: 形状不匹配! Got {output_b.shape}"

            print(">> Case B 通过: seg_len=1 时的 reshape 逻辑正常。")

        except Exception as e:
            print(f">> Case B 发生错误: {e}")

        # ---------------------------------------------------------
        # 测试案例 C: 梯度检查 (确保嵌入层参与计算)
        # ---------------------------------------------------------
        print("\n[Case C] 梯度流动检查...")
        loss = output_a.sum()
        loss.backward()

        # 检查 valueEmbedding 的权重是否有梯度
        has_grad = model_a.valueEmbedding[0].weight.grad is not None
        print(f"ValueEmbedding 梯度存在: {has_grad}")

        if has_grad:
            print(">> Case C 通过: 模型反向传播正常，可作为特征提取器训练。")
        else:
            print(">> Case C 警告: 没有检测到梯度 (如果是 eval 模式则是正常的)。")

        print("\n===== 所有测试完成 =====")

    if __name__ == "__main__":
        test_segrnn_backbone()