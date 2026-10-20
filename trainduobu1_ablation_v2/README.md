# trainduobu1 修正后的消融实验版本

## 定位

这不是论文原始 GTSRNet 的复现版本，而是基于 `scripts/trainduobu1.py` 的模型思路
构建的改进模型：空间 Transformer 分支 + SegRNN 时间分支 + 动态 MLP 门控融合。若把
结果写入论文，应称为“modified trainduobu1 / T1 Dynamic Gate”，不能称为论文原始模型。

## 原压缩包中已修复的问题

1. `st_iseg_combined1.py` 将空间输出 `[B,T,1,H,W]` 与时间输出 `[B,T,H*W]`
   直接 `torch.stack`，两者形状不一致，默认模型无法正确融合。
2. `dataset.py` 在训练/验证/测试切分前对全部数据拟合 min-max，测试信息泄漏到训练。
3. 验证集从相互重叠的滑动窗口中随机抽取，训练目标和验证目标大量重合。
4. `trainduobu1.py` 忽略 `-cuda` 参数、固定使用 `cuda:0`，并把损失函数固定放到默认
   GPU；多卡运行时会错卡。
5. 含 `meta` 或 `cross` 的验证分支仍会反向传播和更新参数。
6. 模型有小时嵌入，但训练脚本没有把 `X_meta` 的小时传给模型，因此该嵌入始终使用
   phase 0。
7. 同一测试集被预测两次；逐步指标漏掉 `t+12`；scheduler 在每个 epoch 开始前 step。

新训练入口在训练观察窗口上拟合归一化、采用按时间顺序且带预测窗口间隔的
train/validation/test 切分，并把 phase 明确传给空间分支。

## 六个模型变体

```text
full      空间 Transformer + SegRNN + 动态 MLP 门控（主模型）
wo_g      用 Concat + Linear 取代动态门控
wo_seg    时间支路取消分段，seg_len=1
wo_sthe   空间支路移除 channel/hour/joint 嵌入
wo_s      仅保留 SegRNN 时间支路
wo_t      仅保留空间 Transformer 支路
```

`full` 中的门控是逐样本、逐预测步、逐栅格的动态门控；它比标量 alpha 更灵活，但也
不是论文中的标量门控。`wo_g` 是其真实的无动态门控对照。

## 上传方式

将整个 `trainduobu1_ablation_fixed` 文件夹上传到服务器项目根目录：

```text
GLSTTN_tars-duobunewdata/
├── dataset/
├── utils/
├── scripts/
└── trainduobu1_ablation_fixed/
```

不要覆盖原来的 `scripts/trainduobu1.py` 或 `utils/st_iseg_combined1.py`。

## 先进行模型冒烟检查

在服务器项目根目录、`glsttn` 环境中运行：

```bash
python -u ./trainduobu1_ablation_fixed/train_trainduobu1_ablation.py \
  --project-root "$PWD" --smoke-test --device cuda --gpu 0
```

它检查全部六个变体的前向、反向和输出尺寸，不读取数据，也不进行正式训练。

## 四张显卡运行五个数据集的六模型消融

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
python -u ./trainduobu1_ablation_fixed/run_trainduobu1_ablation_4gpu.py \
  --project-root "$PWD" \
  --output-dir "$PWD/results_trainduobu1_seed15_bs16" \
  --datasets milan_sms milan_call milan_internet mobile_nj mobile_bj \
  --variants full wo_g wo_seg wo_sthe wo_s wo_t \
  --gpus 0 1 2 3 \
  --batch-size 16 \
  --epochs 300 \
  --learning-rate 0.001 \
  --loss l1 \
  --e-layers 2 \
  --seed 15 \
  --split-mode chronological \
  --split-gap -1 \
  --milan-loader clean
```

日志会实时显示，并保存为：

```text
results_trainduobu1_seed15_bs16/logs/<dataset>/<variant>.log
```

所有 30 个任务成功后，自动生成：

```text
results_trainduobu1_seed15_bs16/figures/fig_ablation_milan_rmse.png
results_trainduobu1_seed15_bs16/figures/fig_ablation_milan_mae.png
results_trainduobu1_seed15_bs16/figures/fig_ablation_mobile_bj_nj_rmse.png
results_trainduobu1_seed15_bs16/figures/fig_ablation_mobile_bj_nj_mae.png
results_trainduobu1_seed15_bs16/summary_all.csv
```

## 结果解释边界

这份实现不能保证 `full` 在每个数据集、每个 seed 都优于全部消融；真实的结果应由
相同数据、相同切分、相同 seed 和相同训练预算得到。若 `full` 没有稳定优于 `wo_g`
或单分支模型，应如实报告，或再从验证集而不是测试集调整模型/超参数。
