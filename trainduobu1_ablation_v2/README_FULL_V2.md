# full_v2：针对 Milan-Call 的改进版本

`full_v2` 以已经完成的 `full` 为基线，不覆盖旧的六个模型。它只针对本次 Call
实验暴露出的两个问题：动态凸门控无法校正两个分支的偏置/尺度，以及后期预测
（特别是 t+3 至 t+12）持续劣化。

## 模型公式

对每个预测步 `h=1,...,12`，先分别校准空间与时间分支：

```text
s_h = a_spatial[h] * y_spatial[h] + b_spatial[h]
t_h = a_temporal[h] * y_temporal[h] + b_temporal[h]
alpha_h,cell = sigmoid(MLP([s_h, t_h, horizon_embedding[h]]))
y_hat_h = alpha_h,cell * s_h + (1 - alpha_h,cell) * t_h
```

所有 scale 初始化为 1、bias 初始化为 0，门控输出初始化为 0.5，因此 v2 在训练
开始时等价于未校准的等权融合；不会由随机输出破坏已有分支。门控仍会随样本、
预测步和栅格变化，但现在有显式的预测步信息。

训练阶段增加分支辅助损失，验证集与测试集仍只评估最终融合输出：

```text
L_train = L_fusion + lambda * (L_spatial + L_temporal) / 2
lambda = --branch-aux-weight  (默认 0.1)
```

因此 checkpoint 的选择没有用辅助损失，也没有使用测试集。

## 如何追加到已有的 seed 15 结果

将整个 `trainduobu1_ablation_v2` 文件夹上传到服务器项目根目录。你已有的六个
模型位于 `results_trainduobu1_seed15_bs16`；下面只会训练 5 个 `full_v2` 任务，并
保留这些已有结果：

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
python -u ./trainduobu1_ablation_v2/run_trainduobu1_v2_ablation_4gpu.py \
  --project-root "$PWD" \
  --output-dir "$PWD/results_trainduobu1_seed15_bs16" \
  --datasets milan_sms milan_call milan_internet mobile_nj mobile_bj \
  --variants full_v2 \
  --gpus 0 1 2 3 \
  --batch-size 16 \
  --epochs 300 \
  --learning-rate 0.001 \
  --loss l1 \
  --e-layers 2 \
  --branch-aux-weight 0.1 \
  --seed 15 \
  --split-mode chronological \
  --split-gap -1 \
  --milan-loader clean
```

不要加入 `--force`。已有六个模型会保留；新模型完成后，目录中会有 35 份
`metrics.json`（5 个数据集 × 7 个变体），脚本自动生成包含 `T1 Gate-V2` 的柱状图。

## 先做冒烟检查

```bash
python -u ./trainduobu1_ablation_v2/train_trainduobu1_v2_ablation.py \
  --project-root "$PWD" --smoke-test --device cuda --gpu 0
```

## 如何解释结果

若 `full_v2` 优于 `full`，只能说明本次“逐步校准门控 + 辅助损失”在该固定协议下
有效；它不是论文原始 GTSRNet。应至少再用多个相同 seed 重复，报告均值和标准差；
不要根据测试集结果继续挑选 `lambda`、学习率或分段长度。
