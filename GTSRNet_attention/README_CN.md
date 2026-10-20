# GTSRNet 注意力可解释性与扰动实验

## 目的

本实验回答三个问题：

1. 不同层、不同头是否学习到非均匀且互有差异的网格关系；
2. 高峰、低谷和异常时段的空间关系是否发生合理变化；
3. 高注意力网格被扰动时，性能下降是否大于低注意力和随机网格。

实验直接加载已有 `GTSRNet/best.pt`，注意力提取和输入扰动均不需要重新训练。
`attention_model.py` 只增加返回注意力矩阵的接口，不增加或更改模型参数，因而可以
严格加载原 checkpoint。

## 放置目录

把整个文件夹上传到服务器项目根目录：

```text
GLSTTN_tars-duobunewdata/
├── dataset/
├── utils/
├── GTSRNet_model/
├── results_GTSRNet_baselines/       # 已完成的 seed 16/26/36 实验结果
└── GTSRNet_attention/    # 本文件夹
```

推荐直接使用已经完成的基线实验 checkpoint，目录应包含：

```text
results_GTSRNet_baselines/<dataset>/GTSRNet/seed_16/best.pt
results_GTSRNet_baselines/<dataset>/GTSRNet/seed_26/best.pt
results_GTSRNet_baselines/<dataset>/GTSRNet/seed_36/best.pt
```

支持的数据集为 `milan_sms`、`milan_call`、`milan_internet`、`mobile_bj`、
`mobile_nj`。

## 实验定义

- 注意力矩阵：空间 Transformer 每层每头的 `[N,N]` 行归一化矩阵；
- 事件选择：训练集第 95/5 百分位对应高峰/低谷；异常为相对小时均值绝对
  z-score 最大的测试窗口；
- 注意力汇总：各头平均、加入残差单位阵后进行 attention rollout；
- 遮挡值：训练集对应小时、对应网格的平均流量，不直接置零；
- 遮挡预算：默认 5%、10%、20% 网格；
- 对照：高注意力、低注意力、随机、最高注意力网格周围的等数量局部区域，以及
  高注意力位置打乱；
- 输出：同时评价 `spatial_branch` 和最终 `GTSRNet`，指标为逐窗口
  `delta_rmse`、`delta_mae`。

随机遮挡和位置打乱默认重复 5 次。论文正式实验建议 10 次，并至少分析 64 个均匀
覆盖测试期的窗口。事件样本无论是否落在均匀抽样中都会自动加入。

## 1. 单数据集小规模检查

在项目根目录运行：

```bash
cd /path/to/GLSTTN_tars-duobunewdata

python -u ./GTSRNet_attention/smoke_test.py \
  --project-root "$PWD" --device cpu

PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
python -u ./GTSRNet_attention/analyze_attention.py \
  --project-root "$PWD" \
  --checkpoint "$PWD/results_GTSRNet_baselines/milan_sms/GTSRNet/seed_16/best.pt" \
  --dataset milan_sms \
  --output-dir "$PWD/results_GTSRNet_attention/smoke/milan_sms" \
  --device cuda --gpu 0 \
  --batch-size 1 \
  --max-test-samples 4 \
  --fractions 0.10 \
  --random-repeats 1
```

如果服务器环境安装了另一个同名 `utils` 包，可显式把项目路径放在最前面（修正版
`smoke_test.py` 已自动处理；下面命令也可作为保险）：

```bash
PYTHONPATH="$PWD:$PWD/GTSRNet_model:${PYTHONPATH:-}" \
python -u ./GTSRNet_attention/smoke_test.py \
  --project-root "$PWD" --device cpu
```

成功后应生成：

```text
manifest.json
attention_statistics.csv
perturbation_results.csv
events/peak/attention_heads.png
events/peak/attention_matrices.png
events/peak/attention_rollout_importance.png
events/trough/attention_heads.png
events/trough/attention_matrices.png
events/trough/attention_rollout_importance.png
events/anomaly/attention_heads.png
events/anomaly/attention_matrices.png
events/anomaly/attention_rollout_importance.png
```

## 2. 一个 seed、五个数据集、四张 GPU

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
python -u ./GTSRNet_attention/run_attention_analysis.py \
  --project-root "$PWD" \
  --baseline-results-root "$PWD/results_GTSRNet_baselines" \
  --seeds 16 \
  --output-dir "$PWD/results_GTSRNet_attention" \
  --datasets milan_sms milan_call milan_internet mobile_nj mobile_bj \
  --gpus 0 1 2 3 \
  --batch-size 2 \
  --max-test-samples 64 \
  --fractions 0.05 0.10 0.20 \
  --random-repeats 10 \
  --analysis-seed 2026
```

程序支持断点续跑：存在 `manifest.json` 的任务会跳过。不要加 `--force`，除非确认
需要覆盖已完成的分析。

## 3. 推荐的三 seed 正式实验

你已有基线实验的 seed 16、26、36 checkpoint，不需要重新训练 `GTSRNet`。直接运行：

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
python -u ./GTSRNet_attention/run_attention_analysis.py \
  --project-root "$PWD" \
  --baseline-results-root "$PWD/results_GTSRNet_baselines" \
  --seeds 16 26 36 \
  --output-dir "$PWD/results_GTSRNet_attention_3seed" \
  --datasets milan_sms milan_call milan_internet mobile_nj mobile_bj \
  --gpus 0 1 2 3 \
  --batch-size 2 \
  --max-test-samples 64 \
  --fractions 0.05 0.10 0.20 \
  --random-repeats 10 \
  --analysis-seed 2026
```

显存不足时首先把 `--batch-size 2` 改为 `1`；不要减少遮挡对照种类。若只想先检查
命令和 checkpoint 是否齐全，可在批量命令末尾加 `--dry-run`。

## 4. 单独重新汇总

```bash
python -u ./GTSRNet_attention/summarize_attention.py \
  --results-dir "$PWD/results_GTSRNet_attention_3seed"
```

汇总文件位于：

```text
summary/attention_summary.csv
summary/perturbation_summary.csv
summary/high_vs_random_seed.csv
summary/high_vs_random_paired.csv
summary/figures/fig_perturbation_*.png
```

## 结果判读

重点检查：

1. `normalized_entropy` 是否长期接近 1；若是，注意力近似均匀；
2. `self_mass` 是否远高于随机基准 `1/N`；若是，可能只关注自身；
3. `mean_pairwise_head_js` 是否接近 0；若是，多头可能塌缩；
4. `mask_high_attention` 的性能下降是否稳定大于 `mask_random` 和
   `mask_low_attention`；
5. `high_vs_random_seed.csv` 先检查三个 seed 的方向是否一致，再检查
   `high_vs_random_paired.csv` 中 `high_minus_random` 是否为正，以及分层配对 bootstrap
   置信区间是否不跨 0；
6. 空间分支和最终融合结果是否方向一致。

如果高注意力遮挡没有比随机遮挡造成更明显的下降，应如实报告，不应仅凭热图宣称
注意力具有因果解释性。此时可以把结论收缩为“注意力提供动态关系表示，但注意力权重
本身不等同于完整特征归因”。
