# GTSRNet_sensitivity 审稿补实验包

本文件夹针对三条审稿意见：

1. 解释并验证为何使用 24 小时历史窗口和 12 小时预测范围；
2. 在全局 MAE/RMSE 外，补充可解释指标以及逐预测步、逐网格结果。
3. 详细报告第 1–12 步的误差分布、P95/P99、最大误差和最差网格。

实验模型与已完成结果中的 `GTSRNet_model/GTSRNet` 保持一致。唯一的兼容性修改是：当预测 3 小时而 `seg_len=6` 时，SegRNN 内部解码一个完整的 6 步片段，再截取前 3 步参与损失和评测；所有实验仍保持 `seg_len=6`，不会为了 3 小时任务偷偷改变时间分支结构。

## 放置位置

把整个 `GTSRNet_sensitivity` 放到项目根目录：

```text
GLSTTN_tars-duobunewdata/
├── dataset/
├── utils/
├── GTSRNet_model/
├── results_original_v2_seed2_bs16/
└── GTSRNet_sensitivity/
```

## 1. 冒烟测试

```bash
python -u ./GTSRNet_sensitivity/train_sensitivity.py \
  --project-root "$PWD" --smoke-test --device cuda --gpu 0
```

该测试专门包含 `history=12, horizon=3, seg_len=6`，检查 3 小时预测的补齐/截取、前向和反向传播。

## 2. 24 小时窗口的 ACF/PACF 和周期依据

只使用测试段之前的数据，默认分析 Milan 三个已报告数据集：

```bash
python -u ./GTSRNet_sensitivity/analyze_temporal_dependence.py \
  --project-root "$PWD" \
  --output-dir "$PWD/results_GTSRNet_sensitivity/dependence" \
  --datasets milan_sms milan_call milan_internet \
  --max-lag 72 --max-period 168
```

脚本会检查 Milan 时间戳是否严格为 1 小时间隔。如果不是，会明确警告或拒绝把“步”写成“小时”。生成 `dependence_summary.csv`、完整 ACF/PACF、频谱峰值表和每个数据集的图。

## 3. 输入窗口与预测范围敏感性

采用单因素设计，共 7 个唯一设置：

- 固定预测 12 h，输入历史取 12/24/48/72 h；
- 固定输入历史 24 h，预测范围取 3/6/12/24 h；
- 共享的 24→12 只运行一次。

为保证公平，所有预测范围使用相同的最终预测起点和相同的 23 窗口切分隔离（由最大预测范围 24 自动得到），所以原结果中 `gap=11` 且未对齐预测起点的 24→12 checkpoint不能直接混入敏感性表，需要在新协议下重跑。

先预览任务：

```bash
python -u ./GTSRNet_sensitivity/run_sensitivity.py \
  --project-root "$PWD" --dry-run
```

与现有论文结果对齐的单 seed（seed=2，三数据集，共 21 个任务）：

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
python -u ./GTSRNet_sensitivity/run_sensitivity.py \
  --project-root "$PWD" \
  --output-dir "$PWD/results_GTSRNet_sensitivity" \
  --datasets milan_sms milan_call milan_internet \
  --histories 12 24 48 72 \
  --horizons 3 6 12 24 \
  --base-history 24 --base-horizon 12 \
  --seeds 2 \
  --gpus 0 1 2 3 \
  --batch-size 16 --epochs 300 \
  --learning-rate 0.001 --loss l1 \
  --branch-aux-weight 0.1 \
  --d-model 128 --seg-len 6 --n-heads 4 --e-layers 2 \
  --split-mode chronological --split-gap -1 \
  --milan-loader clean --mase-seasonality 24
```

如果算力允许，推荐三 seed：把上面的 `--seeds 2` 改为 `--seeds 2 12 22`，共 63 个任务，并在论文表中报告均值 ± 样本标准差。

任务支持断点续跑；已有 `metrics.json` 会自动跳过。除非确实要覆盖已完成结果，否则不要加 `--force`。

## 4. 单独重新汇总

```bash
python -u ./GTSRNet_sensitivity/summarize_sensitivity.py \
  --results-dir "$PWD/results_GTSRNet_sensitivity" \
  --histories 12 24 48 72 --horizons 3 6 12 24 \
  --base-history 24 --base-horizon 12
```

汇总目录包含每次运行、逐预测步、效率、输入窗口表、预测范围表和数据集敏感性图。表中包含 MAE、RMSE、R²、sMAPE、MASE、PSNR。

## 5. 给已有 checkpoint 补指标

无需重训，详见 `METRICS_REVIEWER_RESPONSE_CN.md`。最常用命令：

```bash
python -u ./GTSRNet_sensitivity/augment_existing_metrics.py \
  --project-root "$PWD" \
  --results-dir "$PWD/results_original_v2_seed2_bs16" \
  --datasets milan_sms milan_call milan_internet \
  --variants GTSRNet --gpus 0 1 2
```

## 6. 第 1–12 步误差分布（无需重新训练）

完整方法和回复模板见 `MULTISTEP_REVIEWER_RESPONSE_CN.md`。依次运行：

```bash
python -u ./GTSRNet_sensitivity/export_multistep_predictions.py \
  --project-root "$PWD" \
  --results-dir "$PWD/results_original_v2_seed2_bs16" \
  --datasets milan_sms milan_call milan_internet \
  --device cuda --gpu 0

python -u ./GTSRNet_sensitivity/analyze_multistep_errors.py \
  --predictions-dir "$PWD/results_original_v2_seed2_bs16/multistep_diagnostics" \
  --datasets milan_sms milan_call milan_internet
```

## 文件说明

- `train_sensitivity.py`：训练/评测入口；输出整体、逐步、逐网格六类指标。
- `gtsrnet_model.py`：与 GTSRNet 对齐并支持 3 小时输出。
- `run_sensitivity.py`：四卡动态任务队列和断点续跑。
- `summarize_sensitivity.py`：多 seed 汇总、论文表格和敏感性图。
- `analyze_temporal_dependence.py`：训练前数据的 ACF/PACF/频谱分析。
- `augment_existing_metrics.py`：读取已有 best.pt，仅重新评测并补指标。
- `plot_metric_diagnostics.py`：逐预测步曲线和逐网格热力图。
- `export_multistep_predictions.py`：可选；复现已有 MAE/RMSE 后导出原始尺度预测。
- `analyze_multistep_errors.py`：逐步标准差、P95/P99、最大误差、箱线图和 CDF。
- `METRICS_REVIEWER_RESPONSE_CN.md`：指标定义、论文放置建议和回复模板。
- `MULTISTEP_REVIEWER_RESPONSE_CN.md`：多步误差审稿意见的完整处理说明。
