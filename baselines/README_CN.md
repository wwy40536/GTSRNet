# GTSRNet 外部基线统一实验

## 1. 为什么必须做实验

审稿意见明确指出相关工作中讨论了多种近期模型，但实验表格没有对应基线。因此仅补充文字或引用不能解决问题，必须在与 `GTSRNet` 相同的数据划分和评价协议下重新训练外部基线。

本文件夹包含以下方法：

| 方法 | 类别 | 论文中的建议名称 |
|---|---|---|
| `seasonal_ha` | 统计方法 | S-HA |
| `gru_seq2seq` | RNN | GRU-Seq2Seq |
| `mtgnn` | 图神经网络 | MTGNN |
| `patchtst` | Transformer | PatchTST |
| `dlinear` | 直接多步预测 | DLinear |
| `timemixer` | 近期多尺度方法 | TimeMixer |
| `GTSRNet` | 本文完整模型 | GTSRNet |

所有模型统一使用：24 步历史输入、12 步直接/序列预测、按时间顺序划分、训练/验证/测试边界去除 11 个重叠窗口、仅用训练集拟合 min-max、验证集选择 checkpoint、测试集反归一化后计算 RMSE 和 MAE。

## 2. 上传位置

把整个文件夹放到服务器项目根目录：

```text
GLSTTN_tars-duobunewdata/
├── dataset/
├── utils/
├── GTSRNet_model/
└── baselines/             <- 本文件夹
```

进入原项目环境和项目根目录：

```bash
conda activate glsttn
cd /你的路径/GLSTTN_tars-duobunewdata
```

## 3. 先做冒烟测试

下面只使用合成小数据，检查七个模型的前向、反向和输出形状：

```bash
python -u ./baselines/train_baseline.py \
  --project-root "$PWD" \
  --model all \
  --smoke-test \
  --device cuda \
  --gpu 0
```

全部出现 `PASS` 后再执行正式实验。若服务器暂时没有可用 GPU，可将 `--device cuda --gpu 0` 改为 `--device cpu`。

## 4. 检查任务清单而不训练

```bash
python -u ./baselines/run_baselines.py \
  --project-root "$PWD" \
  --output-dir "$PWD/results_GTSRNet_baselines" \
  --models seasonal_ha gru_seq2seq mtgnn patchtst dlinear timemixer GTSRNet \
  --datasets milan_sms milan_call milan_internet mobile_bj mobile_nj \
  --seeds 15 25 35 \
  --gpus 0 1 2 3 \
  --dry-run
```

默认共有 95 个任务：6 个可训练模型 × 5 个数据集 × 3 个种子，加 5 个不需要重复种子的 S-HA 任务。

## 5. 四张 GPU 正式运行

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
python -u ./baselines/run_baselines.py \
  --project-root "$PWD" \
  --output-dir "$PWD/results_GTSRNet_baselines" \
  --models seasonal_ha gru_seq2seq mtgnn patchtst dlinear timemixer GTSRNet \
  --datasets milan_sms milan_call milan_internet mobile_bj mobile_nj \
  --seeds 15 25 35 \
  --gpus 0 1 2 3 \
  --close-size 24 \
  --predict-size 12 \
  --batch-size 16 \
  --epochs 300 \
  --learning-rate 0.001 \
  --loss l1 \
  --branch-aux-weight 0.1 \
  --split-mode chronological \
  --split-gap -1 \
  --milan-loader clean \
  --patience 0 \
  --num-workers 0
```

脚本支持断点续跑：已有 `metrics.json` 的任务会自动跳过，因此中断后直接重新执行同一命令即可。不要加 `--force`，除非确实要覆盖已经完成的实验。

## 6. 显存不足时

优先将批量从 16 降为 8，并用新的输出目录，不能把不同批量的结果混在同一目录：

```bash
python -u ./baselines/run_baselines.py \
  --project-root "$PWD" \
  --output-dir "$PWD/results_GTSRNet_baselines_bs8" \
  --batch-size 8 \
  --epochs 300 \
  --seeds 15 25 35 \
  --gpus 0 1 2 3
```

如果只有一张 GPU，改为 `--gpus 0`，其余参数不变。

## 7. 只跑一个模型检查真实数据

例如先在 Milan SMS 上跑一个 DLinear：

```bash
python -u ./baselines/train_baseline.py \
  --project-root "$PWD" \
  --output-dir "$PWD/results_baseline_check" \
  --model dlinear \
  --dataset milan_sms \
  --seed 15 \
  --device cuda \
  --gpu 0 \
  --batch-size 16 \
  --epochs 3 \
  --split-mode chronological \
  --split-gap -1 \
  --milan-loader clean
```

这个三轮命令只能检查数据与训练流程，不能作为论文结果。

## 8. 汇总已有结果

完整任务正常结束后会自动汇总。也可以手动执行：

```bash
python -u ./baselines/summarize_baselines.py \
  --results-dir "$PWD/results_GTSRNet_baselines" \
  --models seasonal_ha gru_seq2seq mtgnn patchtst dlinear timemixer GTSRNet \
  --datasets milan_sms milan_call milan_internet mobile_bj mobile_nj \
  --seeds 16 26 36
```

主要输出：

```text
results_GTSRNet_baselines/summary/all_runs.csv
results_GTSRNet_baselines/summary/baseline_summary.csv
results_GTSRNet_baselines/summary/per_horizon_summary.csv
results_GTSRNet_baselines/summary/paper_table.csv
results_GTSRNet_baselines/summary/figures/fig_baselines_*.png
```

论文主表使用 `paper_table.csv`，逐预测步分析使用 `per_horizon_summary.csv`。S-HA 是确定性方法，因此标准差为 0；神经模型报告三个种子的均值和样本标准差。

## 9. 查看每个随机种子的独立结果图（不合并）

下面的命令为每个数据集、每个随机种子分别绘图。图中是该种子的原始
RMSE/MAE，不计算跨种子均值，也不绘制标准差误差棒。S-HA 没有随机初始化，
所以同一个确定性结果会作为参照重复显示在 seed 16/26/36 的图中。

```bash
python -u ./baselines/plot_runs.py \
  --results-dir "$PWD/results_GTSRNet_baselines" \
  --seeds 16 26 36 \
  --formats png
```

共生成 5 × 3 = 15 张 PNG，目录为：

```text
results_GTSRNet_baselines/summary/figures_by_seed/
├── seed_16/
├── seed_26/
└── seed_36/
```

如果论文排版还需要矢量 PDF，同时指定 `--formats png pdf` 即可。

## 10. GTSRNet 逐网格空间误差分析

此步骤不重新训练。它加载5个数据集上 seed 16/26/36 已有的 `best.pt`，
仅补跑测试集推理并保存逐样本预测。原来的 `metrics.json` 和训练日志均会保留。

第一步：使用四张 GPU 导出15个 `predictions.npz`：

```bash
python -u ./baselines/run_baselines.py \
  --project-root "$PWD" \
  --output-dir "$PWD/results_GTSRNet_baselines" \
  --models GTSRNet \
  --datasets milan_sms milan_call milan_internet mobile_bj mobile_nj \
  --seeds 16 26 36 \
  --gpus 0 1 2 3 \
  --close-size 24 --predict-size 12 --batch-size 16 --epochs 300 \
  --learning-rate 0.001 --loss l1 --branch-aux-weight 0.1 \
  --split-mode chronological --split-gap -1 --milan-loader clean --patience 0 \
  --eval-only --save-predictions --no-summary
```

第二步：分别生成每个数据集、每个种子的空间诊断图：

```bash
python -u ./baselines/analyze_spatial_errors.py \
  --results-dir "$PWD/results_GTSRNet_baselines" \
  --datasets milan_sms milan_call milan_internet mobile_bj mobile_nj \
  --seeds 16 26 36 \
  --metric mae --curve-horizon 12 --curve-points 168 \
  --formats png pdf
```

每个种子独立输出，不跨种子合并：

```text
results_GTSRNet_baselines/spatial_analysis/<dataset>/seed_<seed>/
├── spatial_diagnostics.png
├── spatial_diagnostics.pdf
├── per_grid_metrics.csv
└── selected_grids.json
```

综合图左上角是逐网格误差热图，其余三个面板分别为误差最低、误差最高、
平均流量最高网格在第12预测步上的真实值与预测值。Milan 裁剪区域同时在
CSV/JSON中保存局部坐标和原始100×100网格坐标。

## 11. 相对第1预测步的误差增长曲线

此步骤直接读取已有 `metrics.json` 中的12步指标，不重新训练。脚本先在每个
种子内部计算：

```text
Delta MAE_h  = MAE_h  - MAE_1
Delta RMSE_h = RMSE_h - RMSE_1
Relative increase (%) = 100 * Delta metric_h / metric_1
```

然后再计算三个种子的均值和样本标准差，避免不同模型的初始误差干扰比较：

```bash
python -u ./baselines/plot_horizon_error_growth.py \
  --results-dir "$PWD/results_GTSRNet_baselines" \
  --datasets milan_sms milan_call milan_internet mobile_bj mobile_nj \
  --models seasonal_ha gru_seq2seq mtgnn patchtst dlinear timemixer GTSRNet \
  --seeds 16 26 36 \
  --formats png pdf
```

主要输出：

```text
results_GTSRNet_baselines/summary/horizon_error_growth/
├── horizon_error_growth_by_run.csv
├── horizon_error_growth_summary.csv
├── figures_absolute_mean_std/       # 论文正文推荐：绝对增量均值±标准差
├── figures_percent_mean_std/        # 相对增长百分比均值±标准差
└── figures_absolute_by_seed/        # seed 16/26/36分别绘图，不合并
```

## 12. 方法来源和表述边界

- MTGNN：项目 `utils/MTGNN.py` 的 PyTorch 移植，通过适配器统一输入输出；论文引用 Wu et al., KDD 2020。
- PatchTST：本文件夹实现了 channel-independent patching、Transformer encoder 和 RevIN 风格归一化的网格适配版；论文中应写“PatchTST adapted to grid-cell series”，不要声称直接运行了官方仓库的原始数据脚本。
- DLinear：实现趋势/季节分解和两个直接线性预测头；引用 Zeng et al., AAAI 2023。
- TimeMixer：复用项目中的 `utils/timemixer_model.py` 适配实现；引用 Wang et al., ICLR 2024。
- GTSRNet：复制自当前 `GTSRNet_model` 的完整模型，不读取旧 checkpoint，按统一协议重新训练。

请不要把原来 `scripts/train_timexer.py` 产生的结果与本实验混用，因为它的数据划分、归一化和设备处理与这里不同。
