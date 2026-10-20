# GTSRNet 机制级误差分析实验包

本实验包专门回答以下两个审稿问题：

1. 空间分支究竟在哪些网格有效：比较 `wo_s` 与 `GTSRNet` 的逐网格误差。
2. Phase embedding 究竟在哪些小时有效：比较 `no_periodic` 与 `original_phase` 的逐目标小时误差。

差值统一定义为：

```text
消融模型误差 - GTSRNet误差
```

所以差值为正表示组件有效，差值为负表示加入组件后性能退化。负值不能删除，它是论文需要讨论的失败模式。

## 设计说明

- `wo_s`：移除完整空间 Transformer 分支。
- `no_periodic`：仅移除学习式 hour embedding 和 hour-grid joint embedding，保留网格位置 embedding。因此它不会把“周期贡献”和“空间位置贡献”混在一起。
- 默认空间实验覆盖 Milan SMS、Call、Internet、MobileBJ、MobileNJ。
- 默认逐小时周期实验只覆盖 Milan 三个数据集，因为 MobileBJ/NJ 的 `.npy` 文件没有可靠时间戳。不能把数组下标伪装成真实小时来回应审稿人。
- 默认使用三个独立种子 `16 26 36`。
- 每个种子单独产生图片；不会只输出三种子合并图。
- 分析前自动检查两个模型的测试真值、目标小时、形状和关键实验配置是否一致。不一致时直接报错，不会生成不可比图。

默认调度39个任务：空间对照 `5×3×2=30` 个，周期消融新增 `3×3=9` 个。Milan的 `GTSRNet` 同时作为两项实验的共同基准，不重复训练。如果项目根目录存在此前生成的 `results_GTSRNet_baselines`，脚本会先严格核对配置，然后复用其中15个 `GTSRNet` 检查点，只补一次测试推理；此时真正重新训练的是15个 `wo_s` 和9个 `no_periodic`。

## 放置位置

将压缩包解压到原项目根目录。解压后应为：

```text
GLSTTN_tars-duobunewdata/
├── dataset/
├── utils/
└── GTSRNet_error_analysis/
    ├── mechanism_model.py
    ├── train_mechanism_pair.py
    ├── run_mechanism_experiments.py
    └── analyze_mechanism_errors.py
```

下面所有命令都在 `GLSTTN_tars-duobunewdata` 项目根目录执行。

## 服务器运行命令

### 1. 激活原实验环境

```bash
conda activate glsttn
```

如果服务器环境名不是 `glsttn`，换成训练原论文模型时使用的环境。

### 2. 先做模型冒烟测试

```bash
python -u GTSRNet_error_analysis/train_mechanism_pair.py \
  --project-root . --smoke-test --device cuda --gpu 0 \
  --phase-mode original_phase

python -u GTSRNet_error_analysis/train_mechanism_pair.py \
  --project-root . --smoke-test --device cuda --gpu 0 \
  --phase-mode no_periodic
```

应看到所有模型输出 `PASS`。PyTorch Transformer 的 nested-tensor `UserWarning` 不影响运行。

再检查画图和CSV导出（不使用真实数据和GPU）：

```bash
python -u GTSRNet_error_analysis/smoke_test_analysis.py
```

### 3. 检查任务列表，不实际训练

```bash
python -u GTSRNet_error_analysis/run_mechanism_experiments.py \
  --project-root . --python "$(which python)" \
  --gpus 0 1 2 3 --seeds 16 26 36 --dry-run
```

### 4. 正式后台运行（推荐完整范围）

```bash
nohup python -u GTSRNet_error_analysis/run_mechanism_experiments.py \
  --project-root . --python "$(which python)" \
  --gpus 0 1 2 3 --seeds 16 26 36 \
  --epochs 300 --batch-size 16 --learning-rate 0.001 \
  --loss l1 --branch-aux-weight 0.1 \
  --split-mode chronological --split-gap -1 --milan-loader clean \
  --baseline-results results_GTSRNet_baselines \
  --results-dir results_mechanism_error_maps \
  > mechanism_error_maps.log 2>&1 &
```

查看进度：

```bash
tail -f mechanism_error_maps.log
```

查看显卡：

```bash
watch -n 2 nvidia-smi
```

### 5. 如果只想先跑 Milan 三个数据集

```bash
nohup python -u GTSRNet_error_analysis/run_mechanism_experiments.py \
  --project-root . --python "$(which python)" \
  --gpus 0 1 2 3 --seeds 16 26 36 \
  --spatial-datasets milan_sms milan_call milan_internet \
  --phase-datasets milan_sms milan_call milan_internet \
  --epochs 300 --batch-size 16 --learning-rate 0.001 \
  --loss l1 --branch-aux-weight 0.1 \
  --split-mode chronological --split-gap -1 --milan-loader clean \
  --baseline-results results_GTSRNet_baselines \
  --results-dir results_mechanism_error_maps_milan \
  > mechanism_milan.log 2>&1 &
```

## 中断后继续

直接重新执行同一条命令即可：

- 已有 `predictions.npz` 的任务自动跳过；
- 已有 `best.pt` 但没有预测文件的任务自动进入 `--eval-only`，只补测试推理；
- 其余任务正常训练。

只有确实需要覆盖已完成任务时才添加 `--force`。

## 训练结束后的自动输出

默认结果目录：

```text
results_mechanism_error_maps/
├── original_phase/seed_16/...       # GTSRNet和wo_s
├── original_phase/seed_26/...
├── original_phase/seed_36/...
├── no_periodic/seed_16/...          # 周期消融
├── logs/
└── analysis/
    ├── per_seed/
    │   ├── seed_16/<dataset>/
    │   │   ├── spatial_grid_difference.png/.pdf
    │   │   ├── spatial_grid_differences.csv
    │   │   ├── spatial_top_bottom_grids.csv
    │   │   ├── spatial_by_traffic_quartile.csv
    │   │   ├── phase_hour_difference.png/.pdf   # Milan才有
    │   │   └── phase_hour_differences.csv       # Milan才有
    │   ├── seed_26/<dataset>/
    │   └── seed_36/<dataset>/
    ├── spatial_summary_per_seed.csv
    ├── phase_summary_per_seed.csv
    ├── phase_hour_all_seeds.csv
    ├── analysis_manifest.json
    └── MANUSCRIPT_RESULTS_TEMPLATE_CN.md
```

三张种子图分别在 `seed_16`、`seed_26`、`seed_36`，没有被合并。PDF适合论文排版，PNG适合快速查看。

## 单独重新分析

分析本身不使用GPU，也不重新训练：

```bash
python -u GTSRNet_error_analysis/analyze_mechanism_errors.py \
  --results-dir results_mechanism_error_maps \
  --spatial-datasets milan_sms milan_call milan_internet mobile_bj mobile_nj \
  --phase-datasets milan_sms milan_call milan_internet \
  --seeds 16 26 36
```

## 论文使用建议

- 正文可放一个代表性空间差值图和一个逐小时差值图。
- 三个种子、全部数据集的独立图片放补充材料。
- 正文同时报告总体误差降低率、改善网格占比以及改善小时数。
- 若某些数据集、网格或小时差值为负，明确说明组件收益具有数据和时间条件依赖性。

运行完成后，把整个 `results_mechanism_error_maps` 文件夹压缩后发回即可继续整理最终论文图、结果表和审稿回复。
