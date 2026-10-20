# 审稿补实验包：并行解码、稳定性、嵌入消融与复杂度

本文件夹直接放在原项目根目录使用。它不会覆盖旧的 `results_decoder_only`，
新结果分别写入 `results_decoder_reviewer`、`results_embedding_reviewer` 和
`results_reviewer_profile`。

## 1. 这版修正了什么

- Milan 数据按真实正向时间构造窗口，不再使用旧读取器的逆序历史；
- 严格按时间划分训练/验证/测试，并在边界清除重叠预测目标；
- Min–Max 只在训练窗口的输入和目标上拟合，验证/测试不参与拟合；
- PMF/RMF 时间解码器直接接受预测损失，其他参数和训练设置保持一致；
- H=12 对齐论文主实验，H=24 作为长预测压力测试；
- 每个训练任务自动完成 0、0.01、0.03、0.05、0.10 的反馈噪声评估；
- PMF 也执行同样的噪声接口，作为不存在反馈路径的负对照；
- 噪声结果写入独立文件，不覆盖干净测试指标或 checkpoint；
- 嵌入消融严格为 No、Channel、Phase、Channel+Phase、Joint 五个条件；
- 自动输出多种子逐步误差图、train/val loss 曲线、噪声曲线和均值±标准差表；
- 提供统一 batch/GPU 下的参数量、分支参数量、峰值显存和延迟统计。

## 2. 放置位置

解压后应为：

```text
GLSTTN_tars-duobunewdata/
├── dataset/
├── utils/
├── reviewer_response_experiments_v3/
│   ├── README_CN.md
│   ├── decoder_model.py
│   ├── train_decoder_experiment.py
│   └── ...
└── ...
```

以下命令全部在 `GLSTTN_tars-duobunewdata` 项目根目录执行。

## 3. 环境和冒烟测试

```bash
conda activate glsttn

python -m py_compile reviewer_response_experiments_v3/*.py

python reviewer_response_experiments_v3/verify_decoder.py

python reviewer_response_experiments_v3/train_embedding_ablation.py \
  --project-root "$PWD" --smoke-test --device cuda --gpu 0
```

`verify_decoder.py` 必须出现五行 `PASS`；Embedding 冒烟测试必须出现
No、Channel、Phase、Channel+Phase、Joint 五行 `PASS`。

先只展开任务、不训练：

```bash
python reviewer_response_experiments_v3/run_decoder_experiments_4gpu.py \
  --project-root "$PWD" --gpus 0 1 2 3 --dry-run --no-plots

python reviewer_response_experiments_v3/run_embedding_ablation_4gpu.py \
  --project-root "$PWD" --gpus 0 1 2 3 --dry-run --no-plots
```

默认应分别显示 40 个解码任务和 75 个嵌入任务。

## 4. 正式运行解码机制实验（优先运行）

该命令包含：Call/Internet、H=12/24、PMF/RMF、seed 45–49，共40次训练。
每次训练完成后自动执行5档噪声评估，所以不需要再为噪声重新训练。

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
python -u reviewer_response_experiments_v3/run_decoder_experiments_4gpu.py \
  --project-root "$PWD" \
  --output-dir "$PWD/results_decoder_reviewer" \
  --datasets milan_call milan_internet \
  --decoders pmf rmf \
  --horizons 12 24 \
  --seeds 45 46 47 48 49 \
  --noise-levels 0 0.01 0.03 0.05 0.10 \
  --gpus 0 1 2 3 \
  --input-len 24 --seg-len 6 --d-model 128 \
  --epochs 300 --batch-size 32 --learning-rate 0.001 \
  --loss l1 --patience 40 --split-gap -1
```

后台运行：

```bash
nohup env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
python -u reviewer_response_experiments_v3/run_decoder_experiments_4gpu.py \
  --project-root "$PWD" \
  --output-dir "$PWD/results_decoder_reviewer" \
  --datasets milan_call milan_internet \
  --decoders pmf rmf --horizons 12 24 \
  --seeds 45 46 47 48 49 \
  --noise-levels 0 0.01 0.03 0.05 0.10 \
  --gpus 0 1 2 3 --input-len 24 --seg-len 6 \
  --epochs 300 --batch-size 32 --learning-rate 0.001 \
  --loss l1 --patience 40 --split-gap -1 \
  > decoder_reviewer_master.log 2>&1 &

tail -f decoder_reviewer_master.log
```

如果希望把 SMS 也加入机制实验，任务会从40增至60：

```bash
python -u reviewer_response_experiments_v3/run_decoder_experiments_4gpu.py \
  --project-root "$PWD" --output-dir "$PWD/results_decoder_reviewer" \
  --datasets milan_sms milan_call milan_internet \
  --decoders pmf rmf --horizons 12 24 \
  --seeds 45 46 47 48 49 --noise-levels 0 0.01 0.03 0.05 0.10 \
  --gpus 0 1 2 3 --epochs 300 --batch-size 32 --patience 40
```

### 4.1 增量补跑 SMS、MobileBJ 和 MobileNJ

若 Call 和 Internet 的40个任务已经完成，只需补跑下面60个任务。保持输出目录
为原来的 `results_decoder_reviewer`；调度器会跳过其中已经完整的任务，不会覆盖
旧结果。

```bash
nohup env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
python -u reviewer_response_experiments_v3/run_decoder_experiments_4gpu.py \
  --project-root "$PWD" \
  --output-dir "$PWD/results_decoder_reviewer" \
  --datasets milan_sms mobile_bj mobile_nj \
  --decoders pmf rmf --horizons 12 24 \
  --seeds 45 46 47 48 49 \
  --noise-levels 0 0.01 0.03 0.05 0.10 \
  --gpus 0 1 2 3 --input-len 24 --seg-len 6 --d-model 128 \
  --epochs 300 --batch-size 32 --learning-rate 0.001 \
  --loss l1 --patience 40 --split-gap -1 \
  > decoder_sms_bj_nj_master.log 2>&1 &

tail -f decoder_sms_bj_nj_master.log
```

当前 `MobileBJ.npy` 若为 `(168, 672, 1)`，表示168个时间点和28×24个网格。
在 H=24 且严格隔离重叠预测窗口时，可用训练窗口较少；应重点报告五种子的
离散程度，并把结果作为补充证据。不要因为某个种子或数据集不利而删除。

完成后重新汇总全部五个数据集：

```bash
python reviewer_response_experiments_v3/summarize_decoder_results.py \
  --results-dir "$PWD/results_decoder_reviewer" \
  --seeds 45 46 47 48 49

python reviewer_response_experiments_v3/audit_results.py \
  --decoder-results "$PWD/results_decoder_reviewer" \
  --decoder-datasets milan_sms milan_call milan_internet mobile_bj mobile_nj \
  --horizons 12 24 --decoder-seeds 45 46 47 48 49 \
  --skip-embedding
```

完整时应输出 `AUDIT PASS: decoder_runs=100, embedding_runs=0`。

已完成任务会自动跳过，因此可以重复执行命令续跑。只有确认要重训时才加
`--force`。

### 4.2 季节残差基线（`--baseline seasonal`）

背景：诊断发现，在 MobileNJ、MobileBJ 上，原来的 last-value 基线让 PMF 和 RMF
都不如 seasonal naive（直接复制 24 小时前的值）。新增的开关只改变**输出基线**，
PMF 和 RMF 同时生效：

| 参数 | 预测输出 |
|---|---|
| `--baseline last`（默认，原实现） | 残差 + 最后一个观测值 |
| `--baseline seasonal --seasonal-period 24` | 残差 + 24 小时前的值 x(t+h−24) |

保持不变的部分：编码器输入仍按最后值归一化；RMF 的反馈先还原到同一输入空间再送入
`valueEmbedding`；不新增任何参数，PMF/RMF 参数量相同。`last` 模式与旧实现数值完全
一致，旧 checkpoint 可以直接使用。当残差为零时，`seasonal` 模式的输出正好等于
seasonal naive，可以理解为模型以朴素基线为起点学习修正量。

**两种基线的结果必须放在不同目录。** 调度器发现目录里已有不同基线的结果时会直接
报错，汇总脚本发现混合基线时也会报错。

先检查结构：

```bash
python reviewer_response_experiments_v3/verify_decoder.py
```

应出现 `[last]` 和 `[seasonal]` 各 6 行 PASS，最后还有 2 行 PASS。

小规模试跑（4 个任务），确认模型能否超过 seasonal naive：

```bash
python -u reviewer_response_experiments_v3/run_decoder_experiments_4gpu.py \
  --project-root "$PWD" --output-dir "$PWD/results_decoder_seasonal_pilot" \
  --baseline seasonal --seasonal-period 24 \
  --datasets milan_sms mobile_nj --decoders pmf rmf --horizons 24 --seeds 45 \
  --gpus 0 1 2 3 --input-len 24 --seg-len 6 --d-model 128 \
  --epochs 300 --batch-size 32 --learning-rate 0.001 \
  --loss l1 --patience 40 --split-gap -1 --no-plots

python reviewer_response_experiments_v3/diagnose_horizon_curve.py \
  --project-root "$PWD" --dataset mobile_nj --horizon 24 \
  --baseline seasonal --results-dir "$PWD/results_decoder_seasonal_pilot" --seeds 45
```

试跑结果满意后再正式运行：

```bash
nohup env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
python -u reviewer_response_experiments_v3/run_decoder_experiments_4gpu.py \
  --project-root "$PWD" --output-dir "$PWD/results_decoder_seasonal" \
  --baseline seasonal --seasonal-period 24 \
  --datasets milan_sms milan_call milan_internet mobile_bj mobile_nj \
  --decoders pmf rmf --horizons 12 24 --seeds 45 46 47 48 49 \
  --noise-levels 0 0.01 0.03 0.05 0.10 \
  --gpus 0 1 2 3 --input-len 24 --seg-len 6 --d-model 128 \
  --epochs 300 --batch-size 32 --learning-rate 0.001 \
  --loss l1 --patience 40 --split-gap -1 \
  > decoder_seasonal_master.log 2>&1 &

python reviewer_response_experiments_v3/audit_results.py \
  --decoder-results "$PWD/results_decoder_seasonal" \
  --decoder-datasets milan_sms milan_call milan_internet mobile_bj mobile_nj \
  --horizons 12 24 --decoder-seeds 45 46 47 48 49 \
  --decoder-baseline seasonal --skip-embedding
```

论文中若采用该设置，需要写明输出基线的定义，并在逐步误差图中同时给出
seasonal naive 曲线。

## 5. 正式运行五种嵌入消融

默认对五个数据集、五种嵌入和三个种子运行，共75次训练，预测范围固定为
论文主设置 H=12。

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
python -u reviewer_response_experiments_v3/run_embedding_ablation_4gpu.py \
  --project-root "$PWD" \
  --output-dir "$PWD/results_embedding_reviewer" \
  --datasets milan_sms milan_call milan_internet mobile_bj mobile_nj \
  --embeddings none channel phase channel_phase joint \
  --seeds 16 26 36 \
  --gpus 0 1 2 3 \
  --input-len 24 --horizon 12 --seg-len 6 --d-model 128 \
  --epochs 300 --batch-size 16 --learning-rate 0.001 \
  --loss l1 --branch-aux-weight 0.1 --patience 0 --split-gap -1
```

后台运行时把上述 `python` 命令前加 `nohup env
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`，末尾加：

```bash
> embedding_reviewer_master.log 2>&1 &
tail -f embedding_reviewer_master.log
```

五种条件的严格含义：

| 参数 | 实际加入空间 token 的内容 |
|---|---|
| `none` | 不增加额外嵌入 |
| `channel` | 每个网格单元一个可学习向量 |
| `phase` | 目标起始小时的可学习相位向量 |
| `channel_phase` | Channel 与 Phase 两向量相加 |
| `joint` | 完整模型：Channel + Phase +“小时相位×网格单元”交互向量 |

这里的 Joint 对应 `full_v2` 原完整嵌入：Channel、Phase 与联合交互项同时使用。
其参数量天然较大，表格必须同时报告 `embedding_parameters`，不能只比较误差。

## 6. 手动重新汇总和画图

调度器在所有任务成功后会自动执行，下面命令仅用于重新绘图：

```bash
python reviewer_response_experiments_v3/summarize_decoder_results.py \
  --results-dir "$PWD/results_decoder_reviewer" \
  --seeds 45 46 47 48 49

python reviewer_response_experiments_v3/summarize_embedding_results.py \
  --results-dir "$PWD/results_embedding_reviewer" \
  --seeds 16 26 36
```

解码器汇总会生成：

- `decoder_summary.csv`：PMF/RMF 均值±标准差；
- `all_runs.csv`：每个种子的原始指标；
- `noise_all_runs.csv`：每个噪声强度、种子的原始指标；
- `horizon_*.png`：逐预测步误差，浅线为独立种子，粗线和阴影为均值±SD；
- `loss_*.png`：每个种子的 train/val loss；
- `noise_*.png`：噪声强度与输出变化/预测误差的关系。

## 7. 表示可视化

例如可视化 Milan Call、Joint embedding、seed 16：

```bash
python reviewer_response_experiments_v3/visualize_embeddings.py \
  --checkpoint "$PWD/results_embedding_reviewer/milan_call/joint/seed_16/best.pt" \
  --dataset milan_call \
  --output "$PWD/results_embedding_reviewer/summary/figures/representation_milan_call_joint_seed16.png"
```

Channel+Phase：

```bash
python reviewer_response_experiments_v3/visualize_embeddings.py \
  --checkpoint "$PWD/results_embedding_reviewer/milan_call/channel_phase/seed_16/best.pt" \
  --dataset milan_call \
  --output "$PWD/results_embedding_reviewer/summary/figures/representation_milan_call_channel_phase_seed16.png"
```

## 8. 参数量、显存与延迟

必须在服务器同一张空闲 GPU 上单独运行，不要与训练任务同时运行：

```bash
python -u reviewer_response_experiments_v3/profile_complexity.py \
  --project-root "$PWD" \
  --output "$PWD/results_reviewer_profile/complexity_profile.csv" \
  --gpu 0 --batch-size 16 --input-len 24 --horizons 12 24 \
  --height 20 --width 20 --d-model 128 --seg-len 6 \
  --warmup 20 --repeats 100
```

报告中必须同时写清：同一 GPU、batch=16、warmup=20、重复100次。不要把一次
batch 延迟写成单样本延迟。

## 9. 显存不足

在正式开始前把 batch 从32降到16即可。若已有部分 batch=32 结果，不要混合；
改用新目录，例如：

```bash
--batch-size 16 --output-dir "$PWD/results_decoder_reviewer_bs16"
```

## 10. 最终检查

解码器默认应有 40 个 `summary.json`、40 个 `noise_sweep.json`；嵌入默认应有
75 个 `metrics.json`。汇总脚本默认发现缺失结果时直接报错，只有临时查看未完成
结果时才使用 `--allow-partial`。

理论公式和回复信模板见 `THEORY_AND_RESPONSE_CN.md`。

全部实验结束后运行最终审计：

```bash
python reviewer_response_experiments_v3/audit_results.py \
  --decoder-results "$PWD/results_decoder_reviewer" \
  --embedding-results "$PWD/results_embedding_reviewer"
```

默认完整配置应输出：`AUDIT PASS: decoder_runs=40, embedding_runs=75`。
