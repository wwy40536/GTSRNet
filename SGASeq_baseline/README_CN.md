# SGA-Seq：五数据集统一协议、10 个 seed

本文件夹用于在现有 GTSRNet 工程中补跑 SGA-Seq。它复用你此前基线实验的数据读取、时间划分、训练集归一化和评价代码，输出可直接合并到表 9 的 `Step 4`、`Step 8`、`Step 12` 和 `Overall` 指标。

## 1. 实现说明

- SGA-Seq 主体依据作者公开仓库：<https://github.com/OvOYu/SGA-Seq>。
- 保留站点嵌入、自适应图卷积、图注意力、SGA-GRU 门控、逐层残差修正和各层预测相加。
- 原作者代码默认输入长度和输出长度相同（12→12）。本实验按论文统一协议改为 24→12，因此残差通路改成“在24步隐藏序列上逐层修正、每层直接输出未来12步”。
- 原作者的全节点稠密注意力在 400/560 个节点上代价很高。本实现从学习到的自适应图中保留每个节点最强的 8 个连接；可用 `--graph-neighbors` 修改。该设置必须在论文实现细节中说明。
- 这不是重新训练作者原始数据的复现实验，而是 SGA-Seq 在你五个任务统一协议下的适配实验。

## 2. 文件放置

把整个 `SGASeq_baseline` 文件夹上传到原工程根目录。上传后结构应类似：

```text
GLSTTN_tars-duobunewdata/
├── dataset/
│   ├── data_git_version.h5
│   ├── MobileBJ.npy
│   └── MobileNJ.npy
├── SGASeq_baseline/
│   ├── sgaseq_model.py
│   ├── training_core.py
│   ├── train_sgaseq.py
│   ├── run_sgaseq.py
│   └── summarize_sgaseq.py
└── ...
```

## 3. 固定的10个 seed

```text
23, 26, 29, 36, 46, 56, 66, 76, 103, 128
```

其中包含你原基线常用的 `26/36/66/76` 和 GTSRNet 常用的 `23/29/103/128`。默认对五个数据集分别运行这10个 seed，共 50 个训练任务。

## 4. 先做模型自检

在工程根目录执行：

```bash
python -u SGASeq_baseline/train_sgaseq.py \
  --project-root "$(pwd)" \
  --smoke-test \
  --device cuda \
  --gpu 0
```

看到 `PASS SGA-Seq` 后再正式运行。

## 5. 四张 GPU 正式运行

```bash
nohup python -u SGASeq_baseline/run_sgaseq.py \
  --project-root "$(pwd)" \
  --python "$(which python)" \
  --gpus 0 1 2 3 \
  --datasets milan_sms milan_call milan_internet mobile_bj mobile_nj \
  --seeds 23 26 29 36 46 56 66 76 103 128 \
  --close-size 24 \
  --predict-size 12 \
  --batch-size 16 \
  --epochs 300 \
  --learning-rate 0.001 \
  --loss l1 \
  --split-mode chronological \
  --split-gap -1 \
  --milan-loader clean \
  --save-predictions \
  > sgaseq_master.log 2>&1 &
```

查看进度：

```bash
tail -f sgaseq_master.log
```

只有一张 GPU 时，把 `--gpus 0 1 2 3` 改成 `--gpus 0`。显存不足时把 `--batch-size 16` 改成 `8`，但论文计算效率表中的时间必须注明实际 batch size。

脚本支持断点续跑：已有 `metrics.json` 的任务会自动跳过。不要使用 `--force`，除非明确要覆盖已完成实验。

## 6. 输出内容

每次运行保存到：

```text
results_sgaseq/<dataset>/sgaseq/seed_<seed>/
├── best.pt
├── history.csv
├── metrics.json
└── predictions.npz
```

- `metrics.json`：Overall、每个预测步的 RMSE/MAE、参数量、最佳 epoch、训练及推理时间。
- `history.csv`：每轮训练损失、验证损失和耗时，可画收敛曲线。
- `predictions.npz`：测试集预测和真实值，可继续画 Taylor 图与小提琴图。

全部任务结束后自动生成：

```text
results_sgaseq/summary/
├── all_runs.csv
├── step4_step8_step12_overall.csv
├── paper_table_sgaseq.csv
└── efficiency_by_dataset.csv
```

如部分任务未完成，也可手动汇总现有结果：

```bash
python -u SGASeq_baseline/summarize_sgaseq.py \
  --results-dir results_sgaseq \
  --datasets milan_sms milan_call milan_internet mobile_bj mobile_nj \
  --seeds 23 26 29 36 46 56 66 76 103 128 \
  --allow-partial
```

