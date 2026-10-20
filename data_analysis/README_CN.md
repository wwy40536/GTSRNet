# 五个数据集的跨网格交互检验

这套程序用于回答：在去除各网格自身的24小时周期和长期趋势之后，一个网格的历史变化是否仍能显著关联或改善另一个网格的未来预测。

它证明的是**数据中的跨网格统计依赖/预测依赖**，不能写成物理因果关系。

## 方法

程序对每个数据集执行：

1. 按网格分别减去24个小时槽的平均流量，并去除线性趋势；
2. 预先按去周期后标准差选择24个活跃网格，避免事后根据相关性挑选；
3. 计算1～6小时有向滞后相关；
4. 通过500次独立循环位移置换建立零假设分布，置换过程保留单个网格的自相关结构；
5. 进行Granger式嵌套回归检验：比较“仅使用目标网格自身历史”和“同时加入源网格历史”；
6. 两组检验分别进行Benjamini–Hochberg FDR校正；
7. 只有同时通过置换检验和增量预测检验的边，才作为最可靠的跨网格预测依赖证据。

## 上传位置

将整个 `data_analysis` 文件夹放在原工程根目录：

```text
GLSTTN_tars-duobunewdata/
├── dataset/
│   ├── data_git_version.h5
│   ├── MobileBJ.npy
│   └── MobileNJ.npy
└── data_analysis/
```

## 安装依赖

原环境通常已经具备这些库；缺少时运行：

```bash
pip install -r data_analysis/requirements.txt
```

## 运行全部五个数据集

在工程根目录执行：

```bash
nohup bash data_analysis/run_all_datasets.sh \
  > dataset_interaction_analysis.log 2>&1 &
```

查看进度：

```bash
tail -f dataset_interaction_analysis.log
```

该程序主要使用CPU和NumPy/SciPy，不占用大量GPU显存。

## 快速试运行

先只分析Milan Call，并将置换次数减少为100：

```bash
python -u data_analysis/analyze_dataset_interactions.py \
  --project-root "$(pwd)" \
  --datasets milan_call \
  --output-dir results_dataset_interactions_smoke \
  --top-nodes 12 \
  --max-lag 3 \
  --granger-lags 3 \
  --permutations 100 \
  --alpha 0.05
```

正式论文结果仍建议使用默认的24个网格、6阶滞后和500次置换。

## 输出文件

每个数据集生成：

```text
results_dataset_interactions/<dataset>/
├── selected_grids.csv
├── directed_grid_pairs.csv
├── interaction_summary.json
├── fig_lagged_interaction_heatmap.png
├── fig_granger_interaction_heatmap.png
├── fig_period_correlation_matrices.png
└── fig_interaction_network.png
```

总表：

```text
results_dataset_interactions/interaction_summary_all_datasets.csv
```

重点查看：

- `significant_in_both_tests`：同时通过两类检验的有向网格边数；
- `both_tests_ratio`：可靠交互边占全部有向网格对的比例；
- `incremental_r2`：加入源网格历史后，目标网格预测解释度的增加；
- `granger_fdr_q` 和 `lag_fdr_q`：多重检验校正后的显著性。

## 论文表述

如果存在同时通过两项检验的非对角边，可以写：

> 在去除各网格自身的日周期模式与线性趋势后，仍有若干跨网格有向关系同时通过循环位移置换检验和增量预测检验，并在FDR校正后保持显著。这表明数据中存在无法由单个网格自身时间规律完全解释的跨网格预测依赖，为开展空间交互建模提供了数据依据。

不要写“证明了因果关系”，应使用“统计依赖”“预测依赖”或“交互证据”。
