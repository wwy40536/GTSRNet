# GTSRNet 分支可靠性实验

本文件夹直接读取已经训练完成的 `GTSRNet/best.pt`，不会重新训练模型，也不会修改原检查点。实验提取空间分支与时间分支经过预测步校准后、进入动态门控融合前的预测值，同时恢复门控分配给空间分支的权重 `alpha`。

## 一 放置位置

把整个 `GTSRNet_branch_analysis` 文件夹复制到服务器。推荐放在 GTSRNet 工程根目录下，但放在其他位置也可以。

工程需要保持以下原有文件：

```text
GLSTTN_tars-duobunewdata/
├── dataset/
├── utils/
├── GTSRNet_model/
│   ├── train_ablation.py
│   └── gtsrnet_model.py
├── results_ablation_strict_seed128_bs16/
│   ├── milan_call/GTSRNet/best.pt
│   ├── milan_sms/GTSRNet/best.pt
│   └── milan_internet/GTSRNet/best.pt
└── GTSRNet_branch_analysis/
```

如果结果目录中还包含 `seed_128/` 这一层，脚本也会自动搜索对应的 `best.pt`。

## 二 运行命令

进入工程根目录，并激活训练 GTSRNet 时使用的 Python 环境：

```bash
cd /你的路径/GLSTTN_tars-duobunewdata
conda activate 你的训练环境
```

推荐直接运行：

```bash
bash GTSRNet_branch_analysis/run_all.sh \
  "$PWD" \
  results_ablation_strict_seed128_bs16 \
  results_branch_reliability_seed128 \
  0
```

四个参数依次表示：工程根目录、已有消融结果目录、新实验输出目录和 GPU 编号。

也可以直接运行 Python：

```bash
python GTSRNet_branch_analysis/run_branch_reliability.py \
  --project-root "$PWD" \
  --results-root results_ablation_strict_seed128_bs16 \
  --output-dir results_branch_reliability_seed128 \
  --datasets milan_call milan_sms milan_internet \
  --device cuda \
  --gpu 0 \
  --save-predictions
```

如果显存不足，可增加 `--batch-size 4` 或 `--batch-size 8`。这只改变推理批量，不改变实验结果。

如果服务器上的 Milan 数据不在默认位置，可增加：

```bash
--milan-data /服务器路径/data_git_version.h5
```

## 三 输出内容

运行完成后生成：

```text
results_branch_reliability_seed128/
├── branch_reliability_overall.csv
├── fig_branch_error_all_datasets.png
├── fig_branch_reliability_all_datasets.png
├── run_manifest.json
├── milan_call/
│   ├── branch_reliability_by_horizon.csv
│   ├── branch_reliability_summary.json
│   ├── branch_error_by_horizon.png
│   ├── branch_reliability_by_horizon.png
│   └── branch_predictions.npz
├── milan_sms/
└── milan_internet/
```

其中：

- `branch_error_by_horizon.png`：空间分支、时间分支和最终融合结果的逐预测步 MAE/RMSE。
- `spatial_win_rate`：该预测步上空间分支绝对误差小于时间分支的样本—网格位置比例。
- `temporal_win_rate`：时间分支绝对误差更小的比例。
- `mean_alpha_spatial`：门控分配给空间分支的平均权重。
- `gate_winner_accuracy`：以 `alpha >= 0.5` 选择空间分支时，门控是否选择了实际误差更小分支的比例；误差相同的位置不计入。
- `alpha_advantage_correlation`：门控空间权重与“时间分支绝对误差减空间分支绝对误差”的相关系数。正值说明空间分支相对更可靠时，门控通常赋予它更高权重。
- `oracle_mae`：每个位置都选择误差更小分支时的理论 MAE，仅作为门控可改进空间的参考。

论文中建议主要使用 `fig_branch_error_all_datasets.png` 和 `fig_branch_reliability_all_datasets.png`，并在正文中报告 `branch_reliability_overall.csv` 的门控选择准确率及权重—相对误差相关性。

## 四 结果解释边界

两条分支误差曲线交叉或误差差距随预测步变化，只能证明不同预测步的相对优势不同，从而支持预测步感知融合。只有当 `gate_winner_accuracy` 高于随机水平且 `alpha_advantage_correlation` 为正时，才能进一步说明动态门控对分支可靠性具有一定识别能力。

本实验使用完整 `GTSRNet` 模型内部的两个分支输出，不使用 `wo_s` 或 `wo_t` 消融模型的结果。
