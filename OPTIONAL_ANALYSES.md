# 可选统计分析与 SGA-Seq 基线

这些入口也来自同一原始代码归档，不需要额外下载另一个旧工程。
以下命令从项目根目录执行，需先准备真实数据并安装根目录参考依赖。
参数默认值和最终论文统计口径仍应与实际运行配置对齐。

## 1. 跨网格统计关联与增量预测检验

```bash
python -u dataset_interaction_analysis_server/analyze_dataset_interactions.py \
  --project-root . \
  --datasets milan_sms milan_call milan_internet mobile_bj mobile_nj \
  --output-dir results/dataset_interactions \
  --top-nodes 24 --max-lag 6 --granger-lags 6 \
  --permutations 500 --alpha 0.05
```

主要使用 CPU 的 NumPy/SciPy 运算。检验结果反映统计关联或增量预测依赖，
不能单凭此实验宣称因果关系成立。详细实现见该目录的源码及 README。

## 2. Milan 分时段相关性—距离图

```bash
python -u motivation_separate_panels/plot_phase_distance_separately.py \
  --h5 dataset/data_git_version.h5 --traffic all \
  --output-root results/motivation \
  --rows 40 60 --cols 40 60 \
  --input-length 24 --prediction-length 12 \
  --test-windows 168 --validation-ratio 0.10 --split-gap -1 \
  --aggregate-factor 1 --combined-only --dpi 400
```

此命令保留脚本原位置，避免将子目录内脚本误写为项目根目录文件。

## 3. SGA-Seq 适配基线

```bash
python -u sgaseq_10seed_server/train_sgaseq.py \
  --project-root . --smoke-test --device cpu

python -u sgaseq_10seed_server/run_sgaseq_10seed.py \
  --project-root . --gpus 0 \
  --datasets milan_sms milan_call milan_internet mobile_bj mobile_nj \
  --seeds 23 26 29 36 46 56 66 76 103 128 \
  --close-size 24 --predict-size 12 \
  --epochs 300 --batch-size 16 --learning-rate 0.001 --loss l1 \
  --split-mode chronological --split-gap -1 --milan-loader clean \
  --output-dir results/sgaseq
```

这是该目录已经实现的适配实验，而不是原作者原数据与原设置的原样复现。
输入/输出长度、图邻居裁剪等适配细节见 `sgaseq_10seed_server/README_CN.md`。
此处显式列出源码默认十种子；不把它们称为其他模型的默认种子。

## 4. 预测导出与结果可用性

预测导出和误差绘图已包含在 `full_V2_Metric/`、`full_v2_forecast/` 等保留目录。
使用 checkpoint 或预测数组的入口需要相应运行产物；本包不附带这些产物。
最终论文的小型指标及配置可单独提交到 `results_summary/`，但本次没有代填结果。
