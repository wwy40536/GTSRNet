# 早间、夜间、晚间相关性图分别输出

把 `plot_phase_distance_separately.py` 上传到服务器项目根目录，然后执行：

```bash
conda activate glsttn
cd /path/to/GTSRNet

python -u motivation_analysis/plot_phase_distance_separately.py \
  --h5 ./dataset/data_git_version.h5 \
  --traffic all \
  --output-root ./outputs_v2_separate \
  --rows 40 60 \
  --cols 40 60 \
  --input-length 24 \
  --prediction-length 12 \
  --test-windows 168 \
  --validation-ratio 0.10 \
  --split-gap -1 \
  --aggregate-factor 1 \
  --ymin -0.7 \
  --ymax 1.2 \
  --combined-only \
  --dpi 400
```

`--traffic all` 会处理 SMS、Call 和 Internet。若只生成 SMS 的三幅图，改成：

```bash
--traffic sms
```

以 SMS 为例，组合图输出为：

```text
outputs_v2_separate/motivation_sms/correlation_vs_distance_centered.png
outputs_v2_separate/motivation_sms/correlation_vs_distance_centered.pdf
```

移除 `--combined-only` 后，还会同时输出：

```text
outputs_v2_separate/motivation_sms/separate_panels/
├── correlation_vs_distance_night.png
├── correlation_vs_distance_night.pdf
├── correlation_vs_distance_morning.png
├── correlation_vs_distance_morning.pdf
├── correlation_vs_distance_evening.png
└── correlation_vs_distance_evening.pdf
```

脚本保留原实验统计口径：只使用训练期原始数据，先对每个准确小时跨日期计算 Pearson 相关矩阵，再在夜间、早间和晚间内进行 Fisher-z 平均。
