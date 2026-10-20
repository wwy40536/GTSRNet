# 新增 iTransformer、SegRNN、STID 基线（4 seeds）

本目录只新增三个基线，不会覆盖旧基线。实验结束后，`merge_baseline_results.py` 会同时读取旧结果、新结果和 GTSRNet 的四个结果目录，生成统一表格与逐预测步误差增量图。

## 1. 放置位置

把 `recent_baselines_4seed.zip` 上传到项目根目录并解压：

```bash
cd /path/to/GTSRNet
mkdir -p recent_baselines
unzip -o recent_baselines_4seed.zip -d recent_baselines
```

解压后结构为：

```text
/path/to/GTSRNet/
├── dataset/
├── GTSRNet_baseline_benchmark/
│   └── training_core.py
└── recent_baselines/
```

新代码复用原基线的数据切分、训练集归一化和反归一化评估逻辑。默认协议为：输入 24、预测 12、batch size 16、300 epochs、L1、chronological split、purge gap 11。

## 2. 先做冒烟测试

```bash
cd /path/to/GTSRNet
conda activate glsttn

python -u recent_baselines/train_recent_baseline.py \
  --project-root "$PWD" \
  --model all \
  --smoke-test \
  --device cuda \
  --gpu 0
```

应看到 iTransformer、SegRNN、STID 三行 `PASS`。

## 3. 四张 GPU 跑 60 个任务

这里默认采用 `23 29 103 128`，与 GTSRNet 的四个独立运行一致。总任务数为 3 个模型 × 5 个数据集 × 4 个 seeds = 60。

```bash
mkdir -p results_recent_baselines_4seed

nohup python -u recent_baselines/run_recent_baselines.py \
  --project-root "$PWD" \
  --python /home/wky/anaconda3/envs/glsttn/bin/python \
  --output-dir results_recent_baselines_4seed \
  --models itransformer segrnn stid \
  --datasets milan_sms milan_call milan_internet mobile_bj mobile_nj \
  --seeds 23 29 103 128 \
  --gpus 0 1 2 3 \
  --close-size 24 \
  --predict-size 12 \
  --epochs 300 \
  --batch-size 16 \
  --learning-rate 0.001 \
  --loss l1 \
  --split-mode chronological \
  --split-gap -1 \
  --d-model 128 \
  --e-layers 2 \
  --n-heads 4 \
  --d-ff 256 \
  --seg-len 6 \
  > results_recent_baselines_4seed/launcher.log 2>&1 &

echo $!
```

查看进度：

```bash
tail -f results_recent_baselines_4seed/launcher.log
```

统计完成数（最终应为 60）：

```bash
find results_recent_baselines_4seed -name metrics.json | wc -l
```

脚本支持断点续跑：再次执行相同命令时，已有 `metrics.json` 的任务会自动跳过。不要加 `--force`，除非确实要覆盖已有结果。

## 4. 合并旧基线、三个新基线和 GTSRNet

以下命令读取旧基线的 `26 36 66 76`，读取新增基线与 GTSRNet 的 `23 29 103 128`。把最后四个目录替换成服务器上的实际路径；若 BJ/NJ 的 GTSRNet 结果在其他目录，也继续追加在 `--gtsrnet-results` 后面。

```bash
python -u recent_baselines/merge_baseline_results.py \
  --old-results results_GTSRNet_baselines \
  --new-results results_recent_baselines_4seed \
  --old-seeds 26 36 66 76 \
  --new-seeds 23 29 103 128 \
  --gtsrnet-results \
    results_ablation_strict_seed23_bs16 \
    results_ablation_strict_seed29_bs16 \
    results_ablation_strict_seed103_bs16 \
    results_ablation_strict_seed128_bs16 \
  --output-dir results_baselines_merged
```

主要输出：

- `paper_table_mean.csv`：表格直接使用的 RMSE / MAE 四种子均值；
- `paper_table_mean_std.csv`：均值 ± 跨种子标准差；
- `baseline_summary.csv`：长表及每种模型实际使用的 seeds；
- `per_horizon_delta.csv`：每一步的 ΔRMSE、ΔMAE 均值与标准差；
- `per_horizon_absolute.csv`：每个模型、数据集和预测步的绝对 RMSE/MAE；
- `paper_table_steps_4_8_12_overall.csv`：可直接填写论文表格的 Step 4、Step 8、Step 12 和 Overall；
- `figures/fig_delta_rmse_*.png/.pdf`：五个任务各自的 RMSE 增量图；
- `figures/fig_delta_mae_*.png/.pdf`：五个任务各自的 MAE 增量图；
- `gtsrnet_vs_best_baseline.csv`：每个数据集相对最优基线的降低比例；
- `average_improvement.txt`：五项任务平均降低比例。
- `efficiency_table_milan.csv`：表11可直接粘贴的训练时间、测试推理时间和参数量；训练时间按每个 run 的 epoch 均值统计，再对 Milan 三项任务及所选 seeds 报告均值 ± 标准差。

增量图不需要再训练一次“差值实验”。训练产生的 `metrics.json` 已保存每个预测步的绝对 RMSE/MAE，合并脚本直接按每个 seed 先计算：

```text
ΔMAE_h = MAE_h - MAE_1
ΔRMSE_h = RMSE_h - RMSE_1
```

然后再计算四个 seeds 的均值和标准差。

## 5. 完整性检查

合并脚本默认严格检查五个数据集是否都具备各模型结果。如果只是先检查部分任务，可临时加 `--allow-partial`；最终制表时应去掉该选项。

模型依据：

- iTransformer: <https://github.com/thuml/iTransformer>
- SegRNN: <https://github.com/lss-1138/SegRNN>
- STID: <https://github.com/GestaltCogTeam/STID>
