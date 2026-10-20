# full_v2 严格消融：seed 23/29/31 + Fixed Average

本文件夹用于补跑 `23/29/31` 三个随机种子。不要把旧的 seed 2 指标混入本次
均值与标准差，因为本版本修正了旧代码中 `wo_seg/wo_sthe` 同时切换门控、以及只有
`full_v2` 使用分支辅助损失的问题。

## 严格对照组

```text
full_v2    完整模型：预测步校准动态门控
fixed_avg  两分支固定 0.5/0.5 平均，无融合参数
wo_g       用 Concat + Linear 替换主门控
wo_seg     仅令 seg_len=1，保留主门控和其他设置
wo_sthe    仅删除 channel/phase/joint embedding，保留主门控
wo_s       仅保留时间分支
wo_t       仅保留空间分支
```

所有具有两个分支的变体统一使用 `branch_aux_weight=0.1`。`wo_s` 和 `wo_t` 只有一个
分支，因此不使用分支辅助损失。

## 放置位置

将整个文件夹上传到服务器项目根目录：

```text
GLSTTN_tars-duobunewdata/
├── dataset/
├── utils/
├── scripts/
└── trainduobu1_ablation_fixedavg_multiseed/
```

以下命令均在 `GLSTTN_tars-duobunewdata` 根目录运行。

## 1. 冒烟测试

```bash
conda activate glsttn

python -u ./trainduobu1_ablation_fixedavg_multiseed/train_trainduobu1_v2_ablation.py \
  --project-root "$PWD" \
  --smoke-test --device cuda --gpu 0
```

必须看到包括 `fixed_avg` 在内的所有变体均输出 `PASS`。

## 2. 四卡依次运行三个种子（Milan 三个数据集）

```bash
for seed in 23 29 31; do
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  python -u ./trainduobu1_ablation_fixedavg_multiseed/run_trainduobu1_v2_ablation_4gpu.py \
    --project-root "$PWD" \
    --output-dir "$PWD/results_ablation_strict_seed${seed}_bs16" \
    --datasets milan_sms milan_call milan_internet \
    --variants full_v2 fixed_avg wo_g wo_seg wo_sthe wo_s wo_t \
    --gpus 0 1 2 3 \
    --batch-size 16 --epochs 300 \
    --learning-rate 0.001 --loss l1 \
    --branch-aux-weight 0.1 \
    --e-layers 2 --seg-len 6 \
    --seed "$seed" \
    --split-mode chronological --split-gap -1 \
    --milan-loader clean --patience 0 || exit 1
done
```

每个种子运行 `3 × 7 = 21` 个任务，总计 63 个任务。同一时刻每张 GPU 只运行一个
任务。脚本检测到已有 `metrics.json` 时会自动跳过，因此中断后直接重复同一命令即可。
不要添加 `--force`，除非确实需要覆盖完整结果。

## 3. 汇总三个种子

```bash
python -u ./trainduobu1_ablation_fixedavg_multiseed/summarize_multiseed.py \
  --result-dirs \
    "$PWD/results_ablation_strict_seed23_bs16" \
    "$PWD/results_ablation_strict_seed29_bs16" \
    "$PWD/results_ablation_strict_seed31_bs16" \
  --output-dir "$PWD/results_ablation_strict_multiseed_23_29_31"
```

汇总输出：

```text
results_ablation_strict_multiseed_23_29_31/
├── all_runs.csv
├── summary_mean_std.csv
└── fig_ablation_multiseed.png
```

## 4. 如需同时跑 MobileBJ/MobileNJ

将正式运行命令中的数据集改成：

```text
--datasets milan_sms milan_call milan_internet mobile_bj mobile_nj
```

对应地，三个种子总任务数将变为 `5 × 7 × 3 = 105`。
