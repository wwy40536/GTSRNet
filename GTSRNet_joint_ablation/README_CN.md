# w/o Joint 十种子独立实验

本目录只运行 `w/o Joint`。程序中的实际参数为：

```text
--embedding-variant channel_phase
```

它保留 Channel embedding 与 Phase embedding，只移除联合交互嵌入
`phase-by-cell interaction embedding`。其他网络结构和训练设置保持不变。

默认随机种子为：

```text
1 2 3 4 5 6 7 8 9 10
```

默认运行五个数据集，每个数据集10个种子，共50个训练任务，并由GPU
0、1、2、3并行调度。

## 正式运行

在原项目根目录执行：

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
python -u ./GTSRNet_joint_ablation/run_joint_comparison.py \
  --project-root "$PWD" \
  --output-dir "$PWD/results_wo_joint" \
  --datasets milan_sms milan_call milan_internet mobile_bj mobile_nj \
  --seeds 1 2 3 4 5 6 7 8 9 10 \
  --gpus 0 1 2 3 \
  --input-len 24 \
  --horizon 12 \
  --seg-len 6 \
  --d-model 128 \
  --epochs 300 \
  --batch-size 16 \
  --learning-rate 0.001 \
  --loss l1 \
  --branch-aux-weight 0.1 \
  --patience 0 \
  --split-gap -1
```

运行完成后，每个种子的结果位于：

```text
results_wo_joint/<dataset>/channel_phase/seed_<seed>/metrics.json
```

全部单种子结果会写入：

```text
results_wo_joint/summary/wo_joint_each_seed.csv
```

该CSV只列出每个种子的原始MAE和RMSE，不计算跨种子均值。

## 后台运行

```bash
nohup env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
python -u ./GTSRNet_joint_ablation/run_joint_comparison.py \
  --project-root "$PWD" \
  --output-dir "$PWD/results_wo_joint" \
  --datasets milan_sms milan_call milan_internet mobile_bj mobile_nj \
  --seeds 1 2 3 4 5 6 7 8 9 10 \
  --gpus 0 1 2 3 \
  --input-len 24 --horizon 12 --seg-len 6 --d-model 128 \
  --epochs 300 --batch-size 16 --learning-rate 0.001 \
  --loss l1 --branch-aux-weight 0.1 --patience 0 --split-gap -1 \
  > wo_joint_master.log 2>&1 &

tail -f wo_joint_master.log
```

已经存在完整 `metrics.json` 的任务会自动跳过，所以中断后可直接重复执行同一命令续跑。
不要加 `--force`，除非确实需要覆盖并重训已经完成的结果。

## 运行前检查

```bash
python -u ./GTSRNet_joint_ablation/run_joint_comparison.py \
  --project-root "$PWD" \
  --output-dir "$PWD/results_wo_joint" \
  --datasets milan_sms milan_call milan_internet mobile_bj mobile_nj \
  --seeds 1 2 3 4 5 6 7 8 9 10 \
  --gpus 0 1 2 3 --dry-run --skip-summary
```
