# 低秩网格—周期联合表示实验包

本实验包基于已经跑通的 `full_v2` 训练协议，只比较联合表示的参数化方式：

| variant | 含义 |
|---|---|
| `full_v2` | 原始稠密联合查找表，参数量为 `P*N*D` |
| `lowrank_joint` | 低秩联合表示，参数量为 `N*r + P*r + r*D + 1` |
| `wo_joint` | 完全删除联合表示，但保留网格嵌入和周期嵌入 |

低秩形式为：

```text
J(i,p) = scale * W(A_cell[i] * B_phase[p])
```

其中 `*` 是逐元素乘法，默认秩 `r=16`。三个版本的其余网络结构、损失函数、数据划分和训练设置完全一致。

## 放置位置

解压后，把整个 `trainduobu1_lowrank_joint_experiment` 文件夹放在原项目根目录。应满足：

```text
GLSTTN_tars-duobunewdata/
├── dataset/
├── utils/
└── trainduobu1_lowrank_joint_experiment/
```

进入原项目根目录：

```bash
cd /你的路径/GLSTTN_tars-duobunewdata
conda activate glsttn
```

## 1. 必做：冒烟测试

此命令不读取数据、不训练，只检查三种模型的前向传播、反向传播、输出形状和参数量：

```bash
python -u ./trainduobu1_lowrank_joint_experiment/train_lowrank_joint.py \
  --project-root "$PWD" \
  --device cuda \
  --gpu 0 \
  --joint-rank 16 \
  --smoke-test
```

应看到三行 `PASS`：`full_v2`、`lowrank_joint`、`wo_joint`。

## 2. 推荐：只补跑低秩版本（15组）

原来的 `full_v2` 和 `wo_joint` 结果可以继续使用。只补跑5个数据集、3个种子的 `lowrank_joint`：

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
python -u ./trainduobu1_lowrank_joint_experiment/run_lowrank_joint_multiseed_4gpu.py \
  --project-root "$PWD" \
  --output-dir "$PWD/results_lowrank_joint_r16_seed16_26_36_bs16" \
  --datasets milan_sms milan_call milan_internet mobile_nj mobile_bj \
  --variants lowrank_joint \
  --seeds 16 26 36 \
  --gpus 0 1 2 3 \
  --joint-rank 16 \
  --batch-size 16 \
  --epochs 300 \
  --learning-rate 0.001 \
  --loss l1 \
  --branch-aux-weight 0.1 \
  --split-mode chronological \
  --split-gap -1 \
  --milan-loader clean \
  --e-layers 2 \
  --patience 0 \
  --num-workers 0 \
  --log-every 1
```

默认支持断点续跑：已经存在 `metrics.json` 的任务会自动跳过。不要加 `--force`，除非明确需要覆盖并重训已经完成的任务。

## 3. 最严格：三种表示全部重跑（45组）

如果希望生成完全独立、可直接放进论文的三模型对照结果：

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
python -u ./trainduobu1_lowrank_joint_experiment/run_lowrank_joint_multiseed_4gpu.py \
  --project-root "$PWD" \
  --output-dir "$PWD/results_joint_encoding_r16_seed16_26_36_bs16" \
  --datasets milan_sms milan_call milan_internet mobile_nj mobile_bj \
  --variants full_v2 lowrank_joint wo_joint \
  --seeds 16 26 36 \
  --gpus 0 1 2 3 \
  --joint-rank 16 \
  --batch-size 16 \
  --epochs 300 \
  --learning-rate 0.001 \
  --loss l1 \
  --branch-aux-weight 0.1 \
  --split-mode chronological \
  --split-gap -1 \
  --milan-loader clean \
  --e-layers 2 \
  --patience 0 \
  --num-workers 0 \
  --log-every 1
```

完成后会自动生成：

```text
summary_multiseed.csv
figures/fig_joint_encoding_*.png
logs/seed_*/dataset/variant.log
seed_*/dataset/variant/metrics.json
seed_*/dataset/variant/best.pt
```

## 4. 后台运行

把上述完整命令保存为 `run_lowrank.sh` 后：

```bash
chmod +x run_lowrank.sh
nohup bash run_lowrank.sh > lowrank_joint_master.log 2>&1 &
echo $!
tail -f lowrank_joint_master.log
```

也可以直接对运行命令使用 `nohup`，但脚本方式更容易检查和复现。

## 5. 单GPU示例

```bash
python -u ./trainduobu1_lowrank_joint_experiment/train_lowrank_joint.py \
  --project-root "$PWD" \
  --dataset milan_sms \
  --variant lowrank_joint \
  --output-dir "$PWD/results_lowrank_single_seed" \
  --device cuda \
  --gpu 0 \
  --joint-rank 16 \
  --seed 16 \
  --batch-size 16 \
  --epochs 300 \
  --learning-rate 0.001 \
  --loss l1 \
  --branch-aux-weight 0.1 \
  --split-mode chronological \
  --split-gap -1 \
  --milan-loader clean \
  --e-layers 2 \
  --patience 0
```

## 6. 秩敏感性实验

不要在测试集上选择秩。可以先分别设置 `--joint-rank 8`、`16`、`32`，使用验证损失确定一个固定秩；每个秩必须使用不同的 `--output-dir`，避免不同秩的结果互相跳过或覆盖。

确定秩后，再用固定秩运行三个随机种子的正式测试实验。

## 7. 如何判断结果

重点比较：

1. `lowrank_joint` 是否优于原始 `full_v2`：检验低秩约束是否解决大查找表的过拟合。
2. `lowrank_joint` 是否优于 `wo_joint`：检验网格—周期交互本身是否真正有效。
3. 不只看均值，还应看三个种子的标准差和逐种子胜负。

如果 `lowrank_joint` 优于 `full_v2` 但仍不如 `wo_joint`，应选择更简单的 `wo_joint`，不能只因为低秩形式更有新意就把它作为最终模型。
