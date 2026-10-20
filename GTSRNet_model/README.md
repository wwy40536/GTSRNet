# GTSRNet 模型与组件消融

本目录提供 GTSRNet 的模型定义、训练入口和组件消融。
完整模型的公开标识为 `GTSRNet`，类定义位于 `gtsrnet_model.py`。
数据准备、依赖、正式训练命令和种子设置统一见[根目录说明](../README.md)。
门控和辅助损失的实现见[模型说明](MODEL_DETAILS.md)。

## 模型标识

| 标识 | 含义 |
|---|---|
| `GTSRNet` | 完整模型：空间与时间分支、逐预测步校准、动态门控，训练时使用配置的分支辅助损失 |
| `full` | 保留的未校准动态门控对照，不是完整模型的别名 |
| `wo_g` | 以 Concat + Linear 替换门控 |
| `wo_seg` | 时间分支使用 `seg_len=1` |
| `wo_sthe` | 移除空间分支的 channel/hour/joint 嵌入 |
| `wo_s` | 仅保留时间分支 |
| `wo_t` | 仅保留空间分支 |

这些变体保留所提供源码的实际实现。不能仅凭名称把所有变体解释成
“从完整模型中只删除一个模块”；比较时应说明门控结构与辅助损失的配置差异。
固定平均融合和更细的门控对照分别在 `GTSRNet_fusion_ablation/` 与
`GTSRNet_gate_ablation/`，运行入口见根 README。

为保持已有配置，底层构造器和单任务训练器仍保留历史 `full` 默认值。
使用完整模型时必须显式指定 `variant="GTSRNet"`、`--variant GTSRNet`
或调度器参数 `--variants GTSRNet`，不可省略后将默认结果当作完整模型。

## 文件

- `gtsrnet_model.py`：模型与变体定义。
- `train_ablation.py`：单任务训练、评估和合成输入检查。
- `run_ablation.py`：多数据集/变体任务调度。
- `plot_ablation.py`：从真实指标文件绘图。

在项目根目录执行合成输入检查：

```bash
python GTSRNet_model/train_ablation.py \
  --project-root . --smoke-test --device cpu
```

该检查遍历所支持的变体，不读取正式数据，也不构成实验结果复现。
正式实验请使用根 README 中显式选择 `GTSRNet` 的命令；各次运行使用
独立输出目录，保留实际配置、种子、切分和评估指标。

本次仅统一公开名称、路径和展示标签，没有改动数值模型、训练预算或实验结果。
历史结果目录和 checkpoint 元信息不会自动迁移，不能把新默认路径直接指向旧结果。
