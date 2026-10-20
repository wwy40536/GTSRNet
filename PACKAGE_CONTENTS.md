# 本源码包的内容与使用范围

本包从作者提供的 `full_v2_forecast.zip` 整理，合并根目录说明和辅助工具。
它包含模型与实验源码，不附带真实数据、训练结果、预测数组或模型权重。
原归档未被修改；没有重新训练或补填实验数字。

## 核心代码

| 目录 | 用途 |
|---|---|
| `utils/` | 主模型和基线依赖的基础模块；保留原有支持文件 |
| `trainduobu1_ablation_v2/` | 完整 `full_v2` 与组件消融 |
| `full_v2_forecast/` | 六类基线、完整模型、逐步误差分析与绘图 |
| `add_3_recent_baselines/` | iTransformer、SegRNN、STID |
| `trainduobu1_ablation_fixedavg_multiseed/` | 固定平均对照 |
| `trainduobu1_gate_ablation/` | 动态门控、静态逐步权重及辅助损失对照 |
| `reviewer_response_experiments_v3/` | 独立时间预测器 PMF/RMF、嵌入对照、稳定性与复杂度 |
| `full_V2_Metric/` | 窗口/预测长度敏感性、时间相关性、指标和预测导出 |
| `full_v2_attention_experiment/` | 已训练 checkpoint 的注意力分析 |
| `full_v2_mechanism_error_maps/` | 机制对照与误差图 |
| `gstrnet_scalability/` | 共享模型的窗口化大网格实验 |

## 可选实验与统计分析

保留这些源码便于对应论文采用的补实验；包含某个实现不代表论文必须报告该实验。

| 目录 | 用途 |
|---|---|
| `gstrnet_wo_joint_10seed/` | 十种子联合嵌入对照 |
| `trainduobu1_lowrank_joint_experiment/` | 低秩联合嵌入对照 |
| `branch_reliability_experiment/` | 分支可靠性分析 |
| `dataset_interaction_analysis_server/` | 跨网格统计关联与增量预测检验 |
| `motivation_separate_panels/` | Milan 不同时段的相关性—距离图 |
| `sgaseq_10seed_server/` | SGA-Seq 的统一协议适配基线 |

前三项运行方式见 [README.md](README.md)，后三项见
[OPTIONAL_ANALYSES.md](OPTIONAL_ANALYSES.md)。

## 本包排除的内容

- 原始/加工 `.h5`、`.npy`、辅助 CSV，以及全部 `.npz` 预测或分析数组；
- 模型 checkpoint、训练日志、指标结果目录、结果图片和 PDF；
- `__pycache__/`、`.pyc`、`.bak` 和远程开发残留；
- 同类旧工程的重复拷贝、探索版本、旧杂项 `scripts/`；
- 审稿回复草稿、论文替换文本和内部实验计划。

数据目录只保留说明，真实数据需按 [data/README.md](data/README.md) 自行准备。
本次没有从其他本地工程混入同名代码，也没有将已有研究结果重新挑选或汇总。

## 未改变的代码行为与已知限制

模型和训练实现按原归档保留。少数文档及 shell 用法注释中的旧服务器路径
改成通用 `/path/to/GTSRNet` 示例；文件来源与变动记录见
`RELEASE_MANIFEST.json`。

`utils/` 中还包含历史支持模块；根依赖文件覆盖 README 中的主用实验入口，
不承诺历史模型的全部可选依赖都已安装。不要将任意旧工具视为论文主模型入口。

原代码有两项可选绘图缺件，本包没有用不同实验的同名脚本替代：

- 门控调度器缺少其可选绘图脚本，根 README 已显式使用 `--no-plot`；
- wo_joint 底层调度器缺少 `summarize_embedding_results.py`，应使用根 README
  推荐的十种子包装入口；它会自动关闭该可选调用。

原组件入口的有效辅助损失权重、`fixed_avg` 的校准差异等如实记录在根 README。
这些限制没有通过静默修改训练实现来掩盖。

## 上传方式

解压后，以内层 `GTSRNet/` 为仓库根目录，提交其中的源码和说明。
根目录应直接包含 `README.md`、`requirements.txt`、`.gitignore` 和 `utils/`。
无需把本压缩包本身再提交进仓库。

许可证状态见 [LICENSE_NOTICE.md](LICENSE_NOTICE.md)。目前仍需由著作权人
确认正式授权；整理源码包本身没有代替作者选择许可证。
