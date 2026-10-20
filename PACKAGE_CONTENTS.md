# GTSRNet 源码包

本包从作者提供的源码归档整理，统一使用 GTSRNet 和实验用途命名。
不包含真实数据、预测数组、checkpoint、训练结果、缓存或审稿回复草稿。

## 源码目录

| 目录 |
|---|
| `utils/` |
| `GTSRNet_model/` |
| `baselines/` |
| `recent_baselines/` |
| `GTSRNet_fusion_ablation/` |
| `GTSRNet_gate_ablation/` |
| `decoder_comparison/` |
| `GTSRNet_sensitivity/` |
| `GTSRNet_attention/` |
| `GTSRNet_error_analysis/` |
| `GTSRNet_scalability/` |
| `GTSRNet_joint_ablation/` |
| `GTSRNet_lowrank_ablation/` |
| `GTSRNet_branch_analysis/` |
| `data_analysis/` |
| `motivation_analysis/` |
| `SGASeq_baseline/` |

完整模型入口在 `GTSRNet_model/`，主要类名为 `GTSRNet`；
运行命令见 [README.md](README.md)，可选统计与基线见
[OPTIONAL_ANALYSES.md](OPTIONAL_ANALYSES.md)。

随机种子、重复次数、GPU列表、训练预算和模型计算均未因命名整理而改变。
已有历史输出的目录和元信息不会自动迁移；保留原记录，不覆盖实验数字。

## 已知源代码限制

门控调度器缺少其可选绘图脚本，根 README 已使用 `--no-plot`。
联合嵌入底层调度器缺少其可选汇总脚本，根 README 推荐的包装入口
会关闭该调用。本版没有用不同实验的同名文件替代这些缺件。
`utils/` 中保留了历史支持模块，根依赖仅覆盖文档指定的主用入口。

## 数据、环境与授权

实际数据仍放 `dataset/`；`data/` 保存说明，见
[data/README.md](data/README.md)。参考依赖不冒充原服务器版本；
请用环境导出工具记录真实安装环境。许可证状态见
[LICENSE_NOTICE.md](LICENSE_NOTICE.md)。

## 上传

解压后以内层 `GTSRNet/` 为仓库根目录，直接包含 README、依赖文件
和以上源码目录。这是统一命名的完整替换版，不要与之前旧目录并排混放。
没有自动删除或修改远程仓库中的任何文件。
