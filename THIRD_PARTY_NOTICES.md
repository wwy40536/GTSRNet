# 第三方实现与来源线索

以下线索直接来自所核对代码的注释或导入关系，用于协助保留归属说明。
本清单不是完整的版权审查，也不代表所有模块都具有同一许可证。

| 文件或依赖 | 原文件中可确认的线索 | 发布时的处理 |
|---|---|---|
| `utils/dataset.py`、`utils/dataset_new.py` | Chuanting Zhang，2018，文件头标注 BSD license | 保留文件头，并从实际来源补齐适用的完整条款；不能仅凭“BSD”判断具体版本 |
| `utils/layers/SelfAttention_Family.py` | 注明 [Flowformer](https://github.com/thuml/Flowformer)、[FlashAttention-PyTorch](https://github.com/shreyansh26/FlashAttention-PyTorch)、[Informer2020](https://github.com/zhouhaoyi/Informer2020) 的实现来源 | 保留来源注释，按实际使用内容核对各自条款 |
| `utils/layers/RevIN.py` | 注明来自 [ts-kim/RevIN](https://github.com/ts-kim/RevIN)，有少量修改 | 保留归属并核对原始条款 |
| `utils/utils.py` | 包含 2024 LeapLabTHU / Dongchen Han 的版权行 | 保留版权行，核对对应代码片段的来源及条款 |
| `reformer-pytorch`、`einops`、PyTorch 等安装依赖 | 由环境安装；`SelfAttention_Family.py` 会在导入时加载前两个依赖 | 保留包自带许可证；本项目不能重新授权第三方包 |

基线方法的论文引用可查 `baselines/CITATIONS.bib`。引用一篇论文不等同于
取得对应代码或数据的再分发许可。项目级授权状态见
[LICENSE_NOTICE.md](LICENSE_NOTICE.md)。
