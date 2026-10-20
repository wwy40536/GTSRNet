# 支持模块

本目录保留原归档的支持源码，以保持实验间的导入路径兼容。主用入口依赖
`layers/Embed.py`、`layers/SelfAttention_Family.py`、`layers/Transformer_EncDec.py`、
`segrnn_gridB.py`、`MTGNN.py`、`layer.py`、`timemixer_model.py` 和相关支持文件。

这里也包含未被根 README 主用入口调用的历史模型。根依赖清单不承诺这些
历史模型的全部可选依赖都已安装；请从根 README 指定的实验入口运行，
不要将旧模块中的本地路径或数据示例当作本次发布的启动配置。

原代码中的第三方来源和版权注释已保留，参见根目录 `THIRD_PARTY_NOTICES.md`。
