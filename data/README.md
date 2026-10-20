# 数据准备与可用性

本目录用于说明数据。**现有代码实际读取项目根目录的 `dataset/`，不是 `data/`。**
增加本说明不会改变任何数据加载器。

## 1. 文件放置

```text
项目根目录/
├── data/
│   └── README.md
├── dataset/
│   ├── README.md
│   ├── data_git_version.h5
│   ├── MobileBJ.npy
│   ├── MobileNJ.npy
│   └── crawled_feature.csv       # 仅 legacy Milan loader 需要
├── utils/
└── GTSRNet_model/
```

五个预测任务由三个主流量文件提供。以下形状和时间信息来自所核对归档中的
实际文件，不代表其他版本的数据也具有相同形状。

| 任务名 | 文件 | 实际布局 | 默认网格设置 |
|---|---|---|---|
| `milan_sms` | `dataset/data_git_version.h5` | `data.shape=(1488,10000,3)`，通道0 | 100×100 中的 `[40:60,40:60]`，即20×20 |
| `milan_call` | 同一 HDF5 | 通道1 | 同上 |
| `milan_internet` | 同一 HDF5 | 通道2 | 同上 |
| `mobile_bj` | `dataset/MobileBJ.npy` | `(168,672,1)`，float64 | 28×24 |
| `mobile_nj` | `dataset/MobileNJ.npy` | `(504,560,1)`，float64 | 20×28 |

Milan HDF5 至少需要 `data` 和 `idx` 两个键；`idx` 应与时间轴等长。
已核对文件共有1488个时间戳，从 `2013-11-01 00:00:00` 到
`2014-01-01 23:00:00`，相邻间隔均为3600秒。不能将这个加工文件的截止日期
直接写成2013-12-31。

Mobile 文件不包含可核实的日历时间戳。当前代码按小时序列处理，并以序列
起点作为周期相位0；这不能证明首个观测发生在真实午夜。
包内 MobileBJ 只有168个时间点，按小时计为7天，并不是672小时的28天数据。
若论文采用28天版本，必须取得对应数据后重跑；更改说明不会补足缺失数据。

辅助文件 `crawled_feature.csv` 有10000行，列为
`social, BSs, POI_Transportation, POI_Other`。默认 `clean` Milan loader
不使用此文件；选择 `legacy` 时才需要对应辅助数据。

## 2. 来源和下载方式

### Milan 原始数据

可从原始发布渠道了解和下载原始数据：

- [Telecom Italia / Harvard Dataverse 发布页](https://doi.org/10.7910/DVN/EGZHFV)
- [原始数据论文：A multi-source dataset of urban life in the city of Milan and the Province of Trentino](https://www.nature.com/articles/sdata201555)

原始蜂窝记录与本项目的小时级 HDF5 不是相同格式。下载时应查阅发布页的
数据使用条款。当前代码归档未提供从原始记录完整生成此 HDF5 的处理链，
也未记录 SMS/Call 入出业务合并、缺失值处理等全部细节，因此不能声称仅下载
原始数据即可逐字节重建论文使用的加工文件。

已经持有项目加工数据时，将以上三个主文件按规定名称放入 `dataset/`。
需要完全重建加工数据的使用者，应通过论文提供的作者联系渠道取得处理说明
或可发布的加工文件；本说明不虚构加工文件下载地址。

### MobileBJ、MobileNJ 和辅助 CSV

所核对归档没有提供足够信息来确认这三个文件的原始发布地址、DOI、完整时间戳
或再分发许可。本仓库不能据此给它们指定公开下载地址或授权再分发。
作者确认这些信息后应更新本节；未获可发布来源的数据不随代码上传。

## 3. 检查本地格式

在安装项目依赖并准备数据后，从项目根目录执行：

```bash
python tools/check_release.py --project-root . --check-data
```

检查器只核对文件和形状，不修改数据、不启动训练，也不证明数据与论文版本
完全一致。若需要核对28天 MobileBJ 的最小长度，可额外使用
`--bj-min-hours 672`；归档中168步文件会按预期检查失败。

## 4. 默认现代训练协议

主模型、`baselines/` 和 `GTSRNet_sensitivity/` 默认使用正向时间滑窗、
`clean` Milan loader 和按时间顺序的训练/验证/测试划分。历史输入默认24步，
输出默认12步，滑窗步长为1。Milan/NJ 的默认测试集为最后168个窗口，
BJ 为最后24个窗口；这些是重叠滑窗数，不是独立观测数量。

默认隔离间隔 `gap=预测长度-1`，用于避免跨集合的预测目标区间重叠；
不能据此声称各集合的所有历史输入观测完全不重叠。归一化使用训练窗口的
输入和目标拟合一个全局 Min–Max 尺度，验证和测试沿用该尺度，测试指标在
逆归一化后计算。

默认敏感性调度命令以最大预测长度24对齐预测起点和隔离间隔，不能直接混入
其他切分协议下的旧 checkpoint。更改预测范围后，调度器会按配置中的最大范围
推导对齐参数；直接运行单任务训练器不会自动完成跨设置对齐。
独立 PMF/RMF 对照强制使用 `clean` 与
`chronological`。原工程也保留了 `legacy` 和 `random` 等历史入口；它们的
处理方式不能与上述现代默认协议混为一谈。

## 5. 数据与代码分别管理

根目录 `.gitignore` 默认排除 `.h5`、`.npy`、`.npz` 等大型数据和数组。
它不提供数据授权，也不会移除已经被 Git 跟踪的文件。
代码授权状态见 [../LICENSE_NOTICE.md](../LICENSE_NOTICE.md)。
