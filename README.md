# GTSRNet：蜂窝网络多步流量预测

GTSRNet 包含空间预测分支、分段循环时间预测分支和融合模块。
本工程中 **`GTSRNet` 是完整模型**。根目录说明统一介绍数据准备、模型入口、
实验命令和结果管理；各实验的细节仍见对应子目录的 README。

本文档按提供的代码归档逐项核对入口和参数编写。新增说明和检查工具不修改
原模型、训练代码或实验结果。参考安装版本不是原服务器环境的实测导出；
所列命令也不等于已经完成了全部 GPU 训练验证。核对范围见
[RELEASE_CHECKS.md](RELEASE_CHECKS.md)。

本源码包已整合下列模型与实验代码，不需要再覆盖此前的根目录补充包。
收录范围和排除项见 [PACKAGE_CONTENTS.md](PACKAGE_CONTENTS.md)；
另外收录的统计分析和 SGA-Seq 运行命令见 [OPTIONAL_ANALYSES.md](OPTIONAL_ANALYSES.md)。

## 公开名称与实验配置

完整模型的类名、命令标识和新输出标签统一为 `GTSRNet`；例如
`--variants GTSRNet`、`--model GTSRNet`。目录和脚本按用途命名，
不再把随机种子数量、GPU数量或内部版本号写进名称。
种子列表、重复次数、设备、训练预算和消融差异仍按原配置保留。

`GTSRNet` 类也承载消融变体。为保持历史配置，部分底层接口仍默认
`variant="full"`；该标识是未校准门控对照，不是当前完整模型。
直接构造论文完整模型时必须显式传入 `variant="GTSRNet"`；
正式训练按下文显式选择 `GTSRNet`，不要省略模型/变体参数。

本版对目录、模块、类名及标签进行了成套更名。既有权重的层属性和
参数键没有改名，但旧运行目录与 checkpoint/JSON 中的历史模型标签
不会被自动改写。复用历史实验时保留原版本读取记录，或在副本中按
实际配置适配标签；不要把新默认路径当作旧结果已经迁移完成。

## 1. 目录与实验入口

```text
项目根目录/
├── README.md
├── requirements.txt
├── .gitignore
├── LICENSE_NOTICE.md
├── THIRD_PARTY_NOTICES.md
├── data/README.md
├── dataset/                         # 本地放置数据，详见 data/README.md
├── tools/                           # 本次增加的检查和环境导出工具
├── utils/                           # 完整模型和部分基线必需的基础模块
├── GTSRNet_model/          # GTSRNet 与主要组件消融
├── baselines/                 # 六类外部基线与 GTSRNet
├── recent_baselines/           # iTransformer、SegRNN、STID
├── GTSRNet_fusion_ablation/
├── GTSRNet_gate_ablation/
├── decoder_comparison/ # 独立时间预测器 PMF/RMF 等对照
├── GTSRNet_sensitivity/                   # 窗口/预测长度、相关性、指标分析
├── GTSRNet_attention/
├── GTSRNet_error_analysis/
└── GTSRNet_scalability/              # 窗口化大网格实现
```

`GTSRNet_joint_ablation/`、`GTSRNet_lowrank_ablation/`、
`GTSRNet_branch_analysis/` 为可选补实验目录。只发布与论文采用的实验相关的
代码时，应同时保留其依赖。特别是 scalability 默认使用
`GTSRNet_gate_ablation/` 中的模型；attention 和 branch reliability 依赖
`GTSRNet_model/` 的训练核心。`utils/` 不能省略。

## 2. 环境

优先使用已完成实验的服务器环境，并先导出其版本。若从新环境开始，可使用
以下参考安装方案。参考 Python 为3.9，直接依赖版本写在 `requirements.txt`；
没有将未核实的版本称为论文原环境，也没有锁定所有传递依赖。

```bash
conda create -n gtsrnet python=3.9 -y
conda activate gtsrnet
```

进入包含 `utils/` 与本 README 的项目根目录。按运行设备选择一种 PyTorch 安装方式。
以下 CPU 和 CUDA 11.8 wheel 命令依据
[PyTorch 官方历史版本安装说明](https://pytorch.org/get-started/previous-versions/)。
GPU wheel 应与服务器驱动及显卡兼容；其他 CUDA 组合请按官方页面选择。

```bash
# NVIDIA GPU：参考 CUDA 11.8 wheel
python -m pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu118
python -m pip install -r requirements.txt
python -m pip check
```

仅做 CPU 检查时，改用下面这一组，不要同时执行两种 wheel 安装：

```bash
python -m pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt
python -m pip check
```

`einops`、`reformer-pytorch` 是注意力文件在导入阶段就会加载的依赖。
即使当前 GTSRNet 分支没有调用 Reformer，也需要安装。
旧数据加载器还会导入 pandas 和 scikit-learn，部分合成图脚本需要 Pillow。

检查结构与已安装版本：

```bash
python tools/check_release.py --project-root . --check-environment
```

`INFO` 的版本差异表示与参考版本不同，不自动代表原服务器环境有错。
记录真实实验环境：

```bash
python tools/export_environment.py --output-dir repro_env/server
```

此命令生成实际安装包版本列表、核心包版本列表和包含 Python/PyTorch/CUDA
信息的 JSON；不导出账号、访问令牌、环境变量或包的私有下载地址。
输出列表是环境记录，不是跨平台依赖锁。恢复 CUDA 版本时仍须选择正确的 wheel 源。

## 3. 数据准备与检查

详细格式、来源和限制见 [data/README.md](data/README.md)。默认读取路径是：

```text
dataset/data_git_version.h5
dataset/MobileBJ.npy
dataset/MobileNJ.npy
```

Milan 三种业务共享一个 HDF5，不是三个不同数据文件。
辅助 `dataset/crawled_feature.csv` 仅 legacy Milan loader 需要。
当前归档中的 MobileBJ 是168步版本，不能称为672小时的28天版本。
原始 Milan 发布源与本项目小时级加工文件也不能直接视为相同文件。

准备文件后执行：

```bash
python tools/check_release.py --project-root . --check-data
```

默认现代协议使用正向时间、按时间顺序切分、训练窗口拟合归一化尺度。
不同入口仍可能采用不同协议，尤其是旧 `legacy`/`random` 入口以及敏感性实验。
引用结果时必须保留实际运行配置。

## 4. 先做小规模检查

以下检查使用合成小输入，不执行正式数据训练：

```bash
python GTSRNet_model/train_ablation.py \
  --project-root . --smoke-test --device cpu

python baselines/train_baseline.py \
  --project-root . --model all --smoke-test --device cpu

python decoder_comparison/verify_decoder.py
```

三个命令全部正常退出后再启动真实数据任务。各训练器接口不同，不能给所有
脚本统一加 `--smoke-test`；PMF/RMF 使用独立的 `verify_decoder.py`。

多数调度器支持 `--dry-run`，但仍可能检查数据或 checkpoint。
主消融、fixed_avg、门控、scalability 和 lowrank 的调度器没有该参数。

## 5. 种子、训练预算与结果口径

下表是所核对源码的默认值，**不是最终论文所有图表的种子清单**：

| 入口 | 源码默认种子 |
|---|---|
| 主模型/组件消融、fixed_avg、门控 | 单数参数 `--seed 42` |
| `baselines` | `--seeds 15 25 35` |
| `recent_baselines` | `--seeds 23 29 103 128` |
| PMF/RMF | `--seeds 45 46 47 48 49` |
| `GTSRNet_sensitivity` | `--seeds 2` |
| 注意力、机制分析、lowrank | `--seeds 16 26 36` |
| 窗口化 scalability | `--seeds 23 29 103` |
| 十种子 wo_joint 包装入口 | `--seeds 1 2 3 4 5 6 7 8 9 10` |

以下提供可执行的重跑示例。主模型与基线示例显式使用23、29、103、128，
以便采用一致的重复次数；这是一份重跑方案，不声明历史结果均来自该方案。
其他补实验也写出完整种子。复算论文既有数字时，应将数据版本、种子、
划分、训练预算与原运行 `metrics.json`/配置一一对齐。

以下命令面向 Linux/Bash，全部从项目根目录执行。`--gpus 0` 为单卡顺序调度；
有多张可用卡时可改为 `--gpus 0 1 2 3`。长任务在服务器上自行使用 tmux 等方式保持运行。

### 5.1 完整 GTSRNet 模型

```bash
for SEED in 23 29 103 128; do
  python -u GTSRNet_model/run_ablation.py \
    --project-root . --gpus 0 \
    --datasets milan_sms milan_call milan_internet mobile_bj mobile_nj \
    --variants GTSRNet --seed "$SEED" \
    --epochs 300 --batch-size 16 --learning-rate 0.001 \
    --loss l1 --branch-aux-weight 0.1 --patience 0 \
    --split-mode chronological --split-gap -1 --milan-loader clean \
    --output-dir "results/main_seed${SEED}"
done
```

必须显式指定 `GTSRNet`；旧的 `full` 不是本文档中的完整模型标识。
该调度器没有复数 `--seeds` 参数，所以用循环并给每个种子独立输出目录。

### 5.2 六类外部基线与三个近期基线

```bash
python -u baselines/run_baselines.py \
  --project-root . --gpus 0 \
  --datasets milan_sms milan_call milan_internet mobile_bj mobile_nj \
  --models seasonal_ha gru_seq2seq mtgnn patchtst dlinear timemixer GTSRNet \
  --seeds 23 29 103 128 \
  --close-size 24 --predict-size 12 \
  --epochs 300 --batch-size 16 --learning-rate 0.001 \
  --loss l1 --branch-aux-weight 0.1 --patience 0 \
  --split-mode chronological --split-gap -1 --milan-loader clean \
  --output-dir results/baselines

python -u recent_baselines/run_recent_baselines.py \
  --project-root . --gpus 0 \
  --datasets milan_sms milan_call milan_internet mobile_bj mobile_nj \
  --seeds 23 29 103 128 \
  --output-dir results/recent_baselines
```

第二条入口默认运行 iTransformer、SegRNN、STID。
统计基线可能不需要重复随机训练，汇总时应披露其实际有效运行次数。
两组基线的专有参数见各子目录说明及运行配置，不应仅凭相同输出长度宣称
所有超参数完全相同。目录间存在重复模型文件，不要混用不同版本的结果。

### 5.3 组件消融

```bash
for SEED in 23 29 103 128; do
  python -u GTSRNet_model/run_ablation.py \
    --project-root . --gpus 0 \
    --datasets milan_sms milan_call milan_internet mobile_bj mobile_nj \
    --variants GTSRNet wo_s wo_t wo_sthe wo_seg wo_g \
    --seed "$SEED" --epochs 300 --batch-size 16 \
    --loss l1 --branch-aux-weight 0.1 --patience 0 \
    --split-mode chronological --split-gap -1 --milan-loader clean \
    --output-dir "results/ablation_seed${SEED}"
done
```

该原始组件入口仅对 `GTSRNet` 实际启用分支辅助损失，其他组件变体的有效辅助
权重为0，即使命令统一传入 `--branch-aux-weight 0.1`。
因此这些原始结果不能解释为只改变一个结构因素、其余训练目标严格不变的对照。

### 5.4 fixed_avg 与门控机制

```bash
for SEED in 23 29 31; do
  python -u GTSRNet_fusion_ablation/run_ablation.py \
    --project-root . --gpus 0 \
    --datasets milan_sms milan_call milan_internet mobile_bj mobile_nj \
    --variants GTSRNet fixed_avg --seed "$SEED" \
    --epochs 300 --batch-size 16 --loss l1 --branch-aux-weight 0.1 --patience 0 \
    --split-mode chronological --split-gap -1 --milan-loader clean \
    --output-dir "results/fixedavg_seed${SEED}"
done

for SEED in 23 29 103 128; do
  python -u GTSRNet_gate_ablation/run_ablation.py \
    --project-root . --gpus 0 \
    --datasets milan_sms milan_call milan_internet mobile_bj mobile_nj \
    --variants GTSRNet fixed_avg horizon_static GTSRNet_no_aux --seed "$SEED" \
    --epochs 300 --batch-size 16 --loss l1 --branch-aux-weight 0.1 --patience 0 \
    --split-mode chronological --split-gap -1 --milan-loader clean \
    --no-plot --output-dir "results/gate_seed${SEED}"
done
```

门控目录缺少调度器可选调用的绘图文件，命令显式使用 `--no-plot`。
已有的训练和指标输出仍可使用；绘图与统计应根据最终采用的结果另行生成。
这两个目录的训练器会对双分支变体启用分支辅助损失，与5.3的原入口有差异。
它们的 `fixed_avg` 直接平均原始分支输出，因此相对 `GTSRNet` 同时移除了动态
门控和逐预测步分支校准。`horizon_static` 保留校准参数，更适合用于隔离检验
样本/网格相关门控的作用；不能将这两种对照称为完全相同的消融。

### 5.5 并行 PMF 与递归 RMF

```bash
python -u decoder_comparison/run_decoder_experiments.py \
  --project-root . --gpus 0 \
  --datasets milan_sms milan_call milan_internet mobile_bj mobile_nj \
  --horizons 12 24 --seeds 45 46 47 48 49 \
  --output-dir results/decoder
```

显式列出五个数据集，避免入口默认仅 Call/Internet 而遗漏 SMS、BJ、NJ。
若复算45–49和301–306共11个历史种子，可将种子参数改为
`--seeds 45 46 47 48 49 301 302 303 304 305 306`，并核对其余实际配置相同。

该入口比较的是从时间分支构建并独立训练的受控预测器，不是完整双分支
GTSRNet 的两个别名。PMF 与 RMF 结果不能直接替代主模型表格。
正式多种子统计应保留全部约定运行，不按测试效果筛选种子。

### 5.6 历史窗口、预测范围与时间相关性

```bash
python -u GTSRNet_sensitivity/analyze_temporal_dependence.py \
  --project-root . \
  --datasets milan_sms milan_call milan_internet \
  --max-lag 72 --max-period 168 \
  --output-dir results/sensitivity/dependence

python -u GTSRNet_sensitivity/run_sensitivity.py \
  --project-root . --gpus 0 \
  --datasets milan_sms milan_call milan_internet mobile_bj mobile_nj \
  --histories 12 24 48 72 --horizons 3 6 12 24 \
  --base-history 24 --base-horizon 12 \
  --seeds 23 29 103 128 \
  --output-dir results/sensitivity
```

此入口对齐不同预测长度的预测起点，24→12也应在该协议下重跑。
在3步输出、片段长6时，时间分支先解码完整片段，再截取前3步；
片段数应写为向上取整，而不是将片段长度自动改为3。
时间相关性分析需要可核实的时间间隔，Milan 文件提供时间戳；Mobile文件没有
同等的日历依据。

### 5.7 注意力分析：先准备完整模型 checkpoint

```bash
python -u GTSRNet_attention/run_attention_analysis.py \
  --project-root . --gpus 0 \
  --baseline-results-root results/baselines \
  --seeds 23 29 103 128 \
  --output-dir results/attention
```

此入口分析已有 checkpoint，不会替代主模型训练。种子必须与
`results/baselines` 中实际保存的 GTSRNet checkpoint 一致。
它会依赖 `GTSRNet_model/` 的训练核心；不能只上传注意力目录。

### 5.8 机制与误差图

```bash
python -u GTSRNet_error_analysis/run_mechanism_experiments.py \
  --project-root . --gpus 0 \
  --seeds 16 26 36 \
  --results-dir results/mechanism
```

该入口的输出参数是 `--results-dir`。默认空间相关实验涵盖五任务，
phase 相关实验只涵盖 Milan；不能将两类任务范围写成完全一致。

### 5.9 窗口化大规模网格实验

```bash
python -u GTSRNet_scalability/run_windowed_experiments.py \
  --project-root . --gpus 0 \
  --datasets milan_internet --seeds 23 29 103 \
  --grid-size 100 --window-size 20 \
  --output-dir results/scalability
```

这是100×100数据上的20×20共享模型窗口化实现，并不是把原始全局注意力
直接扩展到10000个 token。默认训练预算为100 epochs、patience=10，
与上述300 epochs主实验不同。依赖 `GTSRNet_gate_ablation/`。

### 5.10 可选：联合嵌入、低秩与分支可靠性

保留对应代码目录时才执行：

```bash
python -u GTSRNet_joint_ablation/run_joint_comparison.py \
  --project-root . --gpus 0 \
  --seeds 1 2 3 4 5 6 7 8 9 10 \
  --output-dir results/wo_joint

python -u GTSRNet_lowrank_ablation/run_lowrank_ablation.py \
  --project-root . --gpus 0 \
  --variants GTSRNet lowrank_joint wo_joint --seeds 16 26 36 \
  --output-dir results/lowrank

python -u GTSRNet_branch_analysis/run_branch_reliability.py \
  --project-root . --results-root results/ablation_seed23 \
  --output-dir results/reliability_seed23 --gpu 0
```

wo_joint 包装入口要求恰好10个不同种子，并自动关闭其底层缺失的可选绘图入口。
不要改用底层 `run_embedding_ablation.py` 后遗漏 `--no-plots`。
分支可靠性分析读取先前组件实验的结果，不会自动生成这些 checkpoint。

## 6. 保存与公开结果

示例命令将原始运行产物写入 `results/`，该目录默认被 `.gitignore` 排除。
它可包含 checkpoint、配置、指标、损失记录和预测数组；不要覆盖不同种子的
运行目录。部分调度器会跳过已有完成指标的任务，不能把旧配置的结果当作新任务。

论文对应的小型结果可单独整理到 `results_summary/`，并保留：

- 每个报告运行的配置与种子；
- 每个种子的完整指标及需要绘图的逐步误差；
- 汇总统计的脚本、种子集合、有效运行数和标准差定义；
- 论文表图与结果文件的对应关系。

本补充没有生成或补填任何实验数字。大型数组、全部权重和重复图不必随代码
提交，但绘图脚本需要的数值应有可重新生成的途径。只保留测试表现较好的
种子不能作为完整的多种子结果公开。

`.gitignore` 对已经被Git跟踪的文件不起移除作用；发布前应查看实际暂存文件。
说明里的目录选择也不会自动清理服务器上的历史代码或结果。

## 7. 许可证和数据可用性

授权状态见 [LICENSE_NOTICE.md](LICENSE_NOTICE.md)，来源线索见
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。当前归档未提供项目级许可证，
本补充没有代替著作权人选择 MIT 或其他授权。正式公开为开源项目之前，
作者应确认项目授权并保留第三方条款。

Milan 原始发布源、小时级加工文件的缺失处理链，以及 MobileBJ/NJ 来源尚需
作者补充的信息均在 [data/README.md](data/README.md) 明确记录。
仓库不提供未经确认的数据再分发授权或虚构下载链接。
