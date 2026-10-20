# GTSRNet 的融合与训练实现

完整模型通过标识 `GTSRNet` 选择。`full` 是另一种未校准门控对照，
两者不是同一标识，也不能混用其结果。

## 逐预测步校准与动态融合

对于预测步 `h=1,...,T_out`，先校准两个分支的预测：

```text
s_h = a_spatial[h] * y_spatial[h] + b_spatial[h]
t_h = a_temporal[h] * y_temporal[h] + b_temporal[h]
alpha_h,cell = sigmoid(MLP([s_h, t_h, horizon_embedding[h]]))
y_hat_h = alpha_h,cell * s_h + (1 - alpha_h,cell) * t_h
```

`a_spatial`、`a_temporal` 初始化为1，偏置初始化为0；门控输出初始化为0.5。
门控系数可随样本、预测步和栅格变化，不是全局标量门控。
准确的参数结构与初始化以 `gtsrnet_model.py` 中的实现为准。

## 分支辅助损失

```text
L_train = L_fusion + lambda * (L_spatial + L_temporal) / 2
lambda = --branch-aux-weight
```

本目录训练器的辅助权重默认0.1。辅助项仅用于训练；验证集和测试集评估
最终融合输出，checkpoint 按验证集融合指标选择，而不是按测试集选择。
完整模型命令、训练预算、数据切分和实际种子见[根目录说明](../README.md)。

## 名称与历史记录

本次发布整理同步更新了模块导入、模型标识、运行命令及图例，
没有改变层属性名、参数张量、计算公式或随机种子配置。
旧 checkpoint/JSON 中的历史标签与旧输出路径不会自动转换。
复用历史记录时应保留原件，在副本中核对实际配置，避免覆盖或错误混合结果。

模型是否优于对照须由相同协议的实测结果判断，应报告既定种子的完整结果与波动；
名称统一不构成精度、收敛性或普遍优越性的证明。
