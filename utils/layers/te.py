import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft

# 生成模拟时间序列数据
time_domain_data = np.random.rand(100, 100)

# 进行傅里叶变换到频域
frequency_domain_data = np.abs(fft(time_domain_data, axis=1))

# 绘图
fig, axes = plt.subplots(1, 2, figsize=(12, 6))

# 时间域特征热力图
axes[0].imshow(time_domain_data, cmap='viridis')
axes[0].set_title("Time Domain")
axes[0].set_xlabel("Time")
axes[0].set_ylabel("Feature Index")

# 频域特征热力图
axes[1].imshow(frequency_domain_data, cmap='viridis')
axes[1].set_title("Frequency Domain")
axes[1].set_xlabel("Frequency")
axes[1].set_ylabel("Feature Index")

plt.tight_layout()
plt.show()
