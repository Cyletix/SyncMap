import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# 设置字体以防止中文乱码
plt.rcParams['axes.unicode_minus'] = False    # 解决负号显示问题
plt.style.use('dark_background')  # 应用黑暗模式

# 数据点
# NMI_score = np.array([0.9, 0.6, 0.3])
# adaptation_rate = np.array([0.00001, 0.0001, 0.001])

# 从 CSV 文件读取数据
data = pd.read_csv('output_files/results_overlap1_table.csv')
NMI_score = data['NMI_score'].values
adaptation_rate = data['adaptation_rate'].values


# 数据标准化（可选）
NMI_score_scaled = (NMI_score - np.min(NMI_score)) / (np.max(NMI_score) - np.min(NMI_score))
adaptation_rate_log = np.log(adaptation_rate)  # 取对数

# 定义线性函数来拟合对数数据
def linear_func(x, a, b):
    return a * x + b

# # 定义指数函数
# def exponential_func(x, a, b):
#     return a * np.exp(b * x)

# 拟合曲线
params, params_covariance = curve_fit(linear_func, NMI_score_scaled, adaptation_rate_log)

# 获取拟合参数
a0,b0=[0.1,-4.6052]
a, b = params

# 获取拟合参数并转换为指数函数参数
a_log, b_log = params
a = np.exp(b_log)  # 转换回指数函数的参数
b = a_log

print(f"fitted parameter: a = {a:.4f}, b = {b:.4f}")


# 绘制拟合曲线
x_fit = np.linspace(min(NMI_score), max(NMI_score), 100)
y_fit = a * np.exp(b * (x_fit - np.min(NMI_score)) / (np.max(NMI_score) - np.min(NMI_score)))
y_fit0 = a0 * np.exp(b0 * (x_fit - np.min(NMI_score)) / (np.max(NMI_score) - np.min(NMI_score)))

plt.scatter(NMI_score, adaptation_rate, color='salmon', label='data point')
plt.plot(x_fit, y_fit, label=f'fitted curve: $adp = {a:.4f} \cdot e^{{{b:.4f}NMI}}$', color='cornflowerblue')
# plt.plot(x_fit, y0_fit, label=f'fitted curve: $adp = {a0:.4f} \cdot e^{{{b0:.4f}NMI}}$', color='cornflowerblue')
plt.xlabel('NMI score')
plt.ylabel('adaptation rate')
plt.title('Exponential curve fitting')
plt.legend()
plt.yscale('log')  # 为了更好地展示指数关系
plt.grid(True)

# 显示图表
plt.show()
