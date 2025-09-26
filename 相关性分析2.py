import pandas as pd
import statsmodels.api as sm
import numpy as np

# 文件路径
file_path = 'output_files/SyncMapResult.csv'

# 读取数据
data = pd.read_csv(file_path)

# 转换布尔值列
if 'NMI new algo' in data.columns:
    data['NMI new algo'] = data['NMI new algo'].map({False: 0, True: 1})

# 检查非数值列并转换或排除
non_numeric_columns = data.select_dtypes(include=['object']).columns
data = data.drop(columns=non_numeric_columns)

# 填充或移除 NaN 和 Inf 数据
data = data.replace([np.inf, -np.inf], np.nan)  # 替换 inf 为 NaN
data = data.dropna()  # 删除包含 NaN 的行

# 自变量（X）和因变量（y）
X = data.drop(columns=['avg'])  # 假设 'avg' 是目标列
y = data['avg']

# 标准化自变量
X_standardized = (X - X.mean()) / X.std()

# 添加常量项
X_standardized = sm.add_constant(X_standardized)

# 检查标准化结果是否仍存在 NaN 或 Inf
if X_standardized.isna().any().any():
    raise ValueError("清理后 X_standardized 中仍存在 NaN 或 Inf，请检查数据预处理步骤。")

# 拟合OLS模型
model = sm.OLS(y, X_standardized).fit()

# 打印模型摘要
summary = model.summary()
print("模型摘要：")
print(summary)

# 生成参数影响分析表格
effect_params = pd.DataFrame({
    'Parameter': ['Intercept'] + X.columns.tolist(),
    'Coefficient': model.params,
    'P-value': model.pvalues
})

# 排序并标记显著性
effect_params['Abs_Coefficient'] = effect_params['Coefficient'].abs()
effect_params = effect_params.sort_values(by='Abs_Coefficient', ascending=False)
effect_params['Significant'] = effect_params['P-value'] < 0.05

print("\n参数对 avg NMI 分数的影响（按显著性排序）：")
print(effect_params)

# 保存结果到文件
output_file = 'output_files/SyncMapResult_parameter_influence_analysis_cleaned.csv'
effect_params.to_csv(output_file, index=False)
print(f"\n分析完成，结果已保存到 '{output_file}'。")
