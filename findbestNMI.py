import json

jsonpath='output_files/results_overlap1.json'

# 读取数据
with open(jsonpath, 'r') as f:
    data = json.load(f)

# 提取所有代的种群
all_individuals = []
for generation, details in data['generations'].items():
    all_individuals.extend(details['population'])

# 找到 NMI_score 最高的个体
best_individual = max(all_individuals, key=lambda x: x['NMI_score'])

# 输出最优参数
print("最佳参数:")
for key, value in best_individual.items():
    print(f"{key}: {value}")



import pandas as pd
# 提取 generations 数据
generations = data.get('generations', {})

# 转换为表格形式
rows = []
for gen, details in generations.items():
    for individual in details['population']:
        row = {
            'Generation': gen,
            **individual
        }
        rows.append(row)

# 创建 DataFrame
df = pd.DataFrame(rows)

# 保存为 CSV 文件
csv_path = 'output_files/results_overlap1_table.csv'
df.to_csv(csv_path, index=False)

csv_path
