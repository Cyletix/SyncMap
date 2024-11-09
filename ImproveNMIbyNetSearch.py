from itertools import product
from mainfp import run_main_program
# Check TensorFlow GPU
import tensorflow as tf
print(tf.__version__)
print("GPU Available:", tf.test.is_gpu_available())


# 参数范围
adaptation_rate_values = [0.0001, 0.001, 0.005, 0.01]
map_dimensions_values = [2, 3]
eps_values = [0.5, 1.0, 2.0, 3.0, 5.0]
min_samples_values = [1, 2, 3, 4, 5, 7, 10]

# 所有参数组合
param_combinations = list(product(adaptation_rate_values, map_dimensions_values, eps_values, min_samples_values))

# 运行网格搜索
best_score = -1
best_params = None
for adaptation_rate, map_dimensions, eps, min_samples in param_combinations:
    nmi_score, _, _, _ = run_main_program(
        adaptation_rate=adaptation_rate,
        map_dimensions=map_dimensions,
        eps=eps,
        min_samples=min_samples
    )
    if nmi_score > best_score:
        best_score = nmi_score
        best_params = {
            "adaptation_rate": adaptation_rate,
            "map_dimensions": map_dimensions,
            "eps": eps,
            "min_samples": min_samples
        }

print("Best Parameters:", best_params)
print("Best NMI Score:", best_score)
