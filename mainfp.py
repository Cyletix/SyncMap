# main.py functionalized processing

from keras.utils import to_categorical
import numpy as np
import math
import matplotlib.pyplot as plt
import os

# Problems
from ChunkTest import *
from OverlapChunkTest1 import *
from OverlapChunkTest1UpMRKV import *
from OverlapChunkTest2 import *
from LongChunkTest import *
from FixedChunkTest import *
from GraphWalkTest import *
import sys

# Neurons
from SyncMap import *
from SyncMapNoDBSCAN import *
# from MRILNeuron import *
from VAE import *

from sklearn.metrics import normalized_mutual_info_score
import time

import matplotlib
matplotlib.use('TkAgg')  # 或 'Qt5Agg'，根据安装的库决定
import matplotlib.pyplot as plt

# Check TensorFlow GPU
import tensorflow as tf
print(tf.__version__)
print("GPU Available:", tf.config.list_physical_devices('GPU'))


# 移除DBSCAN, 使用n-gram, 计算共生权重
# def calculate_cooccurrence_weights(input_sequence, n, threshold):
#     from collections import Counter, defaultdict

#     # 将输入序列转换为节点索引序列
#     sequence_indices = np.argmax(input_sequence, axis=1)

#     ngrams = [tuple(sequence_indices[i:i + n]) for i in range(len(sequence_indices) - n + 1)]
#     freq_count = Counter(ngrams)

#     # 选择频率大于阈值的 n-gram
#     frequent_ngrams = {ngram: count for ngram, count in freq_count.items() if count > threshold}

#     # 初始化共现计数矩阵
#     cooccurrence_counts = defaultdict(int)

#     for ngram, count in frequent_ngrams.items():
#         nodes_in_ngram = set(ngram)
#         for i in nodes_in_ngram:
#             for j in nodes_in_ngram:
#                 if i != j:
#                     cooccurrence_counts[(i, j)] += count

#     # 转换为矩阵形式
#     num_nodes = input_sequence.shape[1]
#     cooccurrence_matrix = np.zeros((num_nodes, num_nodes))
#     for (i, j), count in cooccurrence_counts.items():
#         cooccurrence_matrix[i, j] = count

#     # 归一化权重矩阵
#     max_count = cooccurrence_matrix.max()
#     if max_count > 0:
#         weight_matrix = cooccurrence_matrix / max_count
#     else:
#         weight_matrix = cooccurrence_matrix

#     return weight_matrix



def run_main_program(adaptation_rate=0.01, map_dimensions=2, m=2, eps=3, min_samples=2, problem_type=6):

    # Determine `save_dir` as the directory where `main.py` is located
    save_dir = os.path.join(os.path.dirname(__file__), "output_files/")
    os.makedirs(save_dir, exist_ok=True)  # Ensure `output_files` directory exists

    # # Command-line arguments handling
    # arg_size = len(sys.argv)
    # if arg_size > 1:
    #     problem_type = sys.argv[1]
    #     save_filename = os.path.join(save_dir, sys.argv[2])
    # else:
        # Default problem type and filenames if not provided
    save_filename = os.path.join(save_dir, "output_file")
    save_truth_filename = save_filename + "_truth"


    problem_type=6
    # print("Problem type:", problem_type)
    time_delay = 10 #tstemp 论文测试条件: 10

    # problem_type = int(problem_type)

    # Initialize the problem environment based on problem_type
    if problem_type == 1:
        env = GraphWalkTest(time_delay)
    elif problem_type == 2:
        env = FixedChunkTest(time_delay)
    elif problem_type == 3:
        env = GraphWalkTest(time_delay, "sequence2.dot")
    elif problem_type == 4:
        env = GraphWalkTest(time_delay, "sequence1.dot")
    elif problem_type == 5:
        env = LongChunkTest(time_delay)
    elif problem_type == 6:
        # env = OverlapChunkTest1(time_delay)
        env = OverlapChunkTest1UpMRKV(time_delay,m=m)
    elif problem_type == 7:
        env = OverlapChunkTest2(time_delay)
    else:
        print("Invalid problem type. Exiting.")
    output_size = env.getOutputSize()

    # print("Output Size:", output_size)

    sequence_length = 100000

    # Generate input sequence and class labels
    input_sequence, input_class = env.getSequence(sequence_length)

    # 计算关联权重矩阵
    # weight_matrix = calculate_cooccurrence_weights(input_sequence, n, frequency_threshold)

    ####### SyncMap #####
    number_of_nodes = output_size
    # If adaptation_rate was provided, multiply it by output_size
    adaptation_rate = adaptation_rate * output_size
    # print("Adaptation rate:", adaptation_rate)

    # SyncMap
    neuron_group = SyncMap(
        input_size=number_of_nodes,
        dimensions=map_dimensions,
        adaptation_rate=adaptation_rate,
        eps=3,
        min_samples=2
    )

    # Create an instance of SyncMap with the provided parameters
    neuron_group = SyncMapNoDBSCAN(
        input_size=number_of_nodes,
        dimensions=map_dimensions,
        adaptation_rate=adaptation_rate,
        # weight_matrix=weight_matrix  # 传递权重矩阵
    )
    ####### SyncMap #####

    # If you want to use VAE instead, you can uncomment and adjust the following:
    ###### VAE #####
    # input_size = output_size
    # latent_dim = 3
    # timesteps = 100
    # neuron_group = VAE(input_size, latent_dim, timesteps)
    ###### VAE #####

    # 调整 `organize` 方法中的 `theta`
    # neuron_group.theta = theta  # 新增传递的距离阈值参数



    # Feed the input sequence to the neuron group (SyncMap)
    neuron_group.input(input_sequence)
    labels = neuron_group.organize()

    print("Learned Labels: ", labels)
    print("Correct Labels: ", env.trueLabel())

    if save_filename is not None:
        with open(save_filename, "a+") as f:
            tmp = np.array2string(labels, precision=2, separator=',')
            f.write(tmp + "\n")
        if labels is not None:
            with open(save_truth_filename, "a+") as f:
                tmp = np.array2string(env.trueLabel(), precision=2, separator=',')
                f.write(tmp + "\n")


    # Get ground truth labels
    trueLabel = env.trueLabel()
    # Get labels predicted by SyncMap
    learned_labels = labels

    # Handle -1 labels (noise) by assigning them a new label
    noise_label = max(learned_labels) + 1
    learned_labels = np.array([label if label != -1 else noise_label for label in learned_labels])

    # Calculate NMI score
    nmi_score = normalized_mutual_info_score(trueLabel, learned_labels)

    # Optionally, print current parameters and NMI score
    # print("Current Parameters:")
    print(f"Adaptation Rate: {adaptation_rate}")
    print(f"Map Dimensions: {map_dimensions}")
    print(f"markov length: 2")
    print(f"DBSCAN eps: {eps}")
    print(f"DBSCAN min_samples: {min_samples}")
    print(f"NMI Score: {nmi_score}")


    # 保存图片
    timestamp = time.strftime("%Y%m%d-%H%M%S") # 获取当前时间戳
    color=trueLabel
    save= False 

    # neuron_group.plot(color,save,filename= "plot_map.png")
    # neuron_group.plot(color, save=save, filename=f"{save_dir}plot_map_{timestamp}.png")
    # input_sequence, input_class = env.getSequence(100000)
    # neuron_group.plotSequence(input_sequence, input_class,filename="plotSequence.png")
    # neuron_group.plotSequence(input_sequence, input_class, save=save, filename=f"{save_dir}plotSequence_{timestamp}.png")

    return nmi_score, learned_labels, trueLabel, output_size

if __name__ == '__main__':
    # set default parameters here or use command-line arguments
    # default parameters:
    adaptation_rate = 0.1 # 论文测试条件: 0.1
    map_dimensions = 3 # 论文测试条件: 3
    # eps = 5
    # min_samples = 2
    m=2 #markov order 论文测试条件: 2

    # Run the main program with the specified parameters  运行一次
    # run_main_program(adaptation_rate, map_dimensions, eps, min_samples,problem_type=6)


    # 按照论文中的实验方法, 运行10次取平均和方差
    nmi_scores = []
    num_runs = 10  # 论文测试条件: 10
    for i in range(num_runs): 
        nmi_score, _, _, _ = run_main_program(
            adaptation_rate=adaptation_rate,
            map_dimensions=map_dimensions,
            problem_type=6,
        )
        nmi_scores.append(nmi_score)
        print(f"Run {i+1}:")

    mean_nmi = np.mean(nmi_scores)
    std_nmi = np.std(nmi_scores)
    # print(f"Average NMI Score: {mean_nmi}, Standard Deviation: {std_nmi}")
    print(f"|m  |Average NMI Score| Standard Deviation|\n|m={m}|{mean_nmi}|{std_nmi}|")