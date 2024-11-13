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
from OverlapChunkTest2UpMRKV import *
from LongChunkTest import *
from FixedChunkTest import *
from GraphWalkTest import *
import sys

# Neurons
from SyncMap import *
from SyncMapDynaAdp2 import *
from SyncMapLouvain import *
from SyncMapWeightDB import *


# from MRILNeuron import *
from VAE import *

from sklearn.metrics import normalized_mutual_info_score
import time

import matplotlib
matplotlib.use('TkAgg')  # 或 'Qt5Agg'，根据安装的库决定
import matplotlib.pyplot as plt
import itertools

# Check TensorFlow GPU
# import tensorflow as tf
# print(tf.__version__)
# print("GPU Available:", tf.config.list_physical_devices('GPU'))


def run_main_program(adaptation_rate=0.01, map_dimensions=2, m=2, eps=3, min_samples=2, problem_type=6,NMI_new=False,model='SyncMap'):

    # Determine `save_dir` as the directory where `main.py` is located
    save_dir = os.path.join(os.path.dirname(__file__), "output_files/")
    os.makedirs(save_dir, exist_ok=True)  # Ensure `output_files` directory exists

    time_delay = 10 #tstemp 论文测试条件: 10

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
        # env = OverlapChunkTest2(time_delay)
        env = OverlapChunkTest2UpMRKV(time_delay,m=m)
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
    # adaptation_rate = adaptation_rate * output_size # 已经动态调节 adaptation_rate，这样的放大处理会导致糟糕的结果
    # print("Adaptation rate:", adaptation_rate)


    if model=="SyncMap":
        # SyncMap
        neuron_group = SyncMap(
            input_size=number_of_nodes,
            dimensions=map_dimensions,
            adaptation_rate=adaptation_rate,
            eps=3,
            min_samples=2
        )    
    elif model=="SyncMapDynaAdp2":
        # Create an instance of SyncMap with the provided parameters
        neuron_group = SyncMapDynaAdp2(
            env,
            input_size=number_of_nodes,
            dimensions=map_dimensions,
            adaptation_rate=adaptation_rate,
        )
    elif model=="SyncMapLouvain":
        neuron_group = SyncMapLouvain(
            input_size=number_of_nodes,
            dimensions=map_dimensions,
            adaptation_rate=adaptation_rate
        )
    elif model=="SyncMapWeightDB":
        neuron_group = SyncMapWeightDB(
            input_size=number_of_nodes,
            dimensions=map_dimensions,
            adaptation_rate=adaptation_rate
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

    # print("Learned Labels: ", labels)
    # print("Correct Labels: ", env.trueLabel())


    # save_filename = os.path.join(save_dir, "output_file")
    # save_truth_filename = save_filename + "_truth"
    # if save_filename is not None:
    #     with open(save_filename, "a+") as f:
    #         tmp = np.array2string(labels, precision=2, separator=',')
    #         f.write(tmp + "\n")
    #     if labels is not None:
    #         with open(save_truth_filename, "a+") as f:
    #             tmp = np.array2string(env.trueLabel(), precision=2, separator=',')
    #             f.write(tmp + "\n")


    # Get ground truth labels
    trueLabel = env.trueLabel()
    # Get labels predicted by SyncMap
    learned_labels = labels

    if NMI_new:
        # 使用自定义新方法计算NMI
        learned_labels_cal=[]
        for i,label in enumerate(learned_labels):
            if label==-1:
                label=max(learned_labels) + 1+i
            learned_labels_cal.append(label)
        learned_labels_cal=np.array(learned_labels_cal)
    else:
        # Handle -1 labels (noise) by assigning them a new label
        noise_label = max(learned_labels) + 1
        learned_labels_cal = np.array([label if label != -1 else noise_label for label in learned_labels])

    # Calculate NMI score
    nmi_score = normalized_mutual_info_score(trueLabel, learned_labels_cal)
    # nmi_score=neuron_group.nmi_score # test


    # print(f"Adaptation Rate: {neuron_group.adaptation_rate}")
    # print(f"Map Dimensions: {map_dimensions}")
    # print(f"markov length: {m}")
    # print(f"DBSCAN eps: {eps}")
    # print(f"DBSCAN min_samples: {min_samples}")
    # print(f"NMI Score: {nmi_score}")


    # 保存图片
    timestamp = time.strftime("%Y%m%d-%H%M%S") # 获取当前时间戳
    color=trueLabel
    save= False 

    neuron_group.plot(color, save=save, filename=f"{save_dir}plot_map_{timestamp}.png")
    # input_sequence, input_class = env.getSequence(1000)
    # neuron_group.plotSequence(input_sequence[-1000:], input_class[-1000:], save=save, filename=f"{save_dir}plotSequence_{timestamp}.png")

    return nmi_score, learned_labels, trueLabel, output_size

if __name__ == '__main__':
    # set default parameters here or use command-line arguments
    # problem_type=7
    # # default parameters:
    # adaptation_rate = 0.1 # 论文测试条件: 0.1
    # map_dimensions = 3 # 论文测试条件: 3
    # eps = 3 # SyncMap.py条件: 3
    # min_samples = 2 # SyncMap.py条件: 2
    # m = 2 # markov order 论文测试条件: 2

    # 定义参数的可能值
    problem_type_values = [6,7]
    adaptation_rate_values = [0.1,0.01,0.001]# , 0.01, 0.001,0.0001
    map_dimensions_values = [3]
    eps_values = [3]
    min_samples_values = [2]
    m_values = [2,3,5]
    NMI_new = [True]
    model_list=["SyncMap","SyncMapLouvain"] # "SyncMap",SyncMapWeightDB, SyncMapLouvain

    # 按照论文中的实验方法, 运行10次取平均和方差
    nmi_scores = []
    num_runs = 10  # 论文测试条件: 10
    print("|model|overlap|Average NMI Score|Standard Deviation|adaptation_rate|map_dimensions|markov length|eps|min_samples||condition|")
    print("|-|-|-|-|-|-|-|-|-|-|")
    for (problem_type, adaptation_rate, map_dimensions, eps, min_samples, m,NMI_new,model) in itertools.product(
    problem_type_values, adaptation_rate_values, map_dimensions_values, eps_values, min_samples_values, m_values,NMI_new,model_list
):
        for i in range(num_runs): 
            # print(f"\nRun {i+1}:")
            nmi_score, _, _, _ = run_main_program(
                adaptation_rate=adaptation_rate,
                map_dimensions=map_dimensions,
                m=m,
                problem_type=problem_type,
                eps=eps,
                min_samples=min_samples,
                NMI_new=NMI_new,
                model=model
            )
            nmi_scores.append(nmi_score)

        mean_nmi = np.mean(nmi_scores)
        std_nmi = np.std(nmi_scores)
        # print("|overlap|Average NMI Score|Standard Deviation|adaptation_rate|map_dimensions|markov length|eps|min_samples|")
        # print("|-|-|-|-|-|-|-|-|")
        print(f"|{model}|{problem_type-5}|{mean_nmi}|{std_nmi}|{adaptation_rate}|{map_dimensions}|{m}|{eps}|{min_samples}|{NMI_new}|")
