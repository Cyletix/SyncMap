import random
import numpy as np
import json
import os
import argparse
from mainfp import run_main_program
from scipy.stats import entropy
# Check TensorFlow GPU
import tensorflow as tf
print(tf.__version__)
print("GPU Available:", tf.test.is_gpu_available())


# def set_random_seed(seed=42):
#     np.random.seed(seed)
#     random.seed(seed)

# set_random_seed()


# Define parameter bounds
param_bounds = {
    'adaptation_rate': (0.0001, 0.01),
    'map_dimensions': (2, 5),
    'eps': (0.5, 10),
    'min_samples': (2, 10),
    'm': (2, 10)  
}

# # NoDBSCAN
# param_bounds = {
#     'adaptation_rate': (0.001, 0.1),
#     'map_dimensions': (3, 3),
#     'm': (2, 2),
#     'n': (2, 5),
#     'frequency_threshold': (1, 10),
#     'theta': (0.1, 2.0)
# }


# Initialize population
def initialize_population(size, previous_best=None):
    population = []
    if previous_best:
        # 如果有上代最优个体，加入种群
        population.append(previous_best)
    
    while len(population) < size:
        chromosome = {
            'adaptation_rate': random.uniform(*param_bounds['adaptation_rate']),
            'map_dimensions': random.randint(*param_bounds['map_dimensions']),
            'eps': random.uniform(*param_bounds['eps']),
            'min_samples': random.randint(*param_bounds['min_samples']),
            'm': random.randint(*param_bounds['m']),  # 添加 m 参数
            # 'n': random.uniform(*param_bounds['n']),
            # 'frequency_threshold': random.uniform(*param_bounds['frequency_threshold']),
            # 'theta': random.uniform(*param_bounds['theta'])
        }
        population.append(chromosome)
    return population


# Fitness function
def fitness_function(chromosome):
    adaptation_rate = chromosome['adaptation_rate']
    map_dimensions = chromosome['map_dimensions']
    m = chromosome['m']
    
    # DBSCAN    
    eps = chromosome['eps']
    min_samples = chromosome['min_samples']

    # PARSER
    # n = chromosome['n']
    # frequency_threshold = chromosome['frequency_threshold']
    # theta = chromosome['theta']
    
    # 多任务测试
    nmi_scores = []
    label_entropies = []
    num_runs = 10  # 设置运行次数
    for problem_type in [6]:
        problem_nmi_scores = []
        problem_label_entropies = []
        for i in range(num_runs):
            print(f"Run {i+1}:")
            nmi_score, learned_labels, _, _ = run_main_program(
                adaptation_rate=adaptation_rate,
                map_dimensions=map_dimensions,
                eps=eps,
                min_samples=min_samples,
                m=m,
                problem_type=problem_type
            )
            problem_nmi_scores.append(nmi_score)
            
            # 计算标签的熵作为分布惩罚
            # _, counts = np.unique(learned_labels, return_counts=True)
            # problem_label_entropies.append(entropy(counts))
            
        
        # 计算当前 problem_type 的平均 NMI 和熵
        avg_nmi = np.mean(problem_nmi_scores)
        nmi_scores.append(avg_nmi)
        # avg_entropy = np.mean(problem_label_entropies)
        # label_entropies.append(avg_entropy)

    # 计算所有 problem_type 的平均 NMI 和熵
    avg_nmi_score = np.mean(nmi_scores)
    # avg_entropy = np.mean(label_entropies)
    
    # 惩罚过低的熵，减去分布惩罚项, 如果不考虑就直接返回avg_nmi_score
    # fitness = avg_nmi_score - 0.1 * (1 - avg_entropy)
    return avg_nmi_score


# Selection, crossover, and mutation functions
def select(population, fitness_scores):
    sorted_population = [chrom for _, chrom in sorted(zip(fitness_scores, population), key=lambda pair: pair[0], reverse=True)]
    return sorted_population[:len(population)//2]

def crossover(parent1, parent2):
    child = {key: random.choice([parent1[key], parent2[key]]) for key in parent1.keys()}
    return child

def mutate(chromosome):
    mutation_prob = 0.2
    for key in chromosome.keys():
        if key == 'NMI_score':  # 跳过 NMI_score
            continue
        if random.random() < mutation_prob:
            if key in ['map_dimensions', 'm', 'n', 'frequency_threshold']:#'min_samples',
                chromosome[key] = random.randint(*param_bounds[key])
            else:
                chromosome[key] = random.uniform(*param_bounds[key])
    return chromosome


def reproduce(selected):
    next_generation = []
    while len(next_generation) < len(selected) * 2:
        parent1, parent2 = random.sample(selected, 2)
        child = crossover(parent1, parent2)
        child = mutate(child)
        next_generation.append(child)
    return next_generation

# Evolutionary algorithm
def evolutionary_algorithm(generations, population_size, continue_training=False):
    os.makedirs('output_files', exist_ok=True)
    
    generations_data = {}
    last_population = None
    
    # 检查是否有已存在的结果文件
    if continue_training and os.path.exists('output_files/results_overlap1.json'):
        with open('output_files/results_overlap1.json', 'r') as f:
            data = json.load(f)
            generations_data = data.get("generations", {})
            last_generation = str(max(map(int, generations_data.keys())))
            last_population = generations_data[last_generation]["population"]
        print("Loaded last generation's population from previous run.")
    
    # 初始化种群
    if last_population:
        population = [chromosome for chromosome in last_population]
    else:
        population = initialize_population(population_size)
    
    best_chromosome = None
    best_fitness = -np.inf  # 初始化全局最佳适应度为负无穷

    for gen in range(generations):
        print(f"Generation {gen}")
        fitness_scores = []

        # Evaluate fitness for each chromosome
        for chromosome in population:
            score = fitness_function(chromosome)
            fitness_scores.append(score)
            chromosome['NMI_score'] = score
        
        # 计算代内统计数据
        gen_avg_fitness = np.mean(fitness_scores)
        gen_min_fitness = np.min(fitness_scores)
        gen_best_fitness = np.max(fitness_scores)
        gen_best_index = fitness_scores.index(gen_best_fitness)
        gen_best_chromosome = population[gen_best_index].copy()

        # 更新全局最佳个体
        if gen_best_fitness > best_fitness:
            best_fitness = gen_best_fitness
            best_chromosome = gen_best_chromosome
        
        # 保存当前代的结果
        generations_data[str(gen)] = {
            "population": population,
            "best": gen_best_chromosome,
            "min": gen_min_fitness,
            "avg": gen_avg_fitness
        }
        
        # 选择和繁殖
        selected = select(population, fitness_scores)
        population = reproduce(selected)
        
        # 将所有代的结果写入文件
        with open('output_files/results_overlap1.json', 'w') as f:
            json.dump({"generations": generations_data}, f, indent=4)
    
    # 返回最后的最佳个体
    return best_chromosome






if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Improve NMI by EA")
    parser.add_argument('--continue', dest='continue_training', action='store_true', help='Continue training from the last saved best chromosome')
    args = parser.parse_args()
    
    generations = 100
    population_size = 10
    best_params = evolutionary_algorithm(generations, population_size, continue_training=args.continue_training)
    
    print("Best Parameters Found:")
    print(f"Adaptation Rate: {best_params['adaptation_rate']}")
    print(f"Map Dimensions: {best_params['map_dimensions']}")
    print(f"DBSCAN eps: {best_params['eps']}")
    print(f"DBSCAN min_samples: {best_params['min_samples']}")
    print(f"Markov Chain Length m: {best_params['m']}")
    print(f"NMI Score: {best_params['NMI_score']}")
