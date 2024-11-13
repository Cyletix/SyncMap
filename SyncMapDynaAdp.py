################################################################################ 
# Code developed by Danilo Vasconcellos Vargas @ Kyushu University / The University of Tokyo
################################################################################ 

from keras.utils import to_categorical
import pickle
import math
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import networkx as nx
from scipy.spatial import distance
from scipy.spatial.distance import pdist, squareform
from sklearn.cluster import DBSCAN
from sklearn.metrics import normalized_mutual_info_score
from sklearn.metrics import mean_squared_error
import time

plt.style.use('dark_background')  # 应用黑暗模式


class SyncMapDynaAdp:
    
    def __init__(
            self, 
            env,
            input_size, 
            dimensions, 
            adaptation_rate,
            eps=3,
            min_samples=2, 
            nmi_score=0, # test
            weight_matrix=None
            ):
        self.env=env
        self.organized= False
        self.space_size= 10
        self.dimensions= dimensions
        self.input_size= input_size
        #syncmap= np.zeros((input_size,dimensions))
        self.syncmap= np.random.rand(input_size,dimensions)
        self.adaptation_rate= adaptation_rate
        #self.syncmap= np.random.rand(dimensions, input_size)
        # self.nmi_score=None
        self.weight_matrix = weight_matrix  # 添加权重矩阵
        self.previous_error = float('inf')
        self.num_clusters = None  # 初始化 num_clusters 属性
        # DBSCAN参数
        self.eps = eps
        self.min_samples = min_samples

        # self.test_x=[]
        # self.test_y=[]
        # self.sequence_size = 0


    def inputGeneral(self, x,  update_interval=10, prediction_interval=100, prediction_steps=100):
        # 初始化标签序列
        if not hasattr(self, 'labels_sequence'):
            self.labels_sequence = []

        plus = x > 0.1 # [array([ True,  True, False, False,...
        minus = ~plus

        # 尝试根据规模自适应
        # prediction_interval=int(len(x)/1000)
        # prediction_steps=int(prediction_interval)

        sequence_size = x.shape[0]
        # self.sequence_size=sequence_size
        for i in range(sequence_size):
            vplus = plus[i, :]
            vminus = minus[i, :]
            plus_mass = vplus.sum()
            minus_mass = vminus.sum()

            if plus_mass <= 1 or minus_mass <= 1:
                continue

            center_plus = np.dot(vplus, self.syncmap) / plus_mass
            center_minus = np.dot(vminus, self.syncmap) / minus_mass

            # 更新位置
            update_plus = vplus[:, None] * (center_plus - self.syncmap)
            update_minus = vminus[:, None] * (center_minus - self.syncmap)

            current_syncmap = self.syncmap
            self.syncmap += self.adaptation_rate * (update_plus - update_minus)
            maximum = self.syncmap.max()
            self.syncmap = self.space_size * self.syncmap / maximum

            # 每隔一定步数更新适应率adaptation_rate
            if (i + 1) % update_interval == 0:
                # 更新聚类
                self.organize()
                # 将当前输入的标签加入序列
                current_label = self.activate(x[i])
                self.labels_sequence.append(current_label)

                # 每隔prediction_interval步进行预测和误差计算
                if (i + 1) % prediction_interval == 0:

                    # 获取当前状态
                    current_state = current_label
                    # 预测未来的聚类序列
                    predicted_states = self.predict_future_sequence(current_state, prediction_steps)
                    # 将预测的聚类标签转换为输入向量
                    predicted_inputs = self.labels_to_inputs(predicted_states)
                    # 确保获取的实际输入序列长度足够
                    actual_inputs = x[i+1:i+1+prediction_steps]
                    min_length = min(len(predicted_inputs), len(actual_inputs))

                    if min_length > 0:
                        # 截断到相同长度
                        predicted_inputs = predicted_inputs[:min_length]
                        actual_inputs = actual_inputs[:min_length]

                        # 计算预测误差
                        prediction_error = self.calculate_prediction_error(predicted_inputs, actual_inputs)
                    # 更新适应率
                    self.update_adaptation_rate_based_on_prediction_error(prediction_error)


    def labels_to_inputs(self, labels):
        # 重新计算聚类中心以确保与标签对应
        self.calculate_cluster_centers()
        
        # 安全的标签转换
        max_label = max(self.labels) + 1
        inputs = []
        for label in labels:
            # 如果标签超出范围，使用最近的有效中心
            if label >= len(self.cluster_centers) or label < 0:
                label = min(len(self.cluster_centers) - 1, max(0, label))
            inputs.append(self.cluster_centers[label])
        return np.array(inputs)

    def calculate_cluster_centers(self):
        # 确保包含所有可能的标签
        unique_labels = np.unique(self.labels)
        self.cluster_centers = []
        
        for label in range(min(unique_labels), max(unique_labels) + 1):
            cluster_indices = np.where(self.labels == label)[0]
            if len(cluster_indices) > 0:  # 只为非空簇计算中心
                cluster_inputs = self.syncmap[cluster_indices]
                center = np.mean(cluster_inputs, axis=0)
                self.cluster_centers.append(center)
            else:  # 对于空簇，使用默认中心点
                self.cluster_centers.append(np.zeros(self.dimensions))

    def predict_future_sequence(self, current_state, n_steps):
        if not hasattr(self, 'label_map'):
            self.build_transition_matrix()
        
        if current_state not in self.label_map:
            # 处理未见过的标签
            return [current_state] * n_steps  # 或其他合适的处理方式
            
        state_idx = self.label_map[current_state]
        predicted_states = []
        
        for _ in range(n_steps):
            next_idx = np.random.choice(self.num_clusters, p=self.transition_matrix[state_idx])
            # 将索引转回原始标签
            next_state = [k for k, v in self.label_map.items() if v == next_idx][0]
            predicted_states.append(next_state)
            state_idx = next_idx
            
        return predicted_states

    def build_transition_matrix(self):
        labels_sequence = self.labels_sequence
        unique_labels = np.unique(labels_sequence)  
        self.label_map = {label: idx for idx, label in enumerate(unique_labels)}
        self.num_clusters = len(unique_labels)
        
        # 初始化转移计数矩阵
        transition_counts = np.zeros((self.num_clusters, self.num_clusters))
        
        # 计算转移计数
        for i in range(len(labels_sequence) - 1):
            curr_label = labels_sequence[i]
            next_label = labels_sequence[i + 1]
            
            if curr_label in self.label_map and next_label in self.label_map:
                curr_idx = self.label_map[curr_label] 
                next_idx = self.label_map[next_label]
                transition_counts[curr_idx, next_idx] += 1

        # 处理全零行
        row_sums = transition_counts.sum(axis=1, keepdims=True)
        # 如果某行和为0,则设置为均匀分布
        zero_rows = (row_sums == 0).flatten()
        transition_counts[zero_rows] = 1.0 / self.num_clusters
        
        # 重新计算行和并归一化
        row_sums = transition_counts.sum(axis=1, keepdims=True)
        self.transition_matrix = transition_counts / row_sums

        return self.transition_matrix

    def calculate_prediction_error(self, predicted_inputs, actual_inputs):
        # 将 actual_inputs 映射到投影空间
        actual_indices = [np.argmax(actual_input) for actual_input in actual_inputs]
        actual_positions = self.syncmap[actual_indices]
        # 确保形状一致
        predicted_inputs = np.array(predicted_inputs)
        actual_positions = np.array(actual_positions)
        # 计算预测误差
        error = mean_squared_error(actual_positions.flatten(), predicted_inputs.flatten())
        return error

    # 3. 预测未来的序列来评估模型的性能
    def update_adaptation_rate_based_on_prediction_error(self, prediction_error):
        # 定义适应率调整范围和参数
        max_rate = 1e-2
        min_rate = 1e-4

        # 打印当前和之前的预测误差
        # print(f"Current prediction error: {prediction_error}")
        # print(f"Previous prediction error: {self.previous_error:.2f}")
        # print(f"Adaptation Rate:{self.adaptation_rate:.5f}")
        # time.sleep(0.1)
        
        # 根据预测误差调整适应率
        if prediction_error > self.previous_error:
            # 如果误差增加，增大学习率
            self.adaptation_rate = min(max_rate, self.adaptation_rate*1)
        elif prediction_error < self.previous_error:
            # 如果误差减少，减小学习率
            self.adaptation_rate = max(min_rate, self.adaptation_rate*1)

        # 更新previous_error, 仅在学习效果好的时候记录更新
        self.previous_error = prediction_error

        return self.adaptation_rate



    def input(self, x):
        
        self.inputGeneral(x)

        return
        
        print(x.shape)
        plus= x > 0.1
        minus = ~ plus
#        print(plus)
#        print(minus)
        
#        print(plus.shape)
#        print(type(plus))

#        print(x.shape)
#        print("in",x[1,:])
#        print("map",self.syncmap)
            
        
        sequence_size = x.shape[0]
        for i in range(sequence_size):
            vplus= plus[i,:]
            vminus= minus[i,:]
            plus_mass = vplus.sum()
            minus_mass = vminus.sum()
            #print(self.syncmap)
            #print("plus",vplus)
            if plus_mass <= 1:
                continue
            
            if minus_mass <= 1:
                continue

            #if plus_mass > 0:
            center_plus= np.dot(vplus,self.syncmap)/plus_mass
            #else:
            #    center_plus= np.dot(vplus,self.syncmap)

            #print(center_plus)
            #exit()
            #if minus_mass > 0:
            center_minus= np.dot(vminus,self.syncmap)/minus_mass
            #else:
            #    center_minus= np.dot(vminus,self.syncmap)

            
            #print("mass", minus_mass)
            #print(center_plus)
            #print("minus",vminus)
            #print(center_minus/minus_mass)
            #print(self.syncmap)
            #exit()

            #print(vplus)
            #print(self.syncmap.shape)
            #a= np.matmul(np.transpose(vplus),self.syncmap)
            #a= vplus.dot(self.syncmap)
            #a= (vplus*self.syncmap.transpose()).transpose()
            #update_plus= vplus[:,np.newaxis]*self.syncmap
        #    update_plus= vplus[:,np.newaxis]*(center_plus -center_minus)*plus_mass
            update_plus= vplus[:,np.newaxis]*(center_plus -center_minus)
        #    update_plus= vplus[:,np.newaxis]*(center_plus -center_minus)/plus_mass
            #update_plus= vplus[:,np.newaxis]*(center_plus -self.syncmap)
        #    update_minus= vminus[:,np.newaxis]*(center_minus -center_plus)*minus_mass
            update_minus= vminus[:,np.newaxis]*(center_minus -center_plus)
        #    update_minus= vminus[:,np.newaxis]*(center_minus -center_plus)/minus_mass
            #update_minus= vminus[:,np.newaxis]*(center_minus -self.syncmap)
            #print(self.syncmap)
            #print(center_plus)
            #print(center_plus - self.syncmap)
            #update_minus= vminus[:,np.newaxis]*self.syncmap
            
            #self.plot()

            #ax.scatter(center_plus[0], center_plus[1])
            #ax.scatter(center_minus[0], center_minus[1])
        
            #plt.show()
            
            update= update_plus + update_minus
            self.syncmap+= self.adaptation_rate*update
        
            maximum=self.syncmap.max()
            self.syncmap= self.space_size*self.syncmap/maximum


    def organize(self):
    
        self.organized= True
        #self.labels= DBSCAN(eps=3, min_samples=2).fit_predict(self.syncmap)
        self.labels= DBSCAN(eps=self.eps, min_samples=self.min_samples).fit_predict(self.syncmap) # 接受外部参数

        return self.labels

            
    # def organize(self):
    #     self.organized= True

    #     # 计算节点之间的距离矩阵
    #     from scipy.spatial.distance import pdist, squareform
    #     dist_matrix = squareform(pdist(self.syncmap, metric='euclidean'))

    #     # 设定距离阈值，构建邻接矩阵
    #     theta = 0.5  # 距离阈值，需要根据实际情况调整
    #     adjacency_matrix = (dist_matrix < theta).astype(int)

    #     # 考虑关联权重，调整邻接矩阵
    #     if self.weight_matrix is not None:
    #         adjacency_matrix = adjacency_matrix * (self.weight_matrix > 0)

    #     # 寻找连通分量进行聚类
    #     import networkx as nx
    #     G = nx.from_numpy_array(adjacency_matrix)
    #     connected_components = list(nx.connected_components(G))

    #     # 为每个节点分配聚类标签
    #     self.labels = -np.ones(self.input_size, dtype=int)
    #     for cluster_id, component in enumerate(connected_components):
    #         for node_idx in component:
    #             self.labels[node_idx] = cluster_id

    #     return self.labels


    # def organize(self):
    #     self.organized = True

    #     # Compute distance matrix
    #     dist_matrix = squareform(pdist(self.syncmap, metric='euclidean'))

    #     # Build adjacency matrix using distance threshold
    #     adjacency_matrix = (dist_matrix < self.theta).astype(int)

    #     # Incorporate co-occurrence weights
    #     if self.weight_matrix is not None:
    #         adjacency_matrix *= (self.weight_matrix > 0)

    #     # Build graph and find connected components
    #     G = nx.from_numpy_array(adjacency_matrix)
    #     connected_components = list(nx.connected_components(G))

    #     # Assign labels based on connected components
    #     self.labels = -np.ones(self.input_size, dtype=int)
    #     for cluster_id, component in enumerate(connected_components):
    #         for node_idx in component:
    #             self.labels[node_idx] = cluster_id

    #     return self.labels



    def activate(self, x):
        '''
        Return the label of the index with maximum input value
        '''

        if self.organized == False:
            print("Activating a non-organized SyncMap")
            return
        
        #maximum output
        max_index= np.argmax(x)

        return self.labels[max_index]

    def plotSequence(self, input_sequence, input_class, save = False,filename="output_files/plotSequence.png"):

        input_sequence= input_sequence[-5000:]
        input_class= input_class[-5000:]

        a= np.asarray(input_class)
        t = [i for i,value in enumerate(a)]
        c= [self.activate(x) for x in input_sequence] 


        plt.plot(t, a, '-g')
        plt.plot(t, c, '-.w')
        #plt.ylim([-0.01,1.2])

        if save == True:
            plt.savefig(filename)
        plt.show()
        plt.close()
    

    def plot(self, color=None, save = False, filename= "output_files/plot_map.png"):
        # 展示 syncmap 投影空间中的节点分布
        if color is None:
            color= self.labels
        
        print("self.syncmap:\n",self.syncmap)
        #print(self.syncmap)
        #print(self.syncmap[:,0])
        #print(self.syncmap[:,1])
        if self.dimensions == 2:
            #print(type(color))
            #print(color.shape)
            ax= plt.scatter(self.syncmap[:,0],self.syncmap[:,1], c=color)
            
        if self.dimensions == 3:
            fig = plt.figure()
            ax = plt.axes(projection='3d')

            ax.scatter3D(self.syncmap[:,0],self.syncmap[:,1], self.syncmap[:,2], c=color)
            #ax.plot3D(self.syncmap[:,0],self.syncmap[:,1], self.syncmap[:,2])
        
        if save == True:
            plt.savefig(filename)
        
        plt.show()
        plt.close()

    def save(self, filename):
        """save class as self.name.txt"""
        file = open(filename+'.txt','w')
        file.write(pickle.dumps(self.__dict__))
        file.close()

    def load(self, filename):
        """try load self.name.txt"""
        file = open(filename+'.txt','r')
        dataPickle = file.read()
        file.close()

        self.__dict__ = pickle.loads(dataPickle)
        

