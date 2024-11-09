################################################################################ 
# Code developed by Danilo Vasconcellos Vargas @ Kyushu University / The University of Tokyo
################################################################################ 

from keras.utils import to_categorical
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial import distance
import pickle
from scipy.spatial.distance import pdist, squareform
import networkx as nx

plt.style.use('dark_background')  # 应用黑暗模式
mpl.rcParams['text.color'] = 'white'


class SyncMapNoDBSCAN:
    
    def __init__(
            self, 
            input_size, 
            dimensions, 
            adaptation_rate, 
            weight_matrix=None, 
            theta=0.5):
        
        self.organized= False
        self.space_size= 10
        self.dimensions= dimensions
        self.input_size= input_size
        #syncmap= np.zeros((input_size,dimensions))
        self.syncmap= np.random.rand(input_size,dimensions)
        self.adaptation_rate= adaptation_rate
        #self.syncmap= np.random.rand(dimensions, input_size)

        # PARSER
        self.weight_matrix = weight_matrix   # 添加权重矩阵
        self.theta = theta  # Distance threshold for adjacency

        
    # def inputGeneral(self, x):
    #     plus= x > 0.1
    #     minus = ~ plus

    #     sequence_size = x.shape[0]
    #     for i in range(sequence_size):
    #         vplus= plus[i,:]
    #         vminus= minus[i,:]
    #         plus_indices = np.where(vplus)[0]
    #         minus_indices = np.where(vminus)[0]
    #         plus_mass = len(plus_indices)
    #         minus_mass = len(minus_indices)

    #         if plus_mass <= 1:
    #             continue
    #         if minus_mass <= 1:
    #             continue

    #         # 计算正样本中心
    #         center_plus= np.mean(self.syncmap[plus_indices], axis=0)
    #         # 计算负样本中心
    #         center_minus= np.mean(self.syncmap[minus_indices], axis=0)

    #         # 更新正样本节点的位置
    #         for idx in plus_indices:
    #             # 计算与其他正样本节点的权重加权更新
    #             weighted_sum = np.zeros(self.dimensions)
    #             total_weight = 0.0
    #             for jdx in plus_indices:
    #                 if idx != jdx:
    #                     weight = self.weight_matrix[idx, jdx] if self.weight_matrix is not None else 1.0
    #                     diff = self.syncmap[jdx] - self.syncmap[idx]
    #                     dist = np.linalg.norm(diff)
    #                     if dist > 0:
    #                         weighted_sum += weight * diff / dist
    #                         total_weight += weight
    #             if total_weight > 0:
    #                 delta = self.adaptation_rate * (weighted_sum / total_weight)
    #                 self.syncmap[idx] += delta

    #         # 更新负样本节点的位置
    #         for idx in minus_indices:
    #             # 计算与正样本节点的权重加权更新
    #             weighted_sum = np.zeros(self.dimensions)
    #             total_weight = 0.0
    #             for jdx in plus_indices:
    #                 weight = self.weight_matrix[idx, jdx] if self.weight_matrix is not None else 1.0
    #                 diff = self.syncmap[jdx] - self.syncmap[idx]
    #                 dist = np.linalg.norm(diff)
    #                 if dist > 0:
    #                     weighted_sum -= weight * diff / dist
    #                     total_weight += weight
    #             if total_weight > 0:
    #                 delta = self.adaptation_rate * (weighted_sum / total_weight)
    #                 self.syncmap[idx] += delta

    #         # 保持节点位置在定义的空间范围内
    #         maximum=self.syncmap.max()
    #         self.syncmap= self.space_size*self.syncmap/maximum

    def inputGeneral(self, x):
        plus = x > 0.1
        minus = ~plus

        sequence_size = x.shape[0]
        for i in range(sequence_size):
            vplus = plus[i, :]
            vminus = minus[i, :]
            plus_mass = vplus.sum()
            minus_mass = vminus.sum()

            if plus_mass <= 1 or minus_mass <= 1:
                continue

            center_plus = np.dot(vplus, self.syncmap) / plus_mass
            center_minus = np.dot(vminus, self.syncmap) / minus_mass

            # Update positions with co-occurrence weights
            update_plus = vplus[:, None] * (center_plus - self.syncmap)
            update_minus = vminus[:, None] * (center_minus - self.syncmap)

            self.syncmap += self.adaptation_rate * (update_plus - update_minus)
            maximum = self.syncmap.max()
            self.syncmap = self.space_size * self.syncmap / maximum

        

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


    def organize(self):
        self.organized = True

        # Compute distance matrix
        dist_matrix = squareform(pdist(self.syncmap, metric='euclidean'))

        # Build adjacency matrix using distance threshold
        adjacency_matrix = (dist_matrix < self.theta).astype(int)

        # Incorporate co-occurrence weights
        if self.weight_matrix is not None:
            adjacency_matrix *= (self.weight_matrix > 0)

        # Build graph and find connected components
        G = nx.from_numpy_array(adjacency_matrix)
        connected_components = list(nx.connected_components(G))

        # Assign labels based on connected components
        self.labels = -np.ones(self.input_size, dtype=int)
        for cluster_id, component in enumerate(connected_components):
            for node_idx in component:
                self.labels[node_idx] = cluster_id

        return self.labels





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

        input_sequence= input_sequence[-1000:]
        input_class= input_class[-1000:]

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
        
        print(self.syncmap)
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
        

