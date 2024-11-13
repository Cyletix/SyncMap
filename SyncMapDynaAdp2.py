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


class SyncMapDynaAdp2:
    
    def __init__(
            self, 
            env,
            input_size, 
            dimensions, 
            adaptation_rate,
            eps=3,
            min_samples=2, 
            nmi_score=0 # test
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
        self.previous_error = float('inf')
        self.num_clusters = None  # 初始化 num_clusters 属性
        # DBSCAN参数
        self.eps = eps
        self.min_samples = min_samples

        # self.test_x=[]
        # self.test_y=[]
        # self.sequence_size = 0


    def inputGeneral(self, x,  update_interval=10, prediction_interval=100, prediction_steps=1000):
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

            if (i + 1) % update_interval == 0:
                self.update_adaptation_rate_based_on_variation(center_plus,center_minus,self.syncmap,current_syncmap)
    
        # plt.plot(self.test_x,self.test_y)

    def update_adaptation_rate_based_on_variation(self, cpt_current, cpt_previous, wi_current, wi_previous):
        # 计算 cpt cnt 和 wi,t 的变化率
        cpt_change_rate = np.linalg.norm(cpt_current - cpt_previous)/cpt_current.size
        wi_change_rate = np.linalg.norm(wi_current - wi_previous)/wi_current.size

        # self.test_x.append(cpt_change_rate)
        # self.test_y.append(wi_change_rate)

        # center_x = np.mean(self.test_x)
        # center_y = np.mean(self.test_y)
        # distance_to_center = np.sqrt((cpt_current - center_x) ** 2 + (wi_current - center_y) ** 2)
        # # 计算斜率，避免除零
        # if cpt_current != center_x:
        #     slope_to_center = (wi_current - center_y) / (cpt_current - center_x)
        # else:
        #     slope_to_center = float('inf')  # 无穷大，表示垂直斜率  

        # # 避免除零错误
        # if wi_change_rate == 0:
        #     wi_change_rate = 1e-6  # 设置一个非常小的值以防止数值错误

        # 根据比值调整适应率
        ratio = cpt_change_rate / wi_change_rate

        # 更新适应率的逻辑
        max_rate = 5e-3
        min_rate = 1e-4

        # 假设一种比值驱动的调整逻辑，可以根据实际需求进行调整

        if ratio > 1:
            self.adaptation_rate = min(max_rate, self.adaptation_rate * (1 + 0.01 * (ratio - 1)))
        elif ratio < 1:
            self.adaptation_rate = max(min_rate, self.adaptation_rate / (1 + 0.1 * (1 - ratio)))
        # self.adaptation_rate *= (1 + 1000/self.sequence_size * (ratio - 1))
        # print(self.adaptation_rate)

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
        

