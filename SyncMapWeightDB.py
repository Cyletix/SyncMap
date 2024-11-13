################################################################################ 
# Code developed by Danilo Vasconcellos Vargas @ Kyushu University / The University of Tokyo
################################################################################ 

from keras.utils import to_categorical
import numpy as np
import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.cluster import DBSCAN
from scipy.spatial import distance
import pickle
from sklearn.cluster import SpectralClustering
import networkx as nx
from scipy.spatial import distance

class SyncMapWeightDB:
    
    def __init__(self, input_size, dimensions, adaptation_rate, eps=3, min_samples=2):
        
        self.organized= False
        self.space_size= 10
        self.dimensions= dimensions
        self.input_size= input_size
        #syncmap= np.zeros((input_size,dimensions))
        self.syncmap= np.random.rand(input_size,dimensions)
        self.adaptation_rate= adaptation_rate
        #self.syncmap= np.random.rand(dimensions, input_size)

        # DBSCAN参数
        self.eps = eps
        self.min_samples = min_samples
    
    def inputGeneral(self, x):
        
        plus= x > 0.1
        minus = ~ plus

        sequence_size = x.shape[0]
        #print(sequence_size, "asfasdfasdfasd")
        for i in range(sequence_size):
            
            vplus= plus[i,:]
            vminus= minus[i,:]
            plus_mass = vplus.sum()
            minus_mass = vminus.sum()

            #print(plus_mass)
            #print(minus_mass)
            
            if plus_mass <= 1:
                continue
            
            if minus_mass <= 1:
                continue

            #print("vplus")
            #print(vplus)
            
            center_plus= np.dot(vplus,self.syncmap)/plus_mass
            center_minus= np.dot(vminus,self.syncmap)/minus_mass
        
            #print(self.syncmap.shape)
            #exit()
            dist_plus= distance.cdist(center_plus[None,:], self.syncmap, 'euclidean')
            dist_minus= distance.cdist(center_minus[None,:], self.syncmap, 'euclidean')
            dist_plus= np.transpose(dist_plus)
            dist_minus= np.transpose(dist_minus)
            
            #update_plus= vplus[:,np.newaxis]*((center_plus - self.syncmap)/dist_plus + (self.syncmap - center_minus)/dist_minus)
            #update_minus= vminus[:,np.newaxis]*((center_minus -self.syncmap)/dist_minus + (self.syncmap - center_plus)/dist_plus)
            update_plus= vplus[:,np.newaxis]*((center_plus - self.syncmap)/dist_plus)# + (self.syncmap - center_minus)/dist_minus)
            update_minus= vminus[:,np.newaxis]*((center_minus -self.syncmap)/dist_minus)# + (self.syncmap - center_plus)/dist_plus)
            
            
            #self.syncmap+= self.adaptation_rate*update
            
            #self.syncmap= self.space_size*self.syncmap/maximum
            
            update= update_plus - update_minus
            self.syncmap+= self.adaptation_rate*update
        
            maximum=self.syncmap.max()
            self.syncmap= self.space_size*self.syncmap/maximum
            

    def input(self, x):
        
        self.inputGeneral(x)

        return


    def organize_distance_weighted(self):
        self.organized = True
        
        # 计算每对点之间的距离
        distances = distance.cdist(self.syncmap, self.syncmap, 'euclidean')
        
        # 设置距离阈值
        distance_threshold = 5
        
        # 初始化聚类标签
        labels = np.full(self.input_size, -1)
        
        # 遍历每个点,根据距离加权进行聚类
        for i in range(self.input_size):
            # 找到与当前点距离小于阈值的所有点的索引
            neighbors = np.where(distances[i] < distance_threshold)[0]
            
            # 如果没有邻居,则标记为噪声点
            if len(neighbors) < self.min_samples:
                labels[i] = -1
            # 如果有邻居,则分配给其中距离加权最大的簇
            else:
                # 计算每个邻居与当前点的距离加权
                weights = 1 / (distances[i, neighbors] + 1e-6)
                
                # 找到距离加权最大的簇
                cluster_label = neighbors[np.argmax(weights)]
                labels[i] = labels[cluster_label]
                
        return labels
        
    def organize(self):
        self.organized = True
        return self.organize_distance_weighted()


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

    def plotSequence(self, input_sequence, input_class,filename="plot.png"):

        input_sequence= input_sequence[1:500]
        input_class= input_class[1:500]

        a= np.asarray(input_class)
        t = [i for i,value in enumerate(a)]
        c= [self.activate(x) for x in input_sequence] 
        

        plt.plot(t, a, '-g')
        plt.plot(t, c, '-.k')
        #plt.ylim([-0.01,1.2])


        plt.savefig(filename)
        # plt.show()
        plt.close()
    

    def plot(self, color=None, save = False, filename= "plot_map.png"):

        if color is None:
            color= self.labels
        
        # print(self.syncmap) # 控制打印节点 看点的位置的时候记得打开
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
        
        # plt.show()
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
        


