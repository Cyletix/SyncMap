from keras.utils import to_categorical
import numpy as np
import math
import matplotlib.pyplot as plt


class OverlapChunkTest2UpMRKV2:
    def __init__(self, time_delay, m=2):
        self.chunk = 0
        self.output_size = 10
        self.counter = -1
        self.tstep = time_delay     # 论文中的 tstep
        self.time_counter = 0       # 追踪当前时间步
        self.output_class = 0
        self.m = m                  # 马尔可夫链阶数
        
        # 存储状态转换历史，每个元素是(state, transition_time)的元组
        self.state_transitions = []
        
        self.sequenceA_length = 4
        self.sequenceB_length = 4
        self.noise_intensity = 0.0
        self.decay_rate = 0.1

    def getOutputSize(self):
        return self.output_size
    
    def trueLabel(self):
        return np.array((0,0,0,0,0,1,1,1,1,1))

    def updateTimeDelay(self):
        self.time_counter += 1
        if self.time_counter >= self.tstep:
            self.time_counter = 0
            return True
        return False

    def getInput(self, reset=False):
        if reset:
            self.chunk = 0
            self.counter = -1
            self.state_transitions.clear()
            self.time_counter = 0
        
        current_time = len(self.state_transitions) * self.tstep + self.time_counter
        update = self.updateTimeDelay()
        
        if update:
            # 更新chunk和counter
            if self.chunk == 0:
                if self.counter > self.sequenceA_length:
                    self.chunk = 1
                    self.counter = 0
                else:
                    self.counter += 1
            else:
                if self.counter > self.sequenceB_length:
                    self.chunk = 0
                    self.counter = 0
                else:
                    self.counter += 1
            
            # 根据chunk确定新的输出类别
            prev_output = self.output_class
            if self.chunk == 0:
                self.output_class = np.random.randint(5)  # 可能的输出 0~4
            else:
                self.output_class = np.random.randint(10)  # 可能的输出 0~9
            
            # 如果状态发生变化，记录转换
            if len(self.state_transitions) == 0 or prev_output != self.output_class:
                self.state_transitions.append((self.output_class, current_time))
                # 只保留最近m个转换
                if len(self.state_transitions) > self.m:
                    self.state_transitions.pop(0)
        
        # 计算输入向量
        input_value = np.zeros(self.output_size)
        current_time = len(self.state_transitions) * self.tstep + self.time_counter
        
        # 对每个记录的状态转换计算其贡献
        for state, trans_time in self.state_transitions:
            time_diff = current_time - trans_time
            # 只考虑时间差小于m*tstep的状态
            if time_diff < self.m * self.tstep:
                decay = np.exp(-self.decay_rate * time_diff)
                input_value += to_categorical(state, self.output_size) * decay
        
        # 添加噪声
        input_value += np.random.randn(self.output_size) * self.noise_intensity
        return input_value

    def getSequence(self, iterations):
        input_class = np.empty(iterations)
        input_sequence = np.empty((iterations, self.output_size))
        
        for i in range(iterations):
            input_value = self.getInput()
            input_class[i] = self.chunk
            input_sequence[i] = input_value
            
        return input_sequence, input_class

    def plot(self,input_class,input_sequence=None,save=False):
        a = np.asarray(input_class)
        t = [i for i,value in enumerate(a)]

        plt.plot(t,a)
        
        if input_sequence is not None:
            sequence = [np.argmax(x) for x in input_sequence]
            plt.plot(t,sequence)

        if save:
            plt.savefig("plot.png")
        
        plt.show()
        plt.close()
    
    def plotSuperposed(self,input_class,input_sequence=None,save=False):
        input_sequence = np.asarray(input_sequence)
        
        t = [i for i,value in enumerate(input_sequence)]
        print(input_sequence.shape)

        for i in range(input_sequence.shape[1]):
            a = input_sequence[:,i]
            plt.plot(t,a)
        
        a = np.asarray(input_class)
        plt.plot(t,a)

        if save:
            plt.savefig("plot.png")
        
        plt.show()
        plt.close()
