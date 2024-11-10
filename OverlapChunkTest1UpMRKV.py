from keras.utils import to_categorical
import numpy as np
import math
import matplotlib.pyplot as plt

class OverlapChunkTest1UpMRKV:
    
    def __init__(self, time_delay, m=2):
        self.chunk = 0
        self.output_size = 8
        self.counter = -1
        self.time_delay = time_delay
        self.time_counter = time_delay
        self.output_class = 0
        self.previous_output_class = None
        self.previous_previous_output_class = None
        self.sequenceA_length = 4
        self.sequenceB_length = 4
        self.m = m  # 设置马尔可夫链阶数
        self.state_history = []  # 用于存储过去 m 个状态
        self.noise_intensity = 0.0  # 原始代码中的噪声强度
        self.decay_rate = 0.05 # 定义衰减速率 default=0.1

    def getOutputSize(self):
        return self.output_size
    
    def trueLabel(self):
        truelabel = np.array((0, 0, 0, 1, 1, 2, 2, 2))
        return truelabel

    def updateTimeDelay(self):
        self.time_counter += 1
        if self.time_counter > self.time_delay:
            self.time_counter = 0
            return True
        else:
            return False

    # 创建系统的输入模式
    def getInput(self, reset=False):
        
        if reset:
            self.chunk = 0
            self.counter = -1
            self.state_history.clear()  # 重置状态历史队列

        update = self.updateTimeDelay()

        if update:
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

            # 存储过去 m 个状态的历史
            if len(self.state_history) >= self.m:
                self.state_history.pop(0)  # 移除最早的状态
            self.state_history.append(self.output_class)  # 添加当前状态

            # 更新当前状态（取决于当前 chunk 是否为 0）
            if self.chunk == 0:
                self.output_class = np.random.randint(5)  # 可能的输出 0~4
            else:
                self.output_class = 3 + np.random.randint(5)  # 可能的输出 3~7

        # 基于状态历史队列生成输入值
        input_value = np.zeros(self.output_size)
        for i, state in enumerate(reversed(self.state_history)):
            decay = np.exp(-self.decay_rate * (self.time_counter + i * self.time_delay)) # 衰减速率 default=0.1
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

    def plot(self, input_class, input_sequence=None, save=False):
        a = np.asarray(input_class)
        t = [i for i, value in enumerate(a)]

        plt.plot(t, a)
        
        if input_sequence is not None:
            sequence = [np.argmax(x) for x in input_sequence]
            plt.plot(t, sequence)

        if save:
            plt.savefig("plot.png")
        
        plt.show()
        plt.close()
    
    def plotSuperposed(self, input_class, input_sequence=None, save=False):
        input_sequence = np.asarray(input_sequence)
        
        t = [i for i, value in enumerate(input_sequence)]
        print(input_sequence.shape)

        for i in range(input_sequence.shape[1]):
            a = input_sequence[:, i]
            plt.plot(t, a)
        
        a = np.asarray(input_class)
        plt.plot(t, a)

        if save:
            plt.savefig("plot.png")
        
        plt.show()
        plt.close()
