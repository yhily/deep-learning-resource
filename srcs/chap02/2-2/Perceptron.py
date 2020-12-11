# -*- coding: utf-8 -*-
"""
Created on Thu Sep 28 13:43:39 2017

@author: Yuhong
"""


class Perceptron(object):
    def __init__(self, input_para_num, acti_func):
        '''
        初始化感知器，设置输入参数的个数及激活函数。
        '''
        self.activator = acti_func
        # 权重向量初始化为0       
        self.weights =[0.0 for _ in range(input_para_num)]
        
    def __str__(self):
        '''
        打印学习到的权重，其中w0为偏置项
        '''
        return 'final weights\n\tw0 = {:.2f}\n\tw1 = {:.2f}\n\tw2 = {:.2f}' \
                .format(self.weights[0],self.weights[1],self.weights[2])
    def predict(self, row_vec):
        '''
        输入向量，输出感知器的计算结果
        '''
        act_values = 0.0
        for i in range(len(self.weights)):
            act_values += self.weights[ i ] * row_vec [ i ]
        return self.activator(act_values)
    
    def train(self, dataset, iteration, rate):
        '''
        输入训练数据：一组向量、与每个向量对应的label；以及训练轮数、学习率
        '''
        for i in range(iteration):
            for input_vec_label in dataset:
                # 计算感知器在当前权重下的输出
                prediction = self.predict(input_vec_label)
                # 更新权重
                self._update_weights(input_vec_label,prediction, rate)
            
    def _update_weights(self, input_vec_label, prediction, rate):
        '''
        按照感知器规则更新权重
        '''
        delta =  input_vec_label[-1] - prediction 
        for i in range(len(self.weights)):
             self.weights[ i ] += rate * delta * input_vec_label[ i ]
           
# 定义激活函数f      
def func_activator(input_value):
    return 1.0 if input_value >= 0.0 else 0.0

def get_training_dataset():
    '''
    基于and真值表构建训练数据
    '''
    # 构建训练数据
    dataset = [[-1, 1, 1, 1], [-1, 0, 0, 0], [-1, 1, 0, 0], [-1, 0, 1, 0]]
    # 期望的输出列表，注意要与输入一一对应
    # [-1,1,1] -> 1, [-1, 0,0] -> 0, [-1, 1,0] -> 0, [-1, 0,1] -> 0
    return dataset

def train_and_perceptron():
    '''
    使用and真值表训练感知器
    '''
    # 创建感知器，输入参数个数为3（虽然and是二元函数，把哑元-1算上，是三个），激活函数为func_activator
    p = Perceptron(3, func_activator)
    # 训练，迭代10轮, 学习速率为0.1
    dataset = get_training_dataset()
    p.train(dataset, 10, 0.1)
    #返回训练好的感知器
    return p

if __name__ == '__main__': 
    # 训练and感知器
    and_perception = train_and_perceptron()
    # 打印训练获得的权重
    print (and_perception)
    # 测试
    print ('1 and 1 = %d' % and_perception.predict([-1, 1, 1]))
    print ('0 and 0 = %d' % and_perception.predict([-1, 0, 0]))
    print ('1 and 0 = %d' % and_perception.predict([-1, 1, 0]))
    print ('0 and 1 = %d' % and_perception.predict([-1, 0, 1]))