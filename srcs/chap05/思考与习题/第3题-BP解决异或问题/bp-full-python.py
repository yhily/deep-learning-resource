# -*- coding: utf-8 -*-
"""
Created on Tue Oct 31 11:43:34 2020

利用NumPy手工实现

@author: Yuhong
"""

from math import exp
from random import seed
from random import random

# 初始化神经网络
def initialize_network(n_inputs, n_hidden, n_outputs):
    network = list()
    hidden_layer = [{'weights':[random() for i in range(n_inputs + 1)]} for i in range(n_hidden)]
    network.append(hidden_layer)
    output_layer = [{'weights':[random() for i in range(n_hidden + 1)]} for i in range(n_outputs)]
    network.append(output_layer)
    return network

# # 计算神经元的激活值（加权之和）
def activate(weights, inputs):
    activation = weights[-1]
    for i in range(len(weights)-1):
        activation += weights[i] * inputs[i]
    return activation

# 定义激活函数
def transfer(activation):
    return 1.0 / (1.0 + exp(-activation))

# 计算神经网络的正向传播
def forward_propagate(network, row):
    inputs = row
    for layer in network:
        new_inputs = []
        for neuron in layer:
            activation = activate(neuron['weights'], inputs)
            neuron['output'] = transfer(activation)
            new_inputs.append(neuron['output'])
        inputs = new_inputs
    return inputs

# 计算激活函数的导数
def transfer_derivative(output):
    return output * (1.0 - output)

# 反向传播误差信息，并将纠偏责任存储在神经元中
def backward_propagate_error(network, expected):
    for i in reversed(range(len(network))):
        layer = network[i]
        errors = list()
        if i != len(network)-1:
            for j in range(len(layer)):
                error = 0.0
                for neuron in network[i + 1]:
                    error += (neuron['weights'][j] * neuron['responsibility'])
                errors.append(error)
        else:
            for j in range(len(layer)):
                neuron = layer[j]
                errors.append(expected[j] - neuron['output'])
        for j in range(len(layer)):
            neuron = layer[j]
            neuron['responsibility'] = errors[j] * transfer_derivative(neuron['output'])

# 根据误差，更新网络权重
def update_weights(network, row, l_rate):
    for i in range(len(network)):
        inputs = row[:-1]
        if i != 0:
            inputs = [neuron['output'] for neuron in network[i - 1]]
        for neuron in network[i]:
            for j in range(len(inputs)):
                neuron['weights'][j] += l_rate * neuron['responsibility'] * inputs[j]
            neuron['weights'][-1] += l_rate * neuron['responsibility']

# 根据指定的训练周期训练网络
def train_network(network, train, l_rate, n_epoch, n_outputs):
    for epoch in range(n_epoch):
        sum_error = 0
        for row in train:
            outputs = forward_propagate(network, row)
            expected = [0 for i in range(n_outputs)]
            expected[row[-1]] = 1
            sum_error += sum([(expected[i]-outputs[i])**2 for i in range(len(expected))])
            backward_propagate_error(network, expected)
            update_weights(network, row, l_rate)
        print('>周期=%d, 误差=%.3f' % (epoch, sum_error))

def predict(network, row):
    outputs = forward_propagate(network, row)
    return outputs.index(max(outputs))
if __name__ == '__main__': 
    # 测试BP网络
    seed(2)
#    dataset = [[2.7810836,2.550537003,0],
#    	[1.465489372,2.362125076,0],
#    	[3.396561688,4.400293529,0],
#    	[1.38807019,1.850220317,0],
#    	[3.06407232,3.005305973,0],
#    	[7.627531214,2.759262235,1],
#    	[5.332441248,2.088626775,1],
#    	[6.922596716,1.77106367,1],
#    	[8.675418651,-0.242068655,1],
#    	[7.673756466,3.508563011,1]]

    dataset = [[1,1,0],
            [1,0,1],
            [0,1,1],
            [0,0,0]]   
    n_inputs = len(dataset[0]) - 1
    n_outputs = len(set([row[-1] for row in dataset]))
    network = initialize_network(n_inputs, 2, n_outputs)
    train_network(network, dataset, 0.5, 20000, n_outputs)
    for layer in network:
        print(layer)
    for row in dataset:
        prediction = predict(network, row)
        print('预期值=%d, 实际输出值=%d' % (row[-1], prediction))


