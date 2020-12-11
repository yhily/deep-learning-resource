#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 12 09:53:11 2020

@author: yhilly
"""
import numpy as np
#（1）导入模型
from sklearn.neural_network import MLPClassifier
#（2）方案1：创建多层感知机分类器:注意，此时激活函数使用relu或tanh和solver（默认的adam）匹配
#model = MLPClassifier(hidden_layer_sizes=(4,2,), activation='tanh', 
#                     max_iter = 10000)

#方案2：不同的激活函数，使用不同的solver，需要不断尝试
# default = 'adam' and works well for large data sets.  
model = MLPClassifier(activation='logistic', max_iter=1000, 
              hidden_layer_sizes=(3,),
              solver='lbfgs')
#（3）构造所需训练数据

X = np.array([[1, 1],     
              [0, 0], 
              [1, 0], 
              [0, 1]])
y = np.array([0, 
              0, 
              1, 
              1])

#（4）训练模型
model.fit(X, y)

#（4）模型预测
print ('1 xor 1 = %d' % model.predict([[1, 1]]))
print ('0 xor 0 = %d' %  model.predict([[0, 0]]))
print ('1 xor 0 = %d' %  model.predict([[1, 0]]))
print ('0 xor 1 = %d' %  model.predict([[0, 1]]))

#(5)输出结果
'''
1 xor 1 = 0
0 xor 0 = 0
1 xor 0 = 1
0 xor 1 = 1
'''

