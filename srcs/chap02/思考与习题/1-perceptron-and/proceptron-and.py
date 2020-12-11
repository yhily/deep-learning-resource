#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 12 09:00:35 2020

@author: yhilly
"""
import numpy as np
#(1) 导入模型
from sklearn.linear_model import Perceptron

#（2）构造所需训练数据

X = np.array([[-1, 1, 1],     #-1,表示的是哑元，下同
              [-1, 0, 0], 
              [-1, 1, 0], 
              [-1, 0, 1]])

y = np.array([1, 
              0, 
              0, 
              0])
#(3)构建模型,注意，迭代次数不能太少，否则训练不充分
model = Perceptron(random_state=42, max_iter = 100, tol=0.001)
#（4）训练模型
model.fit(X, y)

#（4）模型预测
#注意：预测向量，必须写成二维数组形式
print ('1 and 1 = %d' % model.predict([[-1, 1, 1]]))
print ('0 and 0 = %d' %  model.predict([[-1, 0, 0]]))
print ('1 and 0 = %d' %  model.predict([[-1, 1, 0]]))
print ('0 and 1 = %d' %  model.predict([[-1, 0, 1]]))

#输出结果
'''
1 and 1 = 1
0 and 0 = 0
1 and 0 = 0
0 and 1 = 0
'''