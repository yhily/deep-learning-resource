#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 12 08:39:05 2020

@author: yhilly
"""

from sklearn.datasets import load_iris

#(1)导入数据集合
iris = load_iris()
X, y = iris.data, iris.target

#（2）分割数据集合：测试集合和训练结合
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 0)

#(3) 导入模型
from sklearn.linear_model import Perceptron
#（4）构建模型
#model = Perceptron(random_state=42, max_iter = 10, tol=0.001)
model = Perceptron()
#（5）训练模型
model.fit(X_train, y_train)

#（6）模型预测
y_predict_on_test = model.predict(X_test)

#（7）模型打分：测试集合的准确率为: 71.11%
from sklearn.metrics import accuracy_score
print('测试集合的准确率为: {:.2f}%'.format(100 * accuracy_score(y_test, y_predict_on_test )))

#思考：预测准确率并不高，如何改进？
