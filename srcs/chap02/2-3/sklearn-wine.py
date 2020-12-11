#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 27 15:10:51 2020

@author: yhily
"""

#(1)导入数据集
from sklearn.datasets import load_wine
wine = load_wine()
#（2）将特征和标签分开
X, y = wine.data, wine.target
#（3）分割数据集合：测试集合和训练结合
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = None)

#（4）数据预处理：数据缩放
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)
#（5）导入模型
from sklearn.neural_network import MLPClassifier
#（6）创建多层感知机分类器
#model = MLPClassifier(solver = "lbfgs",hidden_layer_sizes=(100,))
model = MLPClassifier(solver = "lbfgs",hidden_layer_sizes=(10,10))
#solver : {‘lbfgs’, ‘sgd’, ‘adam’}, default ‘adam’
#（7）训练模型
model.fit(X_train, y_train)
#（8）测试集合据预测
y_predict_on_train = model.predict(X_train)
y_predict_on_test = model.predict(X_test)
#（9）模型评估
from sklearn.metrics import accuracy_score
print('训练集合的准确率为: {:.2f}%'.format(100 * accuracy_score(y_train, y_predict_on_train)))
print('测试集合的准确率为: {:.2f}%'.format(100 * accuracy_score(y_test, y_predict_on_test )))


