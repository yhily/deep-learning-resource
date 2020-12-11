# -*- coding: utf-8 -*-
"""
Created on Sat Feb  1 15:47:28 2020

@author: Yuhong

E-mail： zhangyuhong001@gmail.com

"""

# 使用多层感知机实现多分类n
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense

# 导入数据
df = pd.read_csv('iris.csv', header=None)
# 分割特征X和标签y
X, y = df.values[:, :-1], df.values[:, -1]
#读取鸢尾花的分类
iris_class = np.unique(df.values[:, -1])

#将特征由字符串转换为浮点数
X = X.astype('float32')
# 将y字符串标签转换标签编码: array([0, 1, 2])
y = LabelEncoder().fit_transform(y) #
# 将数据集合分割为测试集合（30%）和训练集合（70%），借用sklearn的分割函数
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = .3)
#输出验证，非必需
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

#获取特征数
n_features = X_train.shape[1]
# 搭建模型
model = Sequential()
#增加一个全连接层
model.add(Dense(10, activation='relu', kernel_initializer='he_normal', input_shape=(n_features,)))
#再增加一个全连接层
model.add(Dense(8, activation='relu', kernel_initializer='he_normal'))
#再增加一个全连接层：输出层，分类数为3
model.add(Dense(3, activation='softmax'))
# 编译模型，设置优化器，损失函数和性能标准
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
# 拟合模型：即模型的训练
model.fit(X_train, y_train, epochs=150, batch_size=32, verbose=0)
# 评估模型：在测试集合上进行
loss, acc = model.evaluate(X_test, y_test, verbose=0)
print('测试集合准确率: {0:.3f}'.format(acc))
# 单个样本预测
row = [5.1,3.5,1.4,0.2]
yhat = model.predict([row])
print('预测样本分类为: {0}'.format(iris_class[(np.argmax(yhat))]))

'''
输出结果为：
测试集合准确率: 0.978
预测样本分类为: Iris-setosa
'''