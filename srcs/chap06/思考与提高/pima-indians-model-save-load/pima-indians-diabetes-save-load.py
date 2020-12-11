# -*- coding: utf-8 -*-
"""
Created on Mon Feb 10 22:21:19 2020

@author: Yuhong

E-mail： zhangyuhong001@gmail.com

"""

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import model_from_json


import pandas as pd
# 导入数据
data = pd.read_csv('pima-indians-diabetes.csv',header = None)
# 分割特征X和标签y
X, y = data.values[:, :-1], data.values[:, -1]

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

# 构建模型
model = Sequential()
model.add(Dense(12,activation='relu', kernel_initializer='he_normal', input_shape=(n_features,)))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
# 编译模型
model.compile(loss='binary_crossentropy', 
              optimizer='adam', 
              metrics=['accuracy'])
# 训练模型
model.fit(X_train, y_train, epochs=150, batch_size=10, verbose=0)
# 评估模型
scores = model.evaluate(X_test, y_test, verbose=0)
print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

# 序列化网络拓扑文件 JSON
model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
# 序列化权值文件 HDF5
model.save_weights("model.h5")
print("模型存储至本地！")

# 随后...可分割为两个不同文件

# 导入网络拓扑文件
with open('model.json', 'r') as json_file:
    loaded_model_json = json_file.read()

loaded_model = model_from_json(loaded_model_json)
# 导入权值到模型中
loaded_model.load_weights("model.h5")
print("从磁盘中导入模型!")
# evaluate loaded model on test data
loaded_model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
scores = model.evaluate(X_test, y_test, verbose=0)
print("%s: %.2f%%" % (loaded_model.metrics_names[1], scores[1]*100))

'''
运行结果：
accuracy: 67.53%
模型存储至本地！
从磁盘中导入模型!
accuracy: 67.53%

'''