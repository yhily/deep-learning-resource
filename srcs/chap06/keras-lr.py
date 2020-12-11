# -*- coding: utf-8 -*-
"""
Created on Sun Feb  9 20:46:51 2020

@author: Yuhong

E-mail： zhangyuhong001@gmail.com

"""
import pandas as pd
from tensorflow import keras
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from matplotlib import pyplot
#获取训练数据
data = pd.read_csv('winequality-white.csv')
# 将数据集合中的每个（列）属性都规整化到0-1
scaler = MinMaxScaler()
max_min_data = scaler.fit_transform(data) 
#分割数据为训练集合和测试集合
X = max_min_data[:,:-1]
y = max_min_data[:, -1]
X_train, X_test, y_train, y_test =  train_test_split(X, y, test_size = 0.3, random_state = 0) 

# 定义模型
model = keras.Sequential()               #搭建空顺序模型
model.add( keras.layers.Dense(units = 1,  #输出向量的尺寸
        kernel_initializer = 'uniform',  #权值初始化参数
        input_shape= (len(X[0]),),          #输入向量尺寸  
        activation = lambda x : x,     #线性激活函数
        use_bias = True
        ))
#编译模型
from tensorflow.keras.optimizers import SGD
opt = SGD(lr=0.01, momentum=0.9)  #构建个性化的随机梯度递减

model.compile(optimizer = opt,
              loss='mse',      #均方差损失函数
              metrics = ['mse','mae']) 
#训练模型
history = model.fit(X_train, y_train, 
                    epochs = 100, 
                    batch_size = len(X_train), 
                    verbose = 2)
#绘制训练曲线
pyplot.plot(history.history['mse'])
pyplot.plot(history.history['mae'])
pyplot.show()

#模型预测
for i in range(10):
    pred = model.predict([[X_test[i]]])
    print("期望值 = {0:.2f}, 预测值 = {1:.2f}".format(y_test[i], pred[0][0]))


