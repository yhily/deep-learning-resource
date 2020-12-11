# -*- coding: utf-8 -*-
"""
Created on Sat Feb  8 19:11:36 2020

@author: Yuhong

E-mail： zhangyuhong001@gmail.com

"""

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
#获取训练数据
data = pd.read_csv('winequality-white.csv')
# 将数据集合中的每个（列）属性都规整化到0-1
scaler = MinMaxScaler()
max_min_data = scaler.fit_transform(data) 
#分割数据为训练集合和测试集合
X = max_min_data[:,:-1]
y = max_min_data[:, -1]
X_train, X_test, y_train, y_test =  train_test_split(X, y, test_size = 0.3, random_state = 0) 

# 给待训练得到的两个参数进行随机初始化 
w1 = tf.Variable(tf.zeros(len(X_train[0]), dtype = tf.float64), trainable=True)
w0 = tf.Variable(tf.zeros(1, dtype = tf.float64), trainable=True)

# 定义损失函数
def loss(y_true, y_pred):
    return tf.keras.losses.MSE(y_true,y_pred)
    
def step_gradient(real_x, real_y, learning_rate):
    with tf.GradientTape(persistent=True) as tape:
        # 模型预测
        pred_y = tf.reduce_sum(real_x * w1) + w0
        # 计算损失
        reg_loss = loss(real_y, pred_y)    
    # 计算梯度
    w1_gradients, w0_gradients = tape.gradient(reg_loss, (w1, w0))
    # 更新权值
    w1.assign_sub(w1_gradients * learning_rate)
    w0.assign_sub(w0_gradients * learning_rate)

if __name__ == '__main__': 
    learning_rate = 0.01
    num_iterations = 10
    
    for _ in range(num_iterations):
        for index in range(len(X_train)):
            step_gradient(X_train[index], y_train[index], learning_rate)

    trained_w1 = w1.numpy()
    trained_w0 = w0.numpy()
    # 打印训练获得的权重
    print ("weights = {0}, bias = {1}".format(trained_w1, trained_w0))
    # 测试
    for i in range(10):
        pred = np.dot(trained_w1, X_test[i]) + trained_w0
        print("期望值 = {0}, 预测值 = {1}".format(y_test[i], pred))
        