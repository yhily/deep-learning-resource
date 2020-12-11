#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar  8 10:44:41 2020

@author: yhily
"""

import tensorflow as tf
#构建模型
W1 = tf.Variable(tf.random.uniform([2,20],-1,1))
B1 = tf.Variable(tf.random.uniform([  20],-1,1))
W2 = tf.Variable(tf.random.uniform([20,1],-1,1))
B2 = tf.Variable(tf.random.uniform([   1],-1,1))

@tf.function
def predict(X):
    X = tf.convert_to_tensor(X, tf.float32)
    H1  = tf.nn.leaky_relu(tf.matmul(X,W1) + B1)
    pre = tf.sigmoid(tf.matmul(H1,W2) + B2)
    return pre

def fit(X, y):
    Optim = tf.keras.optimizers.SGD(1e-1)
    num_iter = 10000
    y_true = tf.convert_to_tensor(y, tf.float32)

    for step in range(num_iter):
        if step%(num_iter/10)==0:
            y_pre  = predict(X)
            loss = tf.reduce_mean(tf.square(y_true - y_pre))
            print(step, " Loss:", loss.numpy())

        with tf.GradientTape() as tape:
            y_pre  = predict(X)
            Loss = tf.reduce_mean(tf.square(y_true - y_pre))
            #自动求导
            Grads = tape.gradient(Loss,[W1,B1,W2,B2])
            # 反向传播并更新权值
            Optim.apply_gradients(zip(Grads,[W1,B1,W2,B2]))

if __name__ == '__main__':
    # 构建数据
    X = [[0, 0], [0, 1], [1, 0], [1, 1]]
    y = [[0], [1], [1], [0]]
    fit(X, y)
    pre = predict(X)
    print("预测值： ", pre)


