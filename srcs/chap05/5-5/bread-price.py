# -*- coding: utf-8 -*-
"""
Created on Fri Feb  7 21:27:13 2020

@author: Yuhong

E-mail： zhangyuhong001@gmail.com

"""

import numpy as np
import random
import tensorflow as tf
# 定义损失函数
def loss(y_true, y_pred):
    #return tf.square(tf.subtract(real_y, pred_y))
#    return tf.abs(real_y - pred_y)
    return tf.keras.losses.MSE(y_true,y_pred)

# 生成训练数据
x_train_inch = np.asarray([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11])
y_train_price = np.asarray([num * 10 + 5 for num in x_train_inch]) # y = 10x+5

# 给待训练得到的两个参数进行随机初始化 
w1 = tf.Variable(random.random(), trainable=True)
w0 = tf.Variable(random.random(), trainable=True)

def step_gradient(real_x, real_y, learning_rate):
    with tf.GradientTape(persistent=True) as tape:
        # 模型预测
        pred_y = w1 * real_x + w0
        # 计算损失
        reg_loss = loss(real_y, pred_y)    
    # 计算梯度
    w1_gradients, w0_gradients = tape.gradient(reg_loss, (w1, w0))
    # 更新权值
    w1.assign_sub(w1_gradients * learning_rate)
    w0.assign_sub(w0_gradients * learning_rate)


if __name__ == '__main__': 
    learning_rate = 0.01
    num_iterations = 1000
    
    for _ in range(num_iterations):
        step_gradient(x_train_inch, y_train_price, learning_rate)

    print(f'拟合得到的模型近似为： y ≈ {w1.numpy()}x + {w0.numpy()}')
    #预测
    wheat = 0.9
    price = w1 * wheat + w0
    print ("price = {0:.2f}".format(price))