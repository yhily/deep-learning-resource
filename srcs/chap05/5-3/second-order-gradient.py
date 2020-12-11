# -*- coding: utf-8 -*-
"""
Created on Fri Feb  7 20:14:45 2020

@author: Yuhong

E-mail： zhangyuhong001@gmail.com

"""

import tensorflow as tf

x = tf.Variable(2.0, trainable=True)
with tf.GradientTape() as tape1:
    with tf.GradientTape() as tape2:
        y = x ** 2 + 2 * x - 1
    order_1 = tape2.gradient(y, x)
order_2 = tape1.gradient(order_1, x)
 
print("在x = 2处的一阶梯度为：", order_1.numpy())
print("在x = 2处的二阶梯度为：", order_2.numpy())