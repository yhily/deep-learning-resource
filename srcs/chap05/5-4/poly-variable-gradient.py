# -*- coding: utf-8 -*-
"""
Created on Fri Feb  7 19:28:44 2020

@author: Yuhong

E-mail： zhangyuhong001@gmail.com

"""

import tensorflow as tf

def func(x):           #注意：此处x为一个多元张量
    return x[0] ** 2 + 3 * x[0] * x[1] + x[1] **2 +  x[2] ** 3

def gradient_test():    #求n（=3）元函数的梯度
    x = tf.constant(value = [1.0, 2.0, 3.0])
    with tf.GradientTape() as tape:
        tape.watch(x)
        y = func(x)
    #一阶导数
    order_1 = tape.gradient(target = y, sources = x)
    print("多元函数x = [1.0, 2.0, 3.0]处的梯度为：", order_1.numpy())

if __name__=="__main__":
    gradient_test()