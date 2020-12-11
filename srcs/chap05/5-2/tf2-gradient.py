# -*- coding: utf-8 -*-
"""
Created on Fri Feb  7 07:00:51 2020

@author: Yuhong

E-mail： zhangyuhong001@gmail.com

"""
import tensorflow as tf

def func(x):
    return x ** 2 + 2 * x -1   #1元函数

def gradient_test():
    x = tf.constant(value = 2.0)
    with tf.GradientTape() as tape:
        tape.watch(x)
        y = func(x)
    #一阶导数
    order_1 = tape.gradient(target = y, sources = x)
    print("函数 x ** 2 + 2 * x -1 在x = 2处的梯度为：", order_1.numpy())

if __name__=="__main__":
    gradient_test()
    
