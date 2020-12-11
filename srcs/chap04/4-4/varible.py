# -*- coding: utf-8 -*-
"""
Created on Mon Jan 20 08:42:50 2020

@author: Yuhong

E-mail： zhangyuhong001@gmail.com

测试变量与常量

"""

import tensorflow as tf

my_state    = tf.Variable(1.0)
one         = tf.constant(1.0)

read_and_increment = tf.function(lambda: my_state.assign_add(one)) 

print(my_state)
for _ in range(3):
    read_and_increment()
    tf.print(my_state.numpy())

'''
输出结果：
<tf.Variable 'Variable:0' shape=() dtype=float32, numpy=1.0>
2.0
3.0
4.0

'''