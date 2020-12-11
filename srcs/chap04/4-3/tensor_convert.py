#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 29 07:57:24 2020

E-mail： zhangyuhong001@gmail.com

程序功能： 将Python数据源转换为张量

"""
import tensorflow as tf

a_list = list([1,2,3])
b_tuple = (11.0, 22.2, 33)
c_str_tuple = "a", "b", "c", "d"

tensor_a = tf.convert_to_tensor(a_list,dtype = tf.float32)
tensor_b = tf.convert_to_tensor(b_tuple)
tensor_c = tf.convert_to_tensor(c_str_tuple)
tensor_add = tf.math.add(tensor_a, tensor_b)

print(type(a_list))
print(type(tensor_a))

print(type(b_tuple))
print(type(tensor_b))

print(type(c_str_tuple))
print(type(tensor_c))

print(type(tensor_add))

#输出张量中的数据
tf.print(tensor_c)
tf.print(tensor_add)

