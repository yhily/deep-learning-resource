# -*- coding: utf-8 -*-
"""
Created on Thu Jan 16 17:09:13 2020

@author: Yuhong

E-mail： zhangyuhong001@gmail.com

张量的阶

"""

import tensorflow as tf

tensor_0 = 1                                           
tensor_1 = [b"Tensor", b"flow", b"is", b"great"] 
tensor_2 = [[False, True, False],               
             [True, True, False]]
tensor_3 = [[[0, 0, 0], [0, 0, 1]],              
             [[1, 0, 0], [1, 0, 1]],
             [[2, 0, 0], [2, 0, 1]]]

print("rank of tensor_0:{0}".format(tf.rank(tensor_0)))
print("rank of tensor_1:{0}".format(tf.rank(tensor_1)))
print("rank of tensor_2:{0}".format(tf.rank(tensor_2)))
print("rank of tensor_3:{0}".format(tf.rank(tensor_3)))
