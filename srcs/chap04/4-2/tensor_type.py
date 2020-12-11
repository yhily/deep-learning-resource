# -*- coding: utf-8 -*-
"""
Created on Thu Jan 16 20:02:31 2020

@author: Yuhong

E-mail： zhangyuhong001@gmail.com

程序功能： 将NumPy数组赋值给Tensor

"""

import tensorflow as tf
import numpy as np

n0 = np.array(20, dtype = np.int32)
n1 = np.array([b"Tensor", b"flow", b"is", b"great"])
n2 = np.array([[True, False, False],
                [False, True,False]], 
                dtype = np.bool)
tensor0D = tf.Variable(n0, name = "t_0")
tensor1D = tf.Variable(n1, name = "t_1")
tensor2D = tf.Variable(n2, name = "t_2")


print("tensor0D ： {0}".format(tensor0D))
print("tensor1D ： {0}".format(tensor1D))
print("tensor2D ： {0}".format(tensor2D))

#利用tf.print输出
tf.print(tensor0D)
tf.print(tensor1D)
tf.print(tensor2D)