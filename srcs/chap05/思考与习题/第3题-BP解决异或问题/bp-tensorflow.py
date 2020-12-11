#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar  8 10:44:41 2020

@author: yhily
"""

import tensorflow as tf

#构建数据
X = [[0,0],[0,1],[1,0],[1,1]]
Y = [[0],  [1],  [1],  [0]  ]
X = tf.convert_to_tensor(X,tf.float32)
Y = tf.convert_to_tensor(Y,tf.float32)

#model
W1 = tf.Variable(tf.random.uniform([2,20],-1,1))
B1 = tf.Variable(tf.random.uniform([  20],-1,1))

W2 = tf.Variable(tf.random.uniform([20,1],-1,1))
B2 = tf.Variable(tf.random.uniform([   1],-1,1))

@tf.function
def feedforward(X):
    H1  = tf.nn.leaky_relu(tf.matmul(X,W1) + B1)
    Out = tf.sigmoid(tf.matmul(H1,W2) + B2)
    return Out


#适用随机梯度训练模型
Optim = tf.keras.optimizers.SGD(1e-1)
Steps = 10000

for I in range(Steps):
    if I%(Steps/100)==0:
        Out  = feedforward(X)
        Loss = tf.reduce_mean(tf.square(Y-Out))
        print("损失误差: ",I, Loss.numpy())
    #end if

    with tf.GradientTape() as T:
        Out  = feedforward(X)
        Loss = tf.reduce_mean(tf.square(Y-Out))
    #end with

    #BACKPROPAGATION HERE?
    Grads = T.gradient(Loss,[W1,B1,W2,B2])
    Optim.apply_gradients(zip(Grads,[W1,B1,W2,B2]))
#end for

Out  = feedforward(X)
Loss = tf.reduce_sum(tf.square(Y - Out))
print("损失误差:",Loss.numpy(),"(Last)")

def predict(row):
    X = tf.convert_to_tensor(row,tf.float32);
    outputs = feedforward(X)
    print(row,'\n', outputs.numpy())


predict(X)

   
#(5)输出结果
'''
Loss:  0 0.3684249
Loss:  100 0.22871631
Loss:  200 0.16743718
Loss:  300 0.120626494
Loss:  400 0.08156897
Loss:  500 0.053753678
Loss:  600 0.036274727
Loss:  700 0.025548674
Loss:  800 0.01889082
Loss:  900 0.014633151
Loss:  1000 0.011764675
Loss:  1100 0.009714609
Loss:  1200 0.008180942
L....

测试结果:
predict(X)

tf.Tensor(
[[0. 0.]
 [0. 1.]
 [1. 0.]
 [1. 1.]], shape=(4, 2), dtype=float32) 
 [[0.01796994]
 [0.9831369 ]
 [0.9826642 ]
 [0.01726511]]



predict([[0., 1.]])

[[0.0, 1.0]] 
 [[0.97821474]]
 

predict([[0., 0.]])
[[0.0, 0.0]] 
 [[0.02044367]]
 

'''

