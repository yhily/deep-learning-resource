#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May  2 16:49:23 2020

@author: yhily
"""
# -*- coding: utf-8 -*-
"""
Created on Sat May  2 16:49:23 2020

@author: Yuhong

E-mail： zhangyuhong001@gmail.com

"""

import numpy as np
import tensorflow as tf
from    tensorflow.keras import datasets, layers, optimizers, Sequential, metrics
# 导入数据
(X_train,y_train), (X_test, y_test) = datasets.mnist.load_data()

X_train = tf.convert_to_tensor(X_train, dtype=tf.float32) / 255.
dataset = tf.data.Dataset.from_tensor_slices((X_train,y_train))
dataset = dataset.batch(32).repeat(10)

# 获取图片的大小
in_shape = X_train.shape[1:]    # 形状为(28, 28)
# 获取数字图片的种类
n_classes = len(np.unique(y_train)) #类别数为10


model = Sequential()  #搭建空顺序模型
model.add( layers.Flatten(input_shape=in_shape))
model.add( layers.Dense(n_classes, activation='softmax'))
#设置优化器，学习率设置为0.01
optimizer = optimizers.SGD(lr=0.01)
#设置算法性能的评估标准
acc_meter = metrics.Accuracy()

for step, (x,y) in enumerate(dataset):

    with tf.GradientTape() as tape:
        # 计算模型输出
        out = model(x)
        # 将标签变成独热编码
        y_onehot = tf.one_hot(y, depth=10)
        #计算损失
        loss = tf.square(out - y_onehot)
        # 计算损失均值
        loss = tf.reduce_sum(loss) / 32


    acc_meter.update_state(tf.argmax(out, axis=1), y)

    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))


    if step % 200 == 0:
        print('step {0}, loss:{1:.3f}, acc:{2:.2f} %'.format(step, float(loss), 
              acc_meter.result().numpy() * 100))
        acc_meter.reset_states()

