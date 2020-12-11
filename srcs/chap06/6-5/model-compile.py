# -*- coding: utf-8 -*-
"""
Created on Thu Jan 30 21:49:54 2020

@author: Yuhong

E-mail： zhangyuhong001@gmail.com

"""

import numpy as np
from tensorflow.keras.datasets import load_data
import tensorflow as tf
from tensorflow import keras


def preprocess(x, y): # 自定义的预处理函数
    # 调用此函数时会自动传入x,y 对象，shape 为[b, 28, 28], [b]
    # 标准化到0~1
    x = tf.cast(x, dtype=tf.float32) / 255.
    x = tf.reshape(x, [-1, 28*28]) # 打平
    y = tf.cast(y, dtype=tf.int32) # 转成整型张量
    y = tf.one_hot(y, depth=10) # one-hot 编码
    # 返回的x,y 将替换传入的x,y 参数，从而实现数据的预处理功能
    return x,y


# 导入数据
(X_train,y_train), (X_test, y_test) = load_data()

# 获取图片的大小
in_shape = X_train.shape[1:]    # 形状为(28, 28)
# 获取数字图片的种类
n_classes = len(np.unique(y_train)) #类别数为10

#数据预处理，将0~255缩放到0~1范围
x_train = X_train.astype('float32') / 255.0
x_test = X_test.astype('float32') / 255.0

# 定义模型
#model = keras.Sequential([
#    keras.layers.Flatten(input_shape=in_shape),
#    keras.layers.Dense(n_classes, activation='softmax')
#])

model = keras.Sequential()  #搭建空顺序模型
model.add( keras.layers.Flatten(input_shape=in_shape))
model.add( keras.layers.Dense(n_classes, activation='softmax'))

#编译模型：定义损失函数和优化函数
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
