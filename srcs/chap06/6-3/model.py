# -*- coding: utf-8 -*-
"""
Created on Sat May  2 16:49:23 2020

@author: Yuhong

E-mail： zhangyuhong001@gmail.com

"""

import numpy as np
import tensorflow as tf
from  tensorflow.keras import datasets, Sequential
from  tensorflow.keras import datasets, layers, Sequential

# 导入数据
(X_train,y_train), (X_test, y_test) = datasets.mnist.load_data()

X_train = tf.convert_to_tensor(X_train, dtype=tf.float32) / 255.
dataset = tf.data.Dataset.from_tensor_slices((X_train,y_train))
dataset = dataset.batch(32).repeat(10)


# 获取图片的大小
in_shape = X_train.shape[1:]    # 形状为(28, 28)
# 获取数字图片的种类
n_classes = len(np.unique(y_train)) #类别数为10

#数据预处理，将0~255缩放到0~1范围
#x_train = X_train.astype('float32') / 255.0
#x_test = X_test.astype('float32') / 255.0

# 定义模型
#model = keras.Sequential([
#    keras.layers.Flatten(input_shape=in_shape),
#    keras.layers.Dense(n_classes, activation='softmax')
#])

model = Sequential()  #搭建空顺序模型
model.add( layers.Flatten(input_shape=in_shape))
model.add( layers.Dense(n_classes, activation='softmax'))



