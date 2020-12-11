# -*- coding: utf-8 -*-
"""
Created on Thu Aug 13 17:12:40 2020

@author: Yuhong

E-mail： zhangyuhong001@gmail.com

"""


#（1）读入数据

import os
import tensorflow as tf
from tensorflow.keras.datasets.mnist import load_data

# mnist数据集存储的位置，如何不存在将自动下载
data_path = os.path.abspath(os.path.dirname(__file__)) + '/data/mnist.npz'

class DataSource():
    def __init__(self):
        data_path = os.path.abspath( os.path.dirname (__file__)) + '/data/mnist.npz'
        (x_train, y_train), (x_test, y_test) = load_data (path=data_path)
        # (x_train, y_train), (x_test, y_test) = load_data ()
        # 增加一个通道
        x_train = x_train[..., tf.newaxis]
        x_test  = x_test[..., tf.newaxis]
        # 像素值缩放到 0~1 之间
        x_train, x_test = x_train / 255.0, x_test / 255.0
        self.train_images, self.train_labels = x_train, y_train
        
        self.test_images, self.test_labels = x_test, y_test


#测试：
data = DataSource()