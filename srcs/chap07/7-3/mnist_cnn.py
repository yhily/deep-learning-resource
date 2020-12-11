#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 12 12:56:10 2020

参考代码：https://www.tensorflow.org/tutorials/quickstart/advanced


"""

#（1）读入数据

import os
from tensorflow.keras.datasets.mnist import load_data
class DataSource():
    def __init__(self):
        # mnist数据集存储的位置，如何不存在将自动下载
        data_path = os.path.abspath(os.path.dirname(__file__)) + '/data/mnist.npz'
        (x_train, y_train), (x_test, y_test) = load_data(path=data_path)
        # 6万张训练图片，1万张测试图片
#        train_images = train_images.reshape((60000, 28, 28, 1))
#        test_images = test_images.reshape((10000, 28, 28, 1))
        x_train = x_train[..., tf.newaxis]
        x_test = x_test[..., tf.newaxis]
        # 像素值映射到 0 - 1 之间
        x_train, x_test = x_train / 255.0, x_test / 255.0

        self.train_images, self.train_labels = x_train, y_train
        self.test_images, self.test_labels = x_test, y_test
        
#测试：data = DataSource()
        
#(2)搭建模型
        
import tensorflow as tf
from tensorflow.keras import layers, models
class CNN():
    def __init__(self):
        model = models.Sequential()
        # 第1层卷积，卷积核大小为3*3，32个，28*28为待训练图片的大小
        model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
        model.add(layers.MaxPooling2D((2, 2)))
        # 第2层卷积，卷积核大小为3*3，64个
        model.add(layers.Conv2D(64, (3, 3), activation='relu'))
        model.add(layers.MaxPooling2D((2, 2)))
        # 第3层卷积，卷积核大小为3*3，64个
        model.add(layers.Conv2D(64, (3, 3), activation='relu'))
        # 增加一个平坦层，拉平数据
        model.add(layers.Flatten())
        model.add(layers.Dense(64, activation='relu'))
        model.add(layers.Dense(10, activation='softmax'))

        model.summary()

        self.model = model

#代码测试。network = CNN()

#（3）模型训练
class Train:
    def __init__(self):
        self.network = CNN()
        self.data = DataSource()

    def train(self):
        check_path = './ckpt/cp-{epoch:04d}.ckpt'
        # period 每隔5epoch保存一次
        save_model_cb = tf.keras.callbacks.ModelCheckpoint(check_path, 
                                                           save_weights_only=True, 
                                                           verbose = 1, save_freq='epoch')
                                                           #period=5)
        
        self.network.model.compile(optimizer='adam',
                                   loss='sparse_categorical_crossentropy', 
                                   metrics=['accuracy'])
        self.network.model.fit(self.data.train_images, 
                               self.data.train_labels, 
                               epochs=10, 
                               callbacks=[save_model_cb])
        
        test_loss, test_acc = self.network.model.evaluate(self.data.test_images, self.data.test_labels)
        print("准确率:{0:.2f}%，共测试了{1}张图片 " .format (test_acc * 100, len(self.data.test_labels)))

if __name__ == "__main__":
    mnist_train = Train()
    mnist_train.train()