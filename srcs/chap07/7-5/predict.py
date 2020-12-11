#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 14 00:08:43 2020
'''
python 3.7
tensorflow 2.1
pillow(PIL) 4.3.0

代码参考：https://github.com/geektutu/tensorflow-tutorial-samples


"""

from PIL import Image
import numpy as np

from mnist_cnn import CNN

class Predict(object):
    def __init__(self):
#        latest = tf.train.latest_checkpoint('./ckpt')
        self.network = CNN()
        # 恢复网络权重
        #self.network.model.load_weights(latest)
        self.network.model.load_weights('./ckpt/cp-0004.ckpt')

    def predict(self, image_path):
        # 以黑白方式读取图片
        img = Image.open(image_path).convert('L')
        flatten_img = np.reshape(img, (28, 28, 1))
        x = np.array([1 - flatten_img ])
#        print(x)

        # API refer: https://keras.io/models/model/
        y = self.network.model.predict(x)

        # 因为x只传入了一张图片，取y[0]即可
        # np.argmax()取得最大值的下标，即代表的数字
        print(image_path)
        print(y[0],' -> 预测数字为：', np.argmax(y[0]))


if __name__ == "__main__":
    app = Predict()
    app.predict('./test_images/0_57.png')
    app.predict('./test_images/1_32.png')
    app.predict('./test_images/3_59.png')
