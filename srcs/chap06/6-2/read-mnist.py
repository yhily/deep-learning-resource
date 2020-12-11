# -*- coding: utf-8 -*-
"""
Created on Mon Jan 27 09:36:20 2020

@author: Yuhong

E-mail： zhangyuhong001@gmail.com

读取并绘制MNIST图片

"""

#from tensorflow.keras.datasets.mnist import load_data
from tensorflow.keras import datasets # 导入经典数据集加载模块
import matplotlib.pyplot as plt
# 导入数据
(X_train,y_train), (X_test, y_test)  = datasets.mnist.load_data()
# summarize loaded dataset
print('训练集 : X={0}, y={1}'.format (X_train.shape, y_train.shape))
print('测试集 : X={0}, y={1}'.format(X_test.shape, y_test.shape))
# 绘制前4个图片
for i in range(4):
    # 定义子图
    plt.subplot(2, 2, i+1)
    # 绘制像素数据
    plt.title('label = {}'.format(y_train[i]))
    plt.imshow(X_train[i], cmap=plt.get_cmap('gray_r'))  #try: plt.get_cmap('gray'),黑底白字
    plt.axis('off')
# 显示图片
plt.show()
