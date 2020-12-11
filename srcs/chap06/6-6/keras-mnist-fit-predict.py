
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 27 12:58:57 2020

@author: Yuhong

E-mail： zhangyuhong001@gmail.com

利用Keras包来实现预测

"""

import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets.mnist import load_data
from tensorflow import keras

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
# 模型拟合
model.fit(x_train, y_train, epochs=10, batch_size=128, verbose=0)
# 评估模型
loss, acc = model.evaluate(x_test, y_test, verbose=0)
print('测试集合的预测准确率:{0:.3f}'.format(acc))

#单个图片预测
image = x_train[100]    #选择第101图做测试
import  tensorflow as tf 
image = tf.expand_dims(image, axis = 0)

yhat = model.predict([image])
print('预测的数字为:{0}'.format(np.argmax(yhat)))  
# 绘制像素数据
plt.title('label = {}'.format(y_train[0]))
#plt.imshow(x_train[100], cmap=plt.get_cmap('gray_r'))  #try: plt.get_cmap('gray'),黑底白字，#'seismic'彩色
plt.imshow(x_train[100], cmap=plt.get_cmap('gray')) 
#plt.axis('off')
# 显示图片
plt.show()