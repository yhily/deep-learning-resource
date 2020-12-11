# -*- coding: utf-8 -*-
"""
Created on Mon Feb 10 21:00:56 2020

@author: Yuhong

E-mail： zhangyuhong001@gmail.com

"""

import numpy as np
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
print('测试集合的预测准确率:{0:.2f}%'.format(acc * 100))

# 将模型结构序列化为JSON格式
model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
#将模型权值序列化HDF5格式
model.save_weights("model.h5")
print("成功：将模型保存本地！")