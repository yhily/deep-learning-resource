# -*- coding: utf-8 -*-
"""
Created on Mon Feb 10 21:00:56 2020

@author: Yuhong

E-mail： zhangyuhong001@gmail.com

"""

import numpy as np
from tensorflow.keras.datasets.mnist import load_data
from tensorflow.keras.models import model_from_json
# 导入数据
(X_train,y_train), (X_test, y_test) = load_data()
# 获取图片的大小
in_shape = X_train.shape[1:]    # 形状为(28, 28)
# 获取数字图片的种类
n_classes = len(np.unique(y_train)) #类别数为10

#数据预处理，将0~255缩放到0~1范围
x_train = X_train.astype('float32') / 255.0
x_test = X_test.astype('float32') / 255.0

#读入模型文件
json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
#反序列化：导入模型拓扑结构
loaded_model = model_from_json(loaded_model_json)
# 反序列化：将权值导入到加载的模型中
loaded_model.load_weights("model.h5")
print("成功：从本地文件中导入权值参数！")
#编译导入的模型
loaded_model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
# 测试模型是否可用
loss, acc = loaded_model.evaluate(x_test, y_test, verbose=0)
print('测试集合的预测准确率:{0:.2f} %'.format(acc * 100))
