#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  6 22:40:45 2020

@author: yhily
"""

import numpy as np
from tensorflow.keras.models import model_from_json

"""---------载入已经训练好的模型---------"""

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


import os
import cv2

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
 



"""---------用opencv载入一张待测图片-----"""
# 载入图片
src = cv2.imread('original.png')
cv2.imshow("待测图片", src)
