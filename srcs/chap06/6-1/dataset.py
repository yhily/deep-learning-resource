#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 29 23:27:30 2020

@author: yhily
"""


import tensorflow as tf
import numpy as np

dataset = tf.data.Dataset.from_tensor_slices(
            np.array([1.0, 2.0, 3.0, 4.0, 5.0]))

for element in dataset: 
    print(element) #print(element.numpy())
    