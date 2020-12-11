#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 23 16:54:14 2020

@author: yhily
"""

import tensorflow as tf

a = tf.constant(4, name = "a")
b = tf.constant(2, name = "b")
c = tf.math.multiply(a, b, name ="c")
d = tf.math.add(a, b, name = "d")
e = tf.math.add(c,d, name = "e")
 
print(e)