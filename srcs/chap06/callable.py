#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  3 10:19:03 2020

@author: yhily
"""

class Entity:

    def __init__(self, x, y):
        self.x, self.y = x, y
    
    def __call__(self, x, y):
        '''改变对象内部的值'''
        self.x, self.y = x, y
        print("x = ",self.x, "y = ", self.y) #此为测试语句，非必需

obj = Entity(1, 2) #一个创建实例
obj(3, 4) #实例修改对象的x y 

Entity(1,2)(3,4)
