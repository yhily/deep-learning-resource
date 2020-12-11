# -*- coding: utf-8 -*-
"""
Created on Wed Feb  5 19:08:01 2020

@author: Yuhong

E-mail： zhangyuhong001@gmail.com

"""
# 载入模块
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
#定义函数
def fun(x,y):
    return -(np.power(x,2)+np.power(y,2))

# 创建 3D 图形对象
fig = plt.figure()
ax = Axes3D(fig)
 
# 生成数据
X0 = np.arange(-3, 3, 0.25)
X1 = np.arange(-3, 3, 0.25)
X0, X1 = np.meshgrid(X0, X1)
Z = fun(X0, X1)

# 绘制曲面图，并使用 cmap 着色
ax.plot_surface(X0, X1, Z, cmap=plt.cm.winter)
ax.set_xlabel('X0', color='r')
ax.set_ylabel('X1', color='g')
ax.set_zlabel('f(x)', color='b')#给三个坐标轴注明坐标名称
plt.show()
