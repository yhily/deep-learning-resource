# -*- coding: utf-8 -*-
"""
Created on Wed Feb  5 19:18:17 2020


代码参考了斋藤康毅所著的《深度学习入门》。

E-mail： zhangyuhong001@gmail.com

"""
import numpy as np
import matplotlib.pylab as plt

def _numerical_gradient_no_batch(func, x):
    h = 1e-4 # 设delta = 0.0001
    grad = np.zeros_like(x) #生成与x维度相同的数组，并初始化为0.即计算x中每个元素的梯度
    
    for idx in range(x.size):
        tmp_val = x[idx]
        x[idx] = float(tmp_val) + h
        func_x_h1 = func(x) # 计算f(x+h)
        
        x[idx] = tmp_val - h 
        func_x_h2 = func(x) # 计算f(x-h)
        #计算梯度
        grad[idx] = (func_x_h1 - func_x_h2) / (2 * h)
        
        x[idx] = tmp_val # 还原X的值
        
    return grad

def numerical_gradient(func, X):
    if X.ndim == 1:
        return _numerical_gradient_no_batch(func, X)
    else:
        grad = np.zeros_like(X)
        
        for idx, x in enumerate(X):
            grad[idx] = _numerical_gradient_no_batch(func, x)
        
        return grad

def function_2(x):
    if x.ndim == 1:
        return np.sum(x ** 2 ) #x0^2
    else:
        return np.sum(x ** 2, axis = 1)# x0^2+x1^2+...
    
if __name__ == '__main__':
    x0 = np.arange(-3, 3, 0.25)
    x1 = np.arange(-3, 3, 0.25)
    X, Y = np.meshgrid(x0, x1)   #  meshgrid用于生成网格采样点的函数。
    
    X1 = X.flatten()
    Y1 = Y.flatten()
    
    grad = numerical_gradient(function_2, np.array([X1, Y1]) )
    
    plt.figure()
    #quiver绘制表示梯度变化的图中非常有用
    plt.quiver(X1, Y1, -grad[0], -grad[1],  angles="xy",headwidth = 10,color = "#444444")
    plt.xlim([-3, 3])
    plt.ylim([-3, 3])
    plt.xlabel('x0')
    plt.ylabel('x1')
    plt.grid()
    plt.draw()
    plt.show()