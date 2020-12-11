# -*- coding: utf-8 -*-
"""
Created on Fri Feb  7 07:00:51 2020

@author: Yuhong

E-mail： zhangyuhong001@gmail.com

"""
import tensorflow as tf

def gradient_test():
    #-------------------一元梯度案例---------------------------
    print("一元梯度")
    x=tf.constant(value=3.0)
    with tf.GradientTape(persistent=True,watch_accessed_variables=True) as tape:
        tape.watch(x)
        y1=2*x
        y2=x*x+2
        y3=x*x+2*x
    #一阶导数
    dy1_d x= tape.gradient(target=y1,sources=x)
    dy2_dx = tape.gradient(target=y2, sources=x)
    dy3_dx = tape.gradient(target=y3, sources=x)
    print("dy1_dx:",dy1_dx)
    print("dy2_dx:", dy2_dx)
    print("dy3_dx:", dy3_dx)

if __name__=="__main__":
    gradient_test()
    
    
'''
X = tf.constant([4.,        #张量0：相当于函数中的x0初始值
                 4.         #张量1：相当于函数中的x1初始值
                 ])

for step in range(200):
    with tf.GradientTape() as tape:
        tape.watch([X])
        y = fun(X)

    grads = tape.gradient(y, [X])[0] 
    X -= 0.01*grads

    
    if step % 20 == 0:
        print ('step {}: X = {}, f(x) = {}'
               .format(step, X.numpy(), y.numpy()))

'''