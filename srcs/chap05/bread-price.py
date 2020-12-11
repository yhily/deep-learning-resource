# -*- coding: utf-8 -*-
"""
Created on Wed Oct 18 22:49:03 2017

@author: Yuhong
"""

bread_price = [[0.5,5],[0.6,5.5],[0.8,6],[1.1,6.8],[1.4,7]]
 
def step_gradient(b_current, w_current, points, learningRate):
    b_gradient = 0
    w_gradient = 0
    for i in range(len(points)):
        x = points[i][0]
        y = points[i][1]
        b_gradient += -1.0 * (y - ((w_current * x) + b_current))
        w_gradient += -1.0 * x * (y - ((w_current * x) + b_current))
    new_b = b_current - (learningRate * b_gradient)
    new_m = w_current - (learningRate * w_gradient)
    return [new_b, new_m]
 
def gradient_descent_runner(points, starting_b, starting_m, learning_rate, num_iterations):
    b = starting_b
    m = starting_m
    for i in range(num_iterations):
        b, m = step_gradient(b, m, points, learning_rate)
    return [b, m]
 
def predict(b, m, wheat):
    price = m * wheat + b
    return price

if __name__ == '__main__': 
   b1, m1 =  gradient_descent_runner(bread_price, 1, 1, 0.01, 100)
   
   price = predict(b1, m1, 0.9)
   print ("price = {0:.2f}".format(price))