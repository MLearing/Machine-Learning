# -*- coding:utf-8 -*-

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import scipy.optimize as opt
from costFunction import costFunction
from gradient import *
from predict import predict
from plotDecisionBoundary import plot_decision_boundary
from mapFeature import map_feature
from gaussianKernel import gaussian_kernel

# getcwd()返回当前进程的工作目录
path = os.getcwd() + '\ex2data2.txt'
# data为DataFrame类型
data = pd.read_csv(path, header=None, names=['Exam 1', 'Exam 2', 'Admitted'])
# print(data.head())

# add a ones column - this makes the matrix multiplication work out easier
if (os.path.split(path)[-1]=='ex2data1.txt'):
    data.insert(0, 'Ones', 1)

# set X (training data) and y (target variable)
cols = data.shape[1]
X = data.iloc[:,0:cols-1]
y = data.iloc[:,cols-1:cols]

# convert to numpy arrays and initalize the parameter array theta
X = np.array(X.values)

if (os.path.split(path)[-1]=='ex2data2.txt'):
    X = map_feature(X[:,0],X[:,1])

y = np.array(y.values)
theta = np.zeros(X.shape[1])

result = opt.fmin_tnc(func=costFunction, x0=theta, fprime=batch_gradient_descent, args=(X, y), messages=0)
theta_min = np.matrix(result[0])
predictions = predict(theta_min, X)

# 绘图
plot_decision_boundary(theta_min,X,y)

# zip() 函数用于将可迭代的对象作为参数，将对象中对应的元素打包成一个个元组，然后返回由这些元组组成的列表。
# map() 函数接收两个参数，一个是函数，一个是序列，map将传入的函数依次作用到序列的每个元素，并把结果作为新的list返回。
correct = [1 if ((a == 1 and b == 1) or (a == 0 and b == 0)) else 0 for (a, b) in zip(predictions, y)]  # ()表示元组
accuracy = ((sum(map(int, correct)) / len(correct)))*100
print ('accuracy = {0}%'.format(round(accuracy,2)))


