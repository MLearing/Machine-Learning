# -*- coding:utf-8 -*-
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from LWLR import lwlrTest

path = os.getcwd() + '\ex0.txt'
data = pd.read_csv(path, header=None, names=['ones','Population', 'Profit'])
# print(data.head())

cols = data.shape[1]

X = data.iloc[:,0:cols-1]
y = data.iloc[:,cols-1:cols]

# # convert from data frames to numpy matrices
X = np.matrix(X.values)
y = np.matrix(y.values)

yHat = lwlrTest(X, y, 0.01)
# argsort()函数是将x中的元素从小到大排列，提取其对应的index(索引),axis=0按列排序
strInd = X[:, 1].argsort(0)
# 三维矩阵  第一位先提取三维的行，提取后仍为三维矩阵,然后按照三维矩阵方式提取
xSort = X[strInd][:,0,:]

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.plot(xSort[:, 1], yHat[strInd][:,0])

# X.flatten()就是把A降到一维，默认是按横的方向降
# X.flatten().A又是什么呢? 其实这是因为此时的X是个矩阵，降维后还是个矩阵，矩阵.A（等效于矩阵.getA()）变成了数组
ax.scatter(X[:, 1].flatten().A[0], y.T.flatten().A[0], s = 2, c = 'red')
plt.show()
