# -*- coding:utf-8 -*-

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from featureNormalize import *
from plotData import *
from constructor import Constructor


path = os.getcwd() + '\ex1data1.txt'
data = pd.read_csv(path, header=None, names=['Population', 'Profit'])

# 返回data的前几行数据，默认为前五行，需要前十行则data.head(10)
# print(data.head())
# 生成简要的统计信息
# print(data.describe())

# 使用pandas的DataFrame的plot方法绘制图像
# data.plot(kind='scatter', x='Population', y='Profit', figsize=(6,4))

# append a ones column to the front of the data set
data.insert(0, 'Ones', 1)   # （位置，列名，数据）

# set X (training data) and y (target variable)
cols = data.shape[1]
# iloc对列使用切片的方法对数据进行选取。切片之后类型依旧是dataframe，不能直接进行加减乘除等操作的
X = data.iloc[:,0:cols-1]
y = data.iloc[:,cols-1:cols]

# convert from data frames to numpy matrices
X = np.matrix(X.values)
y = np.matrix(y.values)
theta = np.matrix(np.ones((1,X.shape[1])))

# 特征归一化
mu,sigma,X_n = feature_normalize(X[:,1:X.shape[1]])
X_n=np.c_[X[:,0],X_n]

# initialize variables for learning rate and iterations
alpha = 0.01
iters = 1000

# 构造回归函数
x_all,f_all = Constructor(X, y, theta, alpha, iters, data, X_n)
# 画图
plot_data(x_all,f_all,X_n,data)

