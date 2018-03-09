# -*- coding:utf-8 -*-

"""
@english_name: locally weighted linear regression
@chinese_name:局部加权线性回归
"""

import numpy as np

# 对某一点计算估计值
def lwlr(testPoint, X, y, k = 1.0):
    m = np.shape(X)[0]
    weights = np.mat(np.eye((m))) # np.eye()生成对角矩阵
    for i in range(m):
        diffMat = testPoint - X[i, :]
        weights[i, i] = np.exp(diffMat * diffMat.T/(-2.0 * k**2))      # 计算权重对角矩阵
    xTx = X.T * (weights * X)                                 # 奇异矩阵不能计算
    if np.linalg.det(xTx) == 0.0:  # 矩阵求行列式
        print('This Matrix is singular, cannot do inverse')
        return
    theta = xTx.I * (X.T * (weights * y))                     # 计算回归系数
    return testPoint * theta

# 对所有点计算估计值
def lwlrTest(X, y, k = 1.0):  # k为波长参数，它控制了权值随距离下降的速率
    m = np.shape(X)[0]
    yHat = np.zeros(m)
    for i in range(m):
        yHat[i] = lwlr(X[i], X, y, k)   # X[i]表示第i行所有列
    return yHat
