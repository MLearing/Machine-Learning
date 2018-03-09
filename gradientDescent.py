import numpy as np
from computeCost import computeCost
import random

def batch_gradient_descent(X, y, theta, alpha, iters,lamda=0):
    m=len(X)
    temp = np.matrix(np.zeros(theta.shape))
    parameters = int(theta.ravel().shape[1])  # np.ravel()把多维变为一维行数组（默认行优先）
    cost = np.zeros(iters)  # 生成一维0数组（一维数组是行）

    for i in range(iters):
        error = (X * theta.T) - y  # 对于矩阵来说*为叉乘

        for j in range(parameters):
            term = (alpha / m )* np.sum(np.multiply(error, X[:,j])) #正常项
            term_r= ((alpha * lamda) / m ) * theta[0,j]  #正则项
            temp[0,j] = theta[0,j] - term - term_r

        theta = temp
        cost[i] = computeCost(X, y, theta,lamda)

    return theta, cost


def stochastic_gradient_descent(X, y, theta, alpha, iters,lamda=0):
    eps =0.0001
    iter_count = 0
    loss=10

    temp = np.matrix(np.zeros(theta.shape))
    parameters = int(theta.ravel().shape[1])
    cost = np.zeros(iters)

    # and运算符：只要左边的表达式为真，整个表达式返回的值是右边表达式的值，否则，返回左边表达式的值
    while( loss > eps and iter_count < iters):
        k=random.randint(0,len(X)-1)
        error = (X[k,:] * theta.T) - y[k]

        for j in range(parameters):
            term = alpha * np.multiply(error, X[k,j])
            term_r= (alpha * lamda) * theta[0,j]  #正则项
            temp[0,j] = theta[0,j] - term - term_r
        theta = temp

        loss = 0.5*((X[k,:] * theta.T) - y[k])**2
        cost[iter_count] = computeCost(X, y, theta)
        iter_count += 1

    return theta, cost


def mini_batch_gradient_descent(X, y, theta, alpha, iters, lamda=0):
    temp = np.matrix(np.zeros(theta.shape))
    parameters = int(theta.ravel().shape[1])  # np.ravel()把多维变为一维行数组（默认行优先）
    cost = np.zeros(iters)  # 生成一维0数组（一维数组是行）

    for i in range(iters):
        k=random.randint(0,len(X)-1)
        error = (X[k:len(X),:] * theta.T) - y[k:len(X),:]  # 对于矩阵来说*为叉乘
        m=len(X)-k

        for j in range(parameters):
            term = (alpha / m) * np.sum(np.multiply(error, X[k:len(X),j]))
            term_r= ((alpha * lamda) / m ) * theta[0,j]  #正则项
            temp[0,j] = theta[0,j] - term - term_r

        theta = temp
        cost[i] = computeCost(X, y, theta)

    return theta, cost

