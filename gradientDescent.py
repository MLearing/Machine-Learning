import numpy as np
from computeCost import computeCost
import random

def batch_gradient_descent(X, y, theta, alpha, iters,lamda=0):
    m=len(X)
    temp = np.matrix(np.zeros(theta.shape))
    parameters = int(theta.ravel().shape[1])  # np.ravel()�Ѷ�ά��Ϊһά�����飨Ĭ�������ȣ�
    cost = np.zeros(iters)  # ����һά0���飨һά�������У�

    for i in range(iters):
        error = (X * theta.T) - y  # ���ھ�����˵*Ϊ���

        for j in range(parameters):
            term = (alpha / m )* np.sum(np.multiply(error, X[:,j])) #������
            term_r= ((alpha * lamda) / m ) * theta[0,j]  #������
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

    # and�������ֻҪ��ߵı��ʽΪ�棬�������ʽ���ص�ֵ���ұ߱��ʽ��ֵ�����򣬷�����߱��ʽ��ֵ
    while( loss > eps and iter_count < iters):
        k=random.randint(0,len(X)-1)
        error = (X[k,:] * theta.T) - y[k]

        for j in range(parameters):
            term = alpha * np.multiply(error, X[k,j])
            term_r= (alpha * lamda) * theta[0,j]  #������
            temp[0,j] = theta[0,j] - term - term_r
        theta = temp

        loss = 0.5*((X[k,:] * theta.T) - y[k])**2
        cost[iter_count] = computeCost(X, y, theta)
        iter_count += 1

    return theta, cost


def mini_batch_gradient_descent(X, y, theta, alpha, iters, lamda=0):
    temp = np.matrix(np.zeros(theta.shape))
    parameters = int(theta.ravel().shape[1])  # np.ravel()�Ѷ�ά��Ϊһά�����飨Ĭ�������ȣ�
    cost = np.zeros(iters)  # ����һά0���飨һά�������У�

    for i in range(iters):
        k=random.randint(0,len(X)-1)
        error = (X[k:len(X),:] * theta.T) - y[k:len(X),:]  # ���ھ�����˵*Ϊ���
        m=len(X)-k

        for j in range(parameters):
            term = (alpha / m) * np.sum(np.multiply(error, X[k:len(X),j]))
            term_r= ((alpha * lamda) / m ) * theta[0,j]  #������
            temp[0,j] = theta[0,j] - term - term_r

        theta = temp
        cost[i] = computeCost(X, y, theta)

    return theta, cost

