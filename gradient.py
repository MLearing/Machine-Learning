__author__ = 'Designer'

import numpy as np
from sigmoid import sigmoid
import random


def batch_gradient_descent(theta, X, y):
    theta = np.matrix(theta)
    X = np.matrix(X)
    y = np.matrix(y)

    parameters = int(theta.ravel().shape[1])  # ravel()����ά���齵Ϊһά
    grad = np.zeros(parameters)

    error = sigmoid(X * theta.T) - y
    for i in range(parameters):
        term = np.multiply(error, X[:,i])
        grad[i] = np.sum(term)

    return grad

def mini_batch_gradient_descent(theta, X, y):
    theta = np.matrix(theta)
    X = np.matrix(X)
    y = np.matrix(y)

    parameters = int(theta.ravel().shape[1])  # np.ravel()�Ѷ�ά��Ϊһά�����飨Ĭ�������ȣ�
    grad = np.zeros(parameters)
    k=random.randint(0,len(X)-1)
    error = sigmoid(X[k:len(X),:] * theta.T) - y[k:len(X),:]  # ���ھ�����˵*Ϊ���

    for i in range(parameters):
        term = np.multiply(error, X[k:len(X),i])
        grad[i] = np.sum(term)

    return grad

def stochastic_gradient_descent(theta, X, y):
    theta = np.matrix(theta)
    X = np.matrix(X)
    y = np.matrix(y)

    parameters = int(theta.ravel().shape[1])  # np.ravel()�Ѷ�ά��Ϊһά�����飨Ĭ�������ȣ�
    grad = np.zeros(parameters)
    k=random.randint(0,len(X)-1)

    error = sigmoid(X[k,:] * theta.T) - y[k]

    for i in range(parameters):
        term = np.multiply(error, X[k,i])
        grad[i] = term

    return grad












