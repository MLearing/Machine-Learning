# -*- coding:utf-8 -*-
__author__ = 'Designer'

import numpy as np
import matplotlib.pyplot as plt
from mapFeature import map_feature
from gaussianKernel import  gaussian_kernel

# 绘图
def plot_decision_boundary(theta,X,y):
    fig, ax = plt.subplots(figsize=(6,4))

    # 返回符合某一条件的下标
    pos = np.where(y == 1)
    neg = np.where(y == 0)

    ax.scatter(X[pos, 1], X[pos, 2], s=50, c='b', marker='o', label='Admitted')
    ax.scatter(X[neg, 1], X[neg, 2], s=50, c='r', marker='x', label='Not Admitted')

    ax.set_xlabel('Exam 1 Score')
    ax.set_ylabel('Exam 2 Score')

    if (X.shape[1]<=3):
        plot_x=np.array([np.min(X[:,1]),np.max(X[:,1])])
        plot_y=(-1/theta[0,2])*(theta[0,1]*plot_x+theta[0,0])
        ax.plot(plot_x, plot_y, 'k')
        plt.legend(['Decision boundary','y = 1','y = 0'])
    else:
        u=np.linspace(-1,1.5,50)
        v=np.linspace(-1,1.5,50)
        z=np.zeros((len(u),len(v)))

        for i in range(len(u)):
            for j in range(len(v)):
                z[i,j] = map_feature(u[i],v[j]).dot(theta.T)
        z=z.T
        ax.contour(u,v,z,0,colors='black')
        plt.legend(['y = 1','y = 0','Decision boundary'])

    plt.show()

    return 0


# 绘图-高斯核（有问题）
def plot_decision_boundary1(theta,X,data):
    # data['Admitted']选择data中的'Admitted'列，使用类字典属性,返回的是Series类型
    positive = data[data['Admitted'].isin([1])]   # data['Admitted'].isin([1]) 获取数值为1的下标索引
    negative = data[data['Admitted'].isin([0])]   # 根据某属性来选取指定条件的行，括号中必须为list

    fig, ax = plt.subplots(figsize=(6,4))

    """
    问题：怎么通过下标索引
    """
    ax.scatter(positive['Exam 1'], positive['Exam 2'], s=50, c='b', marker='o', label='Admitted')
    ax.scatter(negative['Exam 1'], negative['Exam 2'], s=50, c='r', marker='x', label='Not Admitted')

    ax.legend() # 显示label
    ax.set_xlabel('Exam 1 Score')
    ax.set_ylabel('Exam 2 Score')


    u=np.linspace(1.5,2.2,100)
    v=np.linspace(1,1.7,100)
    X1,X2=np.meshgrid(u,v)

    z=np.zeros((X1.shape[0],X1.shape[1]))

    k=gaussian_kernel(X1[1,:], X2[6,:],0.1)
    # for i in range(len(X1)):
    #     for j in range(len(X1)):
    #         z[i,j] = gaussian_kernel(X1[i,:], X2[j,:],0.1)
    #         z[j,i] = z[i,j]
    # z[z>=0.0000001]=1
    # z[z<0.0000001]=0
    print(k)
    # ax.contour(u,v,z,[0,0],colors='black')

    # plt.show()
    return 0
