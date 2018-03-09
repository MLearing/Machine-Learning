# -*- coding:utf-8 -*-

import numpy as np
from similarityFunction import *
from sklearn.preprocessing import normalize

def spectral_clustering(X,k):
    # 相似矩阵W
    # W=full_connect_similarity(X)  # 全连接
    W=knn_similarity(X)  # k近邻
    # 度矩阵D^(-1/2)
    D=np.diag(np.zeros(len(W)))
    for i in range(len(W)):
        D[i,i]=np.sum(W[i,:])**(-0.5)
    #拉普拉斯矩阵L
    # L=D-W
    L = np.eye(len(X)) - np.dot(np.dot(D, W), D)

    # 求取特征值和特征向量
    eigval,eigvec=np.linalg.eig(L)
    # 对特征值排序并获取下标（由小到大）
    index=np.argsort(eigval)[:k]
    # 特征向量正则化
    eigvec=normalize(eigvec[:,index])

    return eigval,eigvec
