# -*- coding:utf-8 -*-

import numpy as np
from pdist import pdist
from sklearn.metrics.pairwise import rbf_kernel


def full_connect_similarity(points):
    """
    相似性函数，利用径向基核函数计算相似性矩阵，对角线元素置为０
    对角线元素为什么要置为０我也不清楚，但是论文里是这么说的
    :param points:
    :return:
    """
    # distance=pdist(points,points)
    # res = np.exp(-np.multiply(distance,distance)/(2*0.1))
    res = rbf_kernel(points)
    for i in range(len(res)):
        res[i, i] = 0
    return res


def knn_similarity(data,k=5,delta=0.03):  # 利用KNN获得相似矩阵
    points_num = len(data)
    W = np.zeros((points_num,points_num))

    distance=pdist(data,data)
    dis_matrix = np.exp(-np.multiply(distance,distance)/(2*delta**2))
    for i in range(len(dis_matrix)):
        dis_matrix[i, i] = 0

    W=dis_matrix

    # enumerate函数用于将一个可遍历的数据对象(如列表、元组或字符串)组合为一个索引序列，同时列出数据下标和数据
    for idx,each in enumerate(dis_matrix):
        index  = np.argsort(-each)  # 升序（去负号降序）
        W[idx,index[k:len(index)]] = 0

    return W