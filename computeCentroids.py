# -*- coding:utf-8 -*-

import numpy as np

def compute_centroids(X,index,k):
    centroids=np.zeros((k,2))
    for i in range(k):
        centroids[i,:]=np.mean(X[np.where(index==i),:],1)
    return centroids