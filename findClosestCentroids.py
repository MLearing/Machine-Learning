# -*- coding:utf-8 -*-

from pdist import pdist
import numpy as np

def find_closest_centroids(centroids,X):
    distance=pdist(centroids,X)
    index=np.argmin(distance,1)  # 获得最值所在的下标
    return index