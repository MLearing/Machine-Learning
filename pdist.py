__author__ = 'ubuntu'

import math
import numpy as np

def pdist(centroids,X):
    k=len(centroids)
    m=len(X)
    distance=np.zeros((m,k))
    for i in range(k):
        for j in range(m):
            distance[j,i]=math.sqrt(sum((centroids[i,:]-X[j,:])**2))
    return distance