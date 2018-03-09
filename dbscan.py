__author__ = 'ubuntu'

import numpy as np
from pdist import pdist

def region_query(i,D,epsilon):
    neighbors=np.where(D[:,i]<=epsilon)[0]
    return neighbors

def expend_cluster(i,neighbors,C,IDX,visited,D,epsilon,MinPts):
    IDX[i]=C
    k=1

    while True:
        j=neighbors[k-1]

        if ~visited[j]:
            visited[j]=True
            neighbors2=region_query(j,D,epsilon)
            if len(neighbors2)>=MinPts:
                neighbors=np.r_[neighbors,neighbors2]

        if IDX[j]==0:
            IDX[j]=C

        k=k+1
        if k>len(neighbors):
            break

    return IDX

def dbscan(X,epsilon,MinPts):
    C=0
    n=len(X)
    IDX=np.zeros(n)
    D=pdist(X,X)

    visited=np.zeros(n)
    visited=(visited==1)
    isnoise=(visited==1)

    # i表示第i个点
    for i in range(n):
        if ~visited[i]:
            visited[i]=True

            # 计算第i个点的密度直达点
            neighbors=region_query(i,D,epsilon)
            if len(neighbors)<MinPts:
                isnoise[i]=True
            else:
                C=C+1
                # 计算第i点的密度相连点（所有密度相连的点看作是一个簇）
                IDX=expend_cluster(i,neighbors,C,IDX,visited,D,epsilon,MinPts)
    return IDX,C