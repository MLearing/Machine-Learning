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

    # i��ʾ��i����
    for i in range(n):
        if ~visited[i]:
            visited[i]=True

            # �����i������ܶ�ֱ���
            neighbors=region_query(i,D,epsilon)
            if len(neighbors)<MinPts:
                isnoise[i]=True
            else:
                C=C+1
                # �����i����ܶ������㣨�����ܶ������ĵ㿴����һ���أ�
                IDX=expend_cluster(i,neighbors,C,IDX,visited,D,epsilon,MinPts)
    return IDX,C