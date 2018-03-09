# -*- coding:utf-8 -*-

from findClosestCentroids import find_closest_centroids
from plotDatas import plot_datas
from computeCentroids import compute_centroids

def runk_means(X,centroids,iters=1):
    k=len(centroids)
    for i in range(iters):
        index=find_closest_centroids(centroids,X)
        centroids=compute_centroids(X,index,k)
    plot_datas(X,centroids,index,k)
