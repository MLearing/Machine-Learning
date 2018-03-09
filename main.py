# -*- coding:utf-8 -*-

import numpy as np
import os
from sklearn.cluster import KMeans
from spectralClustering import spectral_clustering
from  plotDatas import plot_datas
from similarityFunction import *
from pdist import pdist

path=os.getcwd() + '\ex8data1.txt'
X=np.loadtxt(path)


# 提取的特征向量前K个
k=6

eigvals,eigvec=spectral_clustering(X,k)
# kmeans聚类
clf = KMeans(n_clusters=2)
s = clf.fit(eigvec)
#每个样本所属的簇
C = s.labels_

# 画图
plot_datas(X,C)
