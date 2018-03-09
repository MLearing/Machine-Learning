# -*- coding:utf-8 -*-

import numpy as np
import os
from runkMeans import runk_means
from highLocalDensitydistance import high_local_density_distance

from plotDatas import plot_density

#load the dataset
path = os.getcwd() + '\ex7data2.txt'
X = np.loadtxt(path)  # 从文本加载数据

# 注意eps的选择对结果影响很大
epsilon=1
density,hdd=high_local_density_distance(X,epsilon)
plot_density(X,density,hdd)

centroids = np.array([[3,3],[6,2],[8,5]])
runk_means(X,centroids,iters=20)

