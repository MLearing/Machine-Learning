# -*- coding:utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import os
from dbscan import dbscan
from plotCluster import plot_cluster

path=os.getcwd() + '\ex8data.txt'
X=np.loadtxt(path)

epsilon=0.5;
MinPts=10;

IDX,C=dbscan(X,epsilon,MinPts)

plot_cluster(X,IDX,C)


