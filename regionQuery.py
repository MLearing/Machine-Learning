# -*- coding:utf-8 -*-

import numpy as np

def region_query(i,D,epsilon):
    neighbors=np.where(D[:,i]<=epsilon)[0]
    return neighbors