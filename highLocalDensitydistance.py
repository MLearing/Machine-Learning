# -*- coding:utf-8 -*-

from regionQuery import region_query
import numpy as np
from pdist import pdist

def high_local_density_distance(X,epsilon):
    m=len(X)
    density=np.zeros(m)
    hdd=np.zeros(m)
    distance=pdist(X,X)

    for i in range(m):
        density[i]=len(region_query(i,distance,epsilon))

    for i in range(m):
        local_density=region_query(i,distance,epsilon)
        number=len(local_density)
        temp=100

        for j in range(number):
            if density[i]<density[local_density[j]]:
                if temp>distance[i,local_density[j]]:
                    temp=distance[i,local_density[j]]

        hdd[i]=temp

    return density,hdd