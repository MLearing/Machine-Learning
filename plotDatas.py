# -*- coding:utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def plot_datas(X,centroids,index,k):

    fig, ax = plt.subplots(figsize=(6,4))
    palette=sns.color_palette("hls", k)   # 调色板
    colors=np.zeros((len(index),3))

    for i in range(k):
        colors[np.where(index==i),:]=list(palette[i])
    ax.scatter(X[:,0],X[:,1],s=30,c=colors)

    ax.scatter(centroids[:,0],centroids[:,1],s=50,color='k',marker='x')

    plt.show()

def plot_density(X,density,hdd):
    fig,ax=plt.subplots(2,1,figsize=(6,9))
    hdd[np.where(hdd==100)[0]]=0.1

    point=np.where(hdd>0.45)
    ax[0].scatter(density,hdd,s=20,c='r')
    ax[1].scatter(X[:,0],X[:,1])
    ax[1].scatter(X[point,0],X[point,1],s=20,c='r')
    plt.show()