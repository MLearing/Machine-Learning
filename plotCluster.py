__author__ = 'ubuntu'

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
def plot_cluster(X,IDX,C):
    C=C+1
    fig,ax=plt.subplots(figsize=(5,4))
    palette=sns.color_palette("hls",C)
    colors=np.zeros((len(IDX),C))

    for i in range(C):
        colors[np.where(IDX==i),:]=list(palette[i])

    ax.scatter(X[:,0],X[:,1],s=40,c=colors,marker='.')
    plt.show()