# -*- coding:utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt


def plot_datas(X,C):
    # 画图
    plt.style.use('ggplot')
    fig, ax = plt.subplots()
    ax.scatter(X[:, 0], X[:, 1], c=C)
    ax.set_title('Spectral Clustering')
    plt.show()