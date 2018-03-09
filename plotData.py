__author__ = 'ubuntu'

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def plot_data(X_all,f,X_n,data):
    # ax子窗口，并且以numpy数组的方式保存在axes中，而fig仍然是整个图像对象
    fig, ax = plt.subplots(3,3,figsize=(12,8))

    ax[0,0].plot(X_all[:,0], f[:,0], 'r', label='Prediction')
    ax[0,0].scatter(data.Population, data.Profit, label='Traning Data')
    ax[0,0].legend(loc=2)
    ax[0,0].set_xlabel('Population')
    ax[0,0].set_ylabel('Profit')
    ax[0,0].set_title('batch_graident_descent')

    ax[0,1].plot(X_all[:,0], f[:,1], 'r', label='Prediction')
    ax[0,1].scatter(data.Population, data.Profit, label='Traning Data')
    ax[0,1].legend(loc=2)
    ax[0,1].set_xlabel('Population')
    ax[0,1].set_title('stochastic_graident_descent')

    ax[0,2].plot(X_all[:,0], f[:,2], 'r', label='Prediction')
    ax[0,2].scatter(data.Population, data.Profit, label='Traning Data')
    ax[0,2].legend(loc=2)
    ax[0,2].set_xlabel('Population')
    ax[0,2].set_title('mini_bacth_gradient_descent')

    plt.tight_layout()  # 调整每隔子图之间的距离
    ax[1,0].plot(X_all[:,0], f[:,3], 'r', label='Prediction')
    ax[1,0].scatter(data.Population, data.Profit, label='Traning Data')
    ax[1,0].legend(loc=2)
    ax[1,0].set_xlabel('Population')
    ax[1,0].set_ylabel('Profit')
    ax[1,0].set_title('Regularization lamda=0.3')

    ax[1,1].plot(X_all[:,0], f[:,4], 'r', label='Prediction')
    ax[1,1].scatter(data.Population, data.Profit, label='Traning Data')
    ax[1,1].legend(loc=2)
    ax[1,1].set_xlabel('Population')
    ax[1,1].set_title('Regularization lamda=0.3')

    ax[1,2].plot(X_all[:,0], f[:,5], 'r', label='Prediction')
    ax[1,2].scatter(data.Population, data.Profit, label='Traning Data')
    ax[1,2].legend(loc=2)
    ax[1,2].set_xlabel('Population')
    ax[1,2].set_title('Regularization lamda=0.3')

    plt.tight_layout()   # 调整每隔子图之间的距离
    ax[2,0].plot(X_all[:,1], f[:,6], 'r', label='Prediction')
    ax[2,0].scatter(X_n[:,1].tolist(), data.Profit, label='Traning Data')
    ax[2,0].legend(loc=2)
    ax[2,0].set_xlabel('Population')
    ax[2,0].set_ylabel('Profit')
    ax[2,0].set_title('Normalize and Regularization,lamda=0.3')

    ax[2,1].plot(X_all[:,1], f[:,7], 'r', label='Prediction')
    ax[2,1].scatter(X_n[:,1].tolist(), data.Profit, label='Traning Data')
    ax[2,1].legend(loc=2)
    ax[2,1].set_xlabel('Population')
    ax[2,1].set_title('Normalize and Regularization,lamda=0.3')

    ax[2,2].plot(X_all[:,1], f[:,8], 'r', label='Prediction')
    ax[2,2].scatter(X_n[:,1].tolist(), data.Profit, label='Traning Data')
    ax[2,2].legend(loc=2)
    ax[2,2].set_xlabel('Population')
    ax[2,2].set_title('Normalize and Regularization,lamda=0.3')

    plt.show()