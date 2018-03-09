__author__ = 'ubuntu'
from gradientDescent import *
import matplotlib.pyplot as plt

def Constructor(X, y, theta, alpha, iters, data, X_n):
    # perform gradient descent to "fit" the model parameters
    g0, cost1 = batch_gradient_descent(X, y, theta, alpha, iters)
    g1, cost1= stochastic_gradient_descent(X, y, theta, alpha, iters)
    g2, cost2= mini_batch_gradient_descent(X, y, theta, alpha, iters)

    x = np.linspace(data.Population.min(), data.Population.max(), 100)
    f0 = g0[0, 0] + (g0[0, 1] * x)
    f1 = g1[0, 0] + (g1[0, 1] * x)
    f2 = g2[0, 0] + (g2[0, 1] * x)

    # 正则化
    g0_r, cost1_r = batch_gradient_descent(X, y, theta, alpha, iters, lamda=0.3)
    g1_r, cost1_r= stochastic_gradient_descent(X, y, theta, alpha, iters, lamda=0.3)
    g2_r, cost2_r= mini_batch_gradient_descent(X, y, theta, alpha, iters, lamda=0.3)

    x = np.linspace(data.Population.min(), data.Population.max(), 100)
    f0_r = g0_r[0, 0] + (g0_r[0, 1] * x)
    f1_r = g1_r[0, 0] + (g1_r[0, 1] * x)
    f2_r = g2_r[0, 0] + (g2_r[0, 1] * x)

    # 归一化和正则化
    g0_n, cost0_n = batch_gradient_descent(X_n, y, theta, alpha, iters, lamda=0.3)
    g1_n, cost1_n= stochastic_gradient_descent(X_n, y, theta, alpha, iters, lamda=0.3)
    g2_n, cost2_n= mini_batch_gradient_descent(X_n, y, theta, alpha, iters, lamda=0.3)

    x1 = np.linspace(X_n[:,1].min(), X_n[:,1].max(), 100)
    f0_n = g0_n[0, 0] + (g0_n[0, 1] * x1)
    f1_n = g1_n[0, 0] + (g1_n[0, 1] * x1)
    f2_n = g2_n[0, 0] + (g2_n[0, 1] * x1)

    x_all = np.c_[x,x1]
    f_all = np.c_[f0,f1,f2,f0_r,f1_r,f2_r,f0_n,f1_n,f2_n]

    # fig, ax = plt.subplots(figsize=(6,4))
    # ax.plot(np.arange(iters), cost, 'r')
    # ax.set_xlabel('Iterations')
    # ax.set_ylabel('Cost')
    # ax.set_title('Error vs. Training Epoch')
    # plt.show()

    return x_all,f_all
