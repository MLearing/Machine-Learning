# -*- coding:utf-8 -*-

from numpy import *
import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets,linear_model
import time
from function import *

# 主函数
if __name__ == "__main__":
    # 获取数据
    data=datas_to_array('ex1data2.txt')
    # 特征标准化
    datas=feature_normalize(data[:,0:len(data[0])-1])

    # 特征均值
    mu=datas[0]
    # 特征标准差
    sigma=datas[1]
    # 特征值
    X=datas[2]
    # 目标值
    y=data[:,-1]
    y_mat=np.mat(y).T
    b = np.ones((X.shape[0],1))  # 生成数值为1的列向量
    X_norm=np.c_[b,X]  # 使用np.c_[]和np.r_[]分别添加列和行

    '''****************normal-equation****************'''
    theta_norm=normal_equation(X_norm,y_mat)
    # print('theta_norm=',theta_norm)

    # 预测
    x=np.r_[1,(np.array([2100,3])-mu)/sigma]
    print('y(x)=',float(x.dot(theta_norm)))

    '''****************gradient-descent****************'''
    alpha=0.4
    iters=1000
    theta_gd=np.zeros((X_norm.shape[1],1))
    theta_gd=gradient_descent(X_norm,y_mat,theta_gd,alpha,iters)
    # print('theta_gd=',theta_gd)

    # 预测
    x=np.r_[1,(np.array([2100,3])-mu)/sigma]
    print('y(x)=',float(x.dot(theta_gd)))

    '''****************sklearn-lineregression****************'''
    # s数据划分训练集和测试集
    X_train=np.mat(X[:-20])
    X_test=np.mat(X[-20:])

    # 目标划分为训练集和测试集
    y_train=y[:-20]
    y_test=y[-20:]
    # print(X_test.shape)

    # 训练模型
    regr=linear_model.LinearRegression()
    regr.fit(X_train,y_train)

    # 预测 y=wx+b
    print('y_calculat(x)=',((np.array([2100,3])-mu)/sigma).dot(regr.coef_)+regr.intercept_)
    t = (np.c_[2100, 3]-mu)/sigma
    # t=np.array([[6.238]])  # 矩阵
    print('y_predict(x)=',float(regr.predict(t)))

    # 回归系数
    # print('Coefficients:\n',regr.coef_,regr.intercept_)

    # 均方误差
    print('the mean sqare error:%.2f' %np.mean((regr.predict(X_test)-y_test)**2)) # **表示乘幂
    print('Variance score:%.2f' %regr.score(X_test,y_test))
    # 散点图
    # plt.scatter(X_test,y_test,color='black')
    # plt.plot(X_test,regr.predict(X_test),color='blue',linewidth=3)
    # plt.xticks()
    # plt.yticks()
    # plt.show()
