# -*- coding:utf-8 -*-
import numpy as np

# 数据处理
def datas_to_array(filename):
    "txt文件数据格式转换为矩阵形式"
    with open(filename,'r') as file:
        '''
        # 紧跟with后面的语句被求值后，返回对象的 __enter__() 方法被调用，这个方法的返回值将被赋值给as后面的变量,
        # 当with后面的代码块全部被执行完之后，将调用前面返回对象的 __exit__()方法。
        '''
        flag=True
        while True:
            data=file.readline()  # data为字符串
            if not data:
                break
            data=data.strip().split(",")  # strip是删去 split是分开 data为字符串类列表

            if flag:
                datas=[float(x) for x in data]  # 字符串转换为浮点
                matrix=np.array(datas)
                flag=False
            else:
                datas=[float(x) for x in data]
                matrix=np.c_[matrix,datas]     # 添加列
        matrix=matrix.T
    return matrix

# 数据标准化
def feature_normalize(X):
    mu=np.mean(X,0)
    sigma=np.std(X,0)
    X_norm=(X-mu)/sigma
    return mu,sigma,X_norm

# 梯度下降
def gradient_descent(X,y,theta,alpha,iters):
    for i in range(iters):
        theta=theta-alpha*np.linalg.inv((X.T).dot(X)).dot((X.T).dot(X.dot(theta)-y))
    return  theta

def normal_equation(X,y):
    # 正规方程计算参数
    theta = ((np.linalg.inv((X.T).dot(X))).dot(X.T)).dot(y)
    return theta

