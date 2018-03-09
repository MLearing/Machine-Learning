__author__ = 'Designer'

import numpy as np

def map_feature(X1,X2):
    degree = 6
    if isinstance(X1,np.ndarray):  # 判断数据类型
        p = np.ones(len(X1))
    else:
        m=np.array([X1])
        p = np.ones(len(m))

    for i in range(1,degree+1):
        for j in range(0,i+1):
            out=np.multiply(X1**(i-j),X2**j)
            if not isinstance(X1,np.ndarray): # not 逻辑非
                out=np.array([out])
            p=np.c_[p,out]
    return p
