__author__ = 'ubuntu'

import numpy as np

# 数据标准化
def feature_normalize(X):
    mu=np.mean(X,0)
    sigma=np.std(X,0)
    X_norm=(X-mu)/sigma
    return mu,sigma,X_norm