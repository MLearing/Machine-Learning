import numpy as np

def computeCost(X, y, theta,lamda=0):
    m=len(X)
    R=lamda/(2*m)*np.sum(np.square(theta))  # np.square ¼ÆËãx^2
    inner = np.power(((X * theta.T) - y), 2)
    return np.sum(inner) / (2 * m) +R

