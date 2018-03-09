__author__ = 'Designer'

from sigmoid import sigmoid

def predict(theta, X):
    probability = sigmoid(X * theta.T)
    return [1 if x >= 0.5 else 0 for x in probability]