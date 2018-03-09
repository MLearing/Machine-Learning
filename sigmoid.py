__author__ = 'Designer'

import numpy as np

def sigmoid(x):
    return 1.0/(1.0+np.exp(-x))