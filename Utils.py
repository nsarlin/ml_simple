import numpy as np

def sigmoid(z):
    """
    The sigmoid function is used to uniformally distribute values from
    [-inf;+inf] to [0;1].
    z can be a matrix, a vector or a scalar
    """
    return 1.0/(1.0+np.exp(-z))


def add_bias(X):
    """
    Adds either a column of ones if X is a matrix, or a single one
    if a is a vector.
    """
    # matrix
    try:
        return np.insert(X, 0, 1, axis=1)
    # vector
    except IndexError:
        return np.insert(X, 0, 1)
