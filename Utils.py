import numpy as np
import numbers

def sigmoid(z):
    """
    The sigmoid function is used to uniformally distribute values from
    [-inf;+inf] to [0;1].
    z can be a matrix, a vector or a scalar
    """
    return 1.0/(1.0+np.exp(-z))


def sigmoid_grad(z):
    """
    Gradient of the sigmoid function
    """
    return np.multiply(sigmoid(z), 1-sigmoid(z))


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


def label_to_bitvector(lbl, nb_classes):
    """
    Converts a lbl to a bitvector where the lblth element is 1 and the others 0
    """
    if isinstance(lbl, numbers.Real):
        return np.array([1 if i == lbl else 0 for i in range(nb_classes)])
    else:
        raise TypeError


def labels_to_bitmatrix(lbls, nb_classes):
    """
    Converts a vector of labels to a matrix made of bitvectors.
    """
    if isinstance(lbls, np.ndarray) and \
       (len(lbls.shape) == 1 or lbls.shape[1] == 1):
        return np.array([label_to_bitvector(int(lbl)-1, nb_classes) for
                         lbl in lbls])
    else:
        raise TypeError
