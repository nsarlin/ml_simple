import numpy as np

def sigmoid(z):
    """
    The sigmoid function is used to uniformally distribute values from
    [-inf;+inf] to [0;1].
    z can be a matrix, a vector or a scalar
    """
    return 1.0/(1.0+np.exp(-z))

def hypothesis(theta, x):
    """
    Here, the hypothesis answers to the question: Given our dataset x,
    what is the probability that x belongs to the right class ?
    """
    return sigmoid(np.dot(theta.transpose(), x))

def cost_vect(theta, X, y):
    m = X.shape[0]
    ucost = np.dot(np.log(sigmoid(np.dot(X, theta))).transpose(), y) +\
            np.dot(np.log(np.ones(m) -\
                          sigmoid(np.dot(X, theta))).transpose(),
                   (np.ones(m)-y))
    return -1.0/m*(ucost)

def cost_loop(theta, X, y):
    """
    With sigmoid, we cannot simply compute cost with difference between
    the hypothesis and the target, because target is either 0 or 1.
    The log is used to take account of the exponential in the
    hypothesis.
    """
    acc = 0
    m = X.shape[0]

    for i in range(m):
        if y[i] == 1:
            acc += -np.log(hypothesis(theta, X[i]))
        else:
             acc += -np.log(1-hypothesis(theta, X[i]))
    return acc/m

def grad_vect(theta, X, y, alpha):
    """
    Computes one step of gradient descent using vectorized algorithm.
    """
    m = X.shape[0]
    theta = theta - (alpha/m)*np.dot(X.transpose(),
                                     sigmoid(np.dot(X, theta)) - y)
    return theta


def grad_loop(theta, X, y, alpha):
    """
    Computes one step of gradient descent using looping algorithm.
    """
    theta_copy = theta.copy()
    n = theta.size
    m = X.shape[0]
    for j in range(n):
        acc = 0
        for i in range(m):
            acc += (hypothesis(theta_copy, X[i])-y[i])*X[i,j]
            
        theta[j] = theta_copy[j] - alpha*acc/m
    return theta

def add_bias(X):
    """
    Adds either a vector of ones if X is a matrix, or a single one
    if a is a vector.
    """
    # matrix
    try:
        return np.insert(X, 0, 1, axis=1)
    # vector
    except IndexError:
        return np.insert(X, 0, 1)

def grad_descent(X, y, alpha, num_iter=None, stop_gap=None,
                 init_theta=None, loop=False):
    if num_iter is None and stop_gap is None:
        raise ValueError("You should either set num_iter or stop_gap")
    if init_theta is None:
        init_theta = np.zeros(X.shape[1])

    theta = init_theta

    if num_iter:

        for i in range(num_iter):
            if loop:
                theta = grad_loop(theta, X, y, alpha)
            else:
                theta = grad_vect(theta, X, y, alpha)
            print("iter {}".format(i))
    else:
        last_cost = cost_loop(theta, X, y)
        while True:
            if loop:
                theta = grad_loop(theta, X, y, alpha)
                cost = cost_loop(theta, X, y)
            else:
                theta = grad_vect(theta, X, y, alpha)
                cost = cost_vect(theta, X, y)
            if abs(last_cost - cost) < stop_gap:
                break
            last_cost = cost
    if loop:
        print("Cost: {}".format(cost_loop(theta, X, y)))
    else:
        print("Cost: {}".format(cost_vect(theta, X, y)))
    return theta

def find_alpha(X, y, init_alpha = 0.01, loop = False):
    alpha = init_alpha
    good = False
    theta = np.zeros(X.shape[1])
    if loop:
        init_cost = cost_loop(theta, X, y)
    else:
        init_cost = cost_vect(theta, X, y)
    while True:
        theta = np.zeros(X.shape[1])
        if loop:
            theta = grad_loop(theta, X, y, alpha)
            cost = cost_loop(theta, X, y)
        else:
            theta = grad_vect(theta, X, y, alpha)
            cost = cost_vect(theta, X, y)
        if np.isnan(cost) or cost >= init_cost:
            alpha /= 3
            if good:
                break
            good = False
        else:
            if not good:
                break
            alpha *= 3
            good = True
    print("Alpha: {}".format(alpha))
    return alpha

def train_all(X, y, nb_classes, loop=False):
    n = X.shape[1]
    Theta = np.zeros((nb_classes, n))
    for c in range(nb_classes):
        print("Class {}".format(c))
        yc = [1 if val == c else 0 for val in y]
        alpha = find_alpha(X, yc, loop=loop)
        Theta[c] = grad_descent(X, yc, alpha, num_iter=400)
    return Theta


def evaluate(Theta, x):
    """
    Estimates the probability for x to belong to each class.
    Then, returns the class with the highest probability.
    """
    classes = Theta.shape[0]
    outs = np.zeros(classes)
    for c in range(classes):
        outs[c] = hypothesis(Theta[c], x)
    return np.argmax(outs)

