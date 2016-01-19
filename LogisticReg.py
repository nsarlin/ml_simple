import numpy as np
from Model import Model
from Utils import *


class LogisticReg(Model):
    """
    Logistic Regression is a simple classification algorithm,
    based on sigmoid function and gradient descent.
    """
    def __init__(self, alpha=None, num_iter=400, stop_gap=None,
                 loop=False):
        """
        @param alpha: Learning Rate. If alpha is big, the algorithm
          will learn fast, but can diverge. If alpha is small, the
          learning will be slow. If it is left unset, the method
          find_alpha will be called to try to set it appropriatly.
        @param num_iter: How many step of gradient descent should we
          perform.
        @param stop_gap: Instead of setting num_iter, we can decide to
          stop learning when the cost reduces less than stop_gap
          between two iterations.
        """
        self.alpha = alpha
        self.num_iter = num_iter
        self.stop_gap = stop_gap
        super().__init__(loop)

    def hypothesis(self, theta, x):
        """
        Here, the hypothesis answers to the question:
        Given our dataset x, what is the probability that x belongs
        to the right class ?
        """
        return sigmoid(np.dot(theta.transpose(), x))

    def cost_vect(self, theta, X, y):
        m = X.shape[0]
        ucost = np.dot(np.log(sigmoid(np.dot(X, theta))).transpose(), y) +\
                np.dot(np.log(np.ones(m) -\
                              sigmoid(np.dot(X, theta))).transpose(),
                       (np.ones(m)-y))
        return -1.0/m*(ucost)

    def cost_loop(self, theta, X, y):
        """
        With sigmoid, we cannot simply compute cost with difference
        between the hypothesis and the target, because target is
        either 0 or 1.
        The log is used to take account of the exponential in the
        hypothesis.
        """
        acc = 0
        m = X.shape[0]

        for i in range(m):
            if y[i] == 1:
                acc += -np.log(self.hypothesis(theta, X[i]))
            else:
                acc += -np.log(1-self.hypothesis(theta, X[i]))
        return acc/m

    def grad_vect(self, theta, X, y):
        """
        Computes one step of gradient descent using vectorized
        algorithm.
        """
        m = X.shape[0]
        theta = theta - (self.alpha/m)*np.dot(X.transpose(),
                                         sigmoid(np.dot(X, theta)) - y)
        return theta


    def grad_loop(self, theta, X, y):
        """
        Computes one step of gradient descent using looping algorithm.
        """
        theta_copy = theta.copy()
        n = theta.size
        m = X.shape[0]
        for j in range(n):
            acc = 0
            for i in range(m):
                acc += (self.hypothesis(theta_copy, X[i])-y[i])*X[i,j]
            theta[j] = theta_copy[j] - self.alpha*acc/m
        return theta

    def grad(self, theta, X, y):
        if self.loop:
            return self.grad_loop(theta, X, y)
        else:
            return self.grad_vect(theta, X, y)

    def grad_descent(self, X, y, init_theta=None):
        """
        Tries to minimize the cost function. Each iteration will make
        a small change in each component of theta, in order to make
        the global cost decrease. The changes are applied using the
        partial derivatives of the cost with respect to each component
        of theta, and alpha, the learning rate.
        """
        if self.num_iter is None and self.stop_gap is None:
            raise ValueError("You should either set num_iter or"
                             " stop_gap")
        if init_theta is None:
            init_theta = np.zeros(X.shape[1])

        theta = init_theta

        if self.num_iter:
            for i in range(self.num_iter):
                theta = self.grad(theta, X, y)
                print("iter {}".format(i))
        else:
            last_cost = self.cost(theta, X, y)
            while True:
                theta = self.grad(theta, X, y)
                cost = self.cost(theta, X, y)
                if abs(last_cost - cost) < self.stop_gap:
                    break
                last_cost = cost

        print("Cost: {}".format(self.cost(theta, X, y)))
        return theta


    def find_alpha(self, X, y, init_alpha = 0.01):
        """
        Tries to find the best learning rate (alpha).
        """
        self.alpha = init_alpha
        good = False
        theta = np.zeros(X.shape[1])

        init_cost = self.cost(theta, X, y)
        while True:
            theta = np.zeros(X.shape[1])
            theta = self.grad(theta, X, y)
            cost = self.cost(theta, X, y)

            if np.isnan(cost) or cost >= init_cost:
                self.alpha /= 3
                if good:
                    break
                good = False
            else:
                if not good:
                    break
                self.alpha *= 3
                good = True

        print("Alpha: {}".format(self.alpha))
        return self.alpha

    def train(self, X, y, nb_classes):
        n = X.shape[1]
        Theta = np.zeros((nb_classes, n))
        for c in range(nb_classes):
            print("Class {}".format(c))
            yc = [1 if val == c else 0 for val in y]
            if self.alpha is None:
                self.find_alpha(X, yc)
            Theta[c] = self.grad_descent(X, yc)
        return Theta

