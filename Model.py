
class Model(object):
    """
    Base class for all models. Should not be instanced directly.
    """

    def __init__(self, loop=False):
        """
        If loop is set, the model will prefer looping algorithm over
        vectorized ones (not recommended for large datasets, but easier
        to understand on small examples).
        """
        self.loop = loop

    def hypothesis(self, theta, x):
        """
        Applies the model with parameters theta on x.
        """
        raise NotImplementedError

    def train(self, X, y):
        """
        Trains model with training examples.
        @param X: Inputs matrix, [#examples*#features]
        @param y: Outputs vector, [#examples]
        """
        raise NotImplementedError

    def predict(self, Theta, x):
        """
        Tries to predict the correct class for input x,
        using learned parameters set Theta.
        """
        classes = Theta.shape[0]
        best = 0
        res = 0
        for c in range(classes):
            tmp = self.hypothesis(Theta[c], x)
            if tmp > best:
                best = tmp
                res = c
        return res

    def evaluate(self, Theta, X, y):
        """
        Gives an evaluation of how the model is performing by
        comparing predictions and labels.
        """
        m = X.shape[0]
        acc = 0
        for (x, lbl) in zip(X, y):
            if self.predict(Theta, x) == lbl:
                acc += 1
        return acc/m

    def cost_loop(self, theta, X, y):
        raise NotImplementedError

    def cost_vect(self, theta, X, y):
        raise NotImplementedError
    
    def cost(self, theta, X, y):
        """
        Evaluates the cost of the parameters set theta on the
        training set.
        """
        if self.loop:
            return self.cost_loop(theta, X, y)
        else:
            return self.cost_vect(theta, X, y)
