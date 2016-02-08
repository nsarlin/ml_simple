
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

    def hypothesis(self, coefs, x):
        """
        Applies the model with parameters coefs on x.
        """
        raise NotImplementedError

    def train(self, X, y):
        """
        Trains model with training examples.
        @param X: Inputs matrix, [#examples*#features]
        @param y: Outputs vector, [#examples]
        """
        raise NotImplementedError

    def predict(self, Coefs, x):
        """
        Tries to predict the correct class for input x,
        using learned parameters set Coefs.
        """
        classes = Coefs.shape[0]
        best = 0
        res = 0
        for c in range(classes):
            tmp = self.hypothesis(Coefs[c], x)
            if tmp > best:
                best = tmp
                res = c
        return res

    def evaluate(self, Coefs, X, y):
        """
        Gives an evaluation of how the model is performing by
        comparing predictions and labels.
        """
        m = X.shape[0]
        acc = 0
        for (x, lbl) in zip(X, y):
            if self.predict(Coefs, x) == lbl:
                acc += 1
        return acc/m

    def cost_loop(self, coefs, X, y):
        raise NotImplementedError

    def cost_vect(self, coefs, X, y):
        raise NotImplementedError
    
    def cost(self, coefs, X, y):
        """
        Evaluates the cost of the parameters set coefs on the
        training set.
        """
        if self.loop:
            return self.cost_loop(coefs, X, y)
        else:
            return self.cost_vect(coefs, X, y)
