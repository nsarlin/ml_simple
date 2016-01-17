
class Model(object):
    """
    Base class for all models. Should not be instanced directly.
    """

    def __init__(self, loop=False):
        """
        If loop is set, the model will prefer looping algorithm over
        vectorized ones.
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

    def evaluate(self, Theta, x):
        """
        Estimates the probability for x to belong to each class.
        Then, returns the class with the highest probability.
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
