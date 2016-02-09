import numpy as np
import numbers
from Model import Model
from Utils import *

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
        return np.array([label_to_bitvector(lbl, nb_classes) for lbl in lbls])
    else:
        raise TypeError


class NeuralNetwork(Model):
    """
    A neural network is a way to learn complex and non linear functions
    for a classification boundary. It is inspired by the formal neuron
    representation.
    A Neuron is a entity that takes several inputs, and outputs an
    activation value. A neural network is composed of several layers,
    and the output of neurons in layer l are the inputs of neurons in
    layers l+1
    """
    def __init__(self, shape, regul=0, num_iter=400, init_range=None,
                 init_coefs=None, loop=False):
        """
        @param shape: The shape of the neural network, given as a list.
          For example, (25, 10, 4) means a network with an input layer of 25
          neurons, a hidden layer of 10 and an output layer of 4.
        @param regul: Regularization parameter. This parameter
          controls how we want to penalize complex sets of coefficients
          generated by the regression. Complex sets of coefficients are more
          prone to overfitting (when the model have good results on the
          training set but does not generalize to new examples).
        @param num_iter: How many step of training should we perform.
        @param init_range: If init_coefs is not provided, the coefficients of
          the neurons will be initialized between -init_range and +init_range.
          If you leave it to none, its default value will be computed
          automatically.
        @param init_coefs: Initial coefficients of the neurons. With coefficients
          initially set at 0, the neutwork will only be able to learn symmetric
          models. By default, these coefficients will be randomized. But you may
          want to provide them to replay computations without randomization.
        """
        if len(shape) < 2:
            raise ValueError("A Neural Network should at least have 2 layers")
        self.regul = regul
        self.num_iter = num_iter
        super().__init__(loop)
        self.shape = shape
        if init_coefs is None:
            self.init_coefs = self.init_network(init_range)
        else:
            self.init_coefs = init_coefs


    def init_network(self, init_range=None):
        """Randomly initialize network coefficients to break symmetry"""
        init_coefs = []

        if init_range is None:
            init_range = np.sqrt(6)/np.sqrt(self.shape[0]+self.shape[-1])

        # The coefficient are used to go from one layer to the next.
        # We have to set up a coefficients matrix between each sets of two
        # layers. If layer j has lj neurons and layer j+1 has lj+1 neurons,
        # The coefficients matrix between j and j+1 is of size:
        # lj+1 * (lj + 1).
        # The "+ 1" takes account of the bias unit added to the output of each
        # layer.
        for in_size, out_size in zip(self.shape[:-1], self.shape[1:]):
            coefs = np.random.rand(out_size, in_size+1)
            coefs = coefs * 2 * init_range - init_range
            init_coefs.append(coefs)

        return init_coefs


    def forward_prop(self, Coefs, x):
        """
        Forward propagation is a set where the dataset is given as input to the
        network, which outputs a probability vector after flowing it through each
        nodes.
        """
        a = [x]
        z = []
        for cnt, coef in enumerate(Coefs):
            z.append(sigmoid(np.dot(add_bias(a[cnt]), coef.transpose())))
            a.append(sigmoid(z[cnt]))
        return z, a


    def hypothesis(self, Coefs, x):
        """
        With neural networks, the hypothesis returns a probability vector,
        Giving for each class the probability for x to belong to it.
        """
        return self.forward_prop(Coefs, x)[1][-1]


    def cost_loop(self, coefs, X, y):
        """
        The global cost of the network is a sum of the costs according to each
        class.
        """

        acc = 0
        m = X.shape[0]

        for i in range(m):
            prob = self.hypothesis(coefs, X[i])
            for k in range(self.shape[-1]):
                if k == y[i]:
                    acc += -np.log(prob[k])
                else:
                    acc += -np.log(1-prob[k])

        if self.regul != 0:
            acc_r = 0
            for l in range(len(coefs)):
                acc_r += np.sum(np.square(coefs[l][:, 1:]))
            acc += self.regul/2 * acc_r

        return acc/m


    def cost_vect(self, coefs, X, y):
        """
        Vectorized implementation of cost function.
        """
        m = X.shape[0]

        Y = labels_to_bitmatrix(y, self.shape[-1])
        probs = self.hypothesis(coefs, X)
        ucost = np.sum(np.sum(np.multiply(-Y, np.log(probs)) -\
                              np.multiply(1-Y, np.log(1-probs))))
        if self.regul != 0:
            for mat in coefs:
                ucost += self.regul/2 * np.sum(np.square(mat[:, 1:]))


        return ucost/m
