
"""
Network.py
~~~~~~~~~~
IT WORKS
A module to implement the stochastic gradient descent learning
algorithm for a feedforward neural network.  Gradients are calculated
using backpropagation.  Note that I have focused on making the code
simple, easily readable, and easily modifiable.  It is not optimized,
and omits many desirable features.
"""

#### Libraries
import random
import numpy as np


class Network(object):
    def __init__(self,sizes):
        """The list ``sizes`` contains the number of neurons in the
                respective layers of the network.  For example, if the list
                was [2, 3, 1] then it would be a three-layer network, with the
                first layer containing 2 neurons, the second layer 3 neurons,
                and the third layer 1 neuron.  The biases and weights for the
                network are initialized randomly, using a Gaussian
                distribution with mean 0, and variance 1.  Note that the first
                layer is assumed to be an input layer, and by convention we
                won't set any biases for those neurons, since biases are only
                ever used in computing the outputs from later layers."""
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
        self.weights = [np.random.randn(y, x) for x, y in zip(sizes[:-1], sizes[1:])]


    def feedforward(self,a):
        """Return the output of the network if ``a`` is input."""
        for b, w in zip(self.biases, self.weights):
            a = sigmoid(np.dot(a, w)+b)
        return a


    def SGD(self, training_data, epochs, mini_batch_size, learning_rate, test_data= None):
        """Train the neural network using mini-batch stochastic
                gradient descent.  The ``training_data`` is a list of tuples
                ``(x, y)`` representing the training inputs and the desired
                outputs.  The other non-optional parameters are
                self-explanatory.  If ``test_data`` is provided then the
                network will be evaluated against the test data after each
                epoch, and partial progress printed out.  This is useful for
                tracking progress, but slows things down substantially."""

        if test_data:
            test_data = list(test_data)
            n_test = len(test_data)
        training_data = list(training_data)
        n = len(training_data)
        for i in range(epochs):
            random.shuffle(training_data)
            mini_batches = [training_data[k: k+mini_batch_size] for k in range(0, n, mini_batch_size)]
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, learning_rate)
            if test_data:
                print("Epoch {} : {} / {}".format(i, self.evaluate(test_data), n_test))
            else:
                print("Epoch {} complete".format(i))

    def update_mini_batch(self, mini_batch, learning_rate):
        """Update the network's weights and biases by applying
                gradient descent using backpropagation to a single mini batch.
                The ``mini_batch`` is a list of tuples ``(x, y)``, and ``eta``
                is the learning rate."""



#### Necessary Functions

def sigmoid(z):
    """The sigmoid function."""
    return  1.0/(1.0+np.exp(-1))

def sigmoid_prime(z):
    """Derivative of the sigmoid function."""
    return sigmoid(z)*(1 - sigmoid(z))
