from abc import ABC, abstractmethod
import numpy as np


class Layer(ABC):
    """ Schema for Layer. """
    def __init__(self):
        self.input = None
        self.output = None

    @abstractmethod
    def forward(self, input_sample):
        """ Compute the output Y of a layer for a given <input_sample> X.
        Args:
            input_sample ().
        Returns:
            """
        raise NotImplementedError

    # computes dE/dX for a given dE/dY (and update parameters if any)
    @abstractmethod
    def backward(self, output_error, learning_rate):
        raise NotImplementedError


class FullyConnectedLayer(Layer):
    """ Fully Connected Layer Class"""
    def __init__(self, input_size: int, output_size: int):
        """
        Args:
            input_size (int): number of input neurons
            output_size (int): number of output neurons
        """
        super(FullyConnectedLayer, self).__init__()

        # initialize random weights and bias in range (-0.5, 0.5)
        self.weights = np.random.rand(input_size, output_size) - 0.5
        self.bias = np.random.rand(1, output_size) - 0.5

    def forward(self, input_sample: np.array):
        """ Compute the output Y of a layer for a given <input_sample> X.
            Update <self.input> and <self.output> attributes.
        Args:
            input_sample (np.array).
        Returns:
            output (np.array)
        """
        self.input = input_sample
        self.output = np.dot(self.input, self.weights) + self.bias
        return self.output

    def backward(self, output_error: np.array, learning_rate: float):
        """ Compute input_error and weights_error. Update weights and bias using learning_rate and computer error.
        Return input_error.
        Args:
            output_error (np.array): Given output Error
            learning_rate (float): Fixed learning rate
        Returns:
            input_error (np.array)
        """
        input_error = np.dot(output_error, self.weights.T)
        weights_error = np.dot(self.input.T, output_error)

        # update parameters
        self.weights -= learning_rate * weights_error
        self.bias -= learning_rate * output_error
        return input_error


class ActivationLayer(Layer):
    """ Activation Layer Class"""
    def __init__(self, activation, activation_prime):
        super(ActivationLayer, self).__init__()
        self.activation = activation
        self.activation_prime = activation_prime

    def forward(self, input_data):
        """ Return activated <input_data>. """
        self.input = input_data
        self.output = self.activation(self.input)
        return self.output

    def backward(self, output_error, learning_rate):
        """ Return activated input error. Learning rate is not used but since it's inherited from base class,
            it stays there."""
        return self.activation_prime(self.input) * output_error
