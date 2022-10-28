import numpy as np


def tanh(x):
    """ activation function."""
    return np.tanh(x)


def tanh_prime(x):
    """ activation function backwards. """
    return 1-np.tanh(x)**2


# loss function and its derivative
def mse(y_true, y_pred):
    """ loss function. """
    return np.mean(np.power(y_true-y_pred, 2))


def mse_prime(y_true, y_pred):
    """ loss function backwards. """
    return 2*(y_pred-y_true)/y_true.size
