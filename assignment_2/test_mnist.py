import matplotlib.pyplot as plt

import numpy as np

from neural_network import Network
from layers import FullyConnectedLayer, ActivationLayer
from utils import tanh, tanh_prime, mse, mse_prime

from keras.datasets import mnist
from keras.utils import np_utils


def preprocess(x, y):
    """ Preprocess MNIST data. """
    # Each sample contains 28x28 array of pixels. Each pixel is given intensivity in range [0,255].
    # First preprocess the data:
    # 1. reshape 28 x 28 matrix to 1 x 28**2 array.
    x = x.reshape(x.shape[0], 1, 28 ** 2)
    x = x.astype('float32')

    # 2. min-max normalization.
    x /= 255

    # 3. hot encode the output. Ex. 3 -> 0,0,0,1,0,0,0,0,0,0
    y = np_utils.to_categorical(y)

    return x, y


def load_mnist_data():
    """ Function loading and preprocessing data from MNIST dataset using <keras.datasets> package. <Tensorflow package>
        needs to be installed before.
    """

    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    x_train, y_train = preprocess(x_train,y_train)
    x_test, y_test = preprocess(x_test, y_test)

    return x_train, y_train, x_test, y_test


def test_mnist(epochs: int, learning_rate: float, train_len: int, test_len: int, layer_size: list = (100, 50)):
    """ Test Network on MNIST data.
    Args:
        epochs (int): number of steps in learning process.
        learning_rate (float): size of each step.
        train_len (int): size of random training sample.
        test_len (int): size of random testing sample.
        layer_size (list): list of sizes for each layer."""
    # load data
    x_train, y_train, x_test, y_test = load_mnist_data()

    # get random samples to train / test
    train_sample_start = np.random.randint(0, len(x_train) - train_len - 1)
    test_sample_start = np.random.randint(0, len(x_test) - test_len - 1)

    # initialize network
    network = Network()

    # set loss function
    network.set_loss_function(mse, mse_prime)

    # first layer
    network.add_layer(FullyConnectedLayer(28 ** 2, layer_size[0]))
    network.add_layer(ActivationLayer(tanh, tanh_prime))

    # additional layers
    for num, size in enumerate(layer_size[:-1]):
        network.add_layer(FullyConnectedLayer(size, layer_size[num + 1]))
        network.add_layer(ActivationLayer(tanh, tanh_prime))

    # last layer
    network.add_layer(FullyConnectedLayer(layer_size[-1], 10))
    network.add_layer(ActivationLayer(tanh, tanh_prime))

    # train network
    network.fit(x_train[train_sample_start:train_sample_start + train_len],
                y_train[train_sample_start:train_sample_start + train_len], epochs=epochs, learning_rate=learning_rate)

    # test network
    prediction = network.predict(x_test[test_sample_start:test_sample_start + test_len])
    print("predicted:")
    print([np.argmax(it) for it in prediction])
    print("actual:")
    print([np.where(it == 1)[0][0] for it in y_test[test_sample_start:test_sample_start + test_len]])

    plt.figure(figsize=(10, 5))
    plt.plot(range(1, epochs+1), network.error)
    plt.xlabel('epochs')
    plt.ylabel('error')
    plt.show()


if __name__ == "__main__":
    test_mnist(epochs=30, learning_rate=0.1, train_len=1000, test_len=10, layer_size=[100, 50])
