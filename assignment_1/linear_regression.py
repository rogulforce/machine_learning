#!/usr/bin/env python3

import numpy as np
import pytest
from sklearn.linear_model import LinearRegression

import matplotlib.pyplot as plt


class LinearRegr:
    """ Linear Regression model from scratch.
    y = w_0 + w_1*x_1 + ... + w_m*x_m
    Attributes:
        self.weight (np.array): weights vector. weight = (w_0, w_1, ..., w_m)
    """
    weight = np.array([])

    def fit(self, x: np.array, y: np.array, learning_rate: float = 0.002, max_iter: int = 100000):
        """ Find weight vector minimizing quadratic loss function <self._cost> using <self._gradient_descent> method.
        Args:
            x (np.array[n,m]): n m-dimensional samples of parameters
            y (np.array[n]): n results.
            learning_rate (float). precision of each step. Defaults to 0.002
            max_iter (int). Number of steps. Defaults to 10**5.
        Returns:
            self.
        """
        w, b = self._gradient_descent(x, y, np.zeros(x.shape[1]), 0, learning_rate=learning_rate, max_iter=max_iter)
        self.weight = np.append(np.array([b]), w)
        return self

    def predict(self, x: np.array):
        """ Find weight vector minimizing quadratic loss function <self._cost> using <self._gradient_descent> method.
        Args:
            x (np.array[n,m]): n m-dimensional samples of parameters
        Returns:
            y (np.array[n]): n results. Calculated using self.weight
        """
        return x.dot(self.weight[1:]) + self.weight[0]

    @staticmethod
    def _cost(x, y, weight: np.array, bias: float):
        """ Return cost of model with given parameter"""
        cost = np.sum((((x.dot(weight) + bias) - y) ** 2))
        return cost

    @staticmethod
    def _gradient_descent(x: np.array, y: np.array, weight: np.array, bias: float, learning_rate: float,
                          max_iter: int):
        """ Calculate minimum of loss function using gradient descent algorithm.
        Args:
            x (np.array[n,m]): n m-dimenstional samples of parameters
            y (np.array[n]): n results.
            weight (np.array[n]). Initial weight.
            bias (float). Initial bias.
            learning_rate (float). precision of each step.
            max_iter (int). Number of steps.
        Returns:
            weight (np.array[n]). Calculated weight
            bias (float). Calculated bias.
        """
        for _ in range(max_iter):
            z = x.dot(weight) + bias
            loss = z - y

            weight_gradient = x.T.dot(loss)
            bias_gradient = np.sum(loss)

            weight = weight - learning_rate * weight_gradient
            bias = bias - learning_rate * bias_gradient

        return weight, bias


def test_RegressionInOneDim():
    X = np.array([1, 3, 2, 5]).reshape((4, 1))
    Y = np.array([2, 5, 3, 8])
    a = np.array([1, 2, 10]).reshape((3, 1))
    expected = LinearRegression().fit(X, Y).predict(a)
    actual = LinearRegr().fit(X, Y).predict(a)
    assert list(actual) == pytest.approx(list(expected))


def test_RegressionInThreeDim():
    X = np.array([1, 2, 3, 5, 4, 5, 4, 3, 3, 3, 2, 5]).reshape((4, 3))
    Y = np.array([2, 5, 3, 8])
    a = np.array([1, 0, 0, 0, 1, 0, 0, 0, 1, 2, 5, 7, -2, 0, 3]).reshape((5, 3))
    expected = LinearRegression().fit(X, Y).predict(a)
    actual = LinearRegr().fit(X, Y).predict(a)
    assert list(actual) == pytest.approx(list(expected))
