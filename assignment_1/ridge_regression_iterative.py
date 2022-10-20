# Piotr Rogula, 249801
import numpy as np
import pytest
from sklearn.linear_model import Ridge


class RidgeRegr:
    def __init__(self, alpha=1.0, ):
        """
        Args:
            alpha (float): regularization parameter.
        Attributes:
            self.weight (np.array): weights vector. weight = (w_0, w_1, ..., w_m)
        """
        self.alpha = alpha
        self.weight = np.array([])

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
        # add bias column to the x matrix.
        x_w_bias_col = np.c_[np.ones((x.shape[0], 1)), x]

        init_weight = np.zeros(x_w_bias_col.shape[1])
        weight = self._gradient_descent(x_w_bias_col, y, init_weight,
                                        learning_rate=learning_rate, max_iter=max_iter)
        self.weight = weight
        return self

    def predict(self, x):
        """ Find weight vector minimizing quadratic loss function <self._cost> using <self._gradient_descent> method.
        Args:
            x (np.array[n,m]): n m-dimensional samples of parameters
        Returns:
            y (np.array[n]): n results. Calculated using self.weight
        """
        return x.dot(self.weight[1:]) + self.weight[0]

    def _gradient_descent(self,x: np.array, y: np.array, weight: np.array, learning_rate: float,
                          max_iter: int):
        """ Calculate minimum of loss function using gradient descent algorithm.
        Args:
            x (np.array[n,m]): n m-dimenstional samples of parameters with added bias column (ones) at beginning.
            y (np.array[n]): n results.
            weight (np.array[n]). Initial weight.
            bias (float). Initial bias.
            learning_rate (float). precision of each step.
            max_iter (int). Number of steps.
        Returns:
            weight (np.array[n]). Calculated weight
            bias (float). Calculated bias.
        """
        # identity matrix A added (mask)
        a = np.identity(x.shape[1]) * self.alpha
        a[0, 0] = 0

        for _ in range(max_iter):
            weight_gradient = 2 * (x.T @ (x @ weight - y) + a @ weight)
            weight = weight - learning_rate * weight_gradient

        return weight


def test_RidgeRegressionInOneDim():
    X = np.array([1, 3, 2, 5]).reshape((4, 1))
    Y = np.array([2, 5, 3, 8])
    X_test = np.array([1, 2, 10]).reshape((3, 1))
    alpha = 0.3
    expected = Ridge(alpha).fit(X, Y).predict(X_test)
    actual = RidgeRegr(alpha).fit(X, Y).predict(X_test)
    assert list(actual) == pytest.approx(list(expected), rel=1e-5)


def test_RidgeRegressionInThreeDim():
    X = np.array([1, 2, 3, 5, 4, 5, 4, 3, 3, 3, 2, 5]).reshape((4, 3))
    Y = np.array([2, 5, 3, 8])
    X_test = np.array([1, 0, 0, 0, 1, 0, 0, 0, 1, 2, 5, 7, -2, 0, 3]).reshape((5, 3))
    alpha = 0.4
    expected = Ridge(alpha).fit(X, Y).predict(X_test)
    actual = RidgeRegr(alpha).fit(X, Y).predict(X_test)
    assert list(actual) == pytest.approx(list(expected), rel=1e-3)
