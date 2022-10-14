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

    def fit(self, x: np.array, y: np.array):
        """ Find weight vector minimizing quadratic loss function <self._cost> using <self._gradient_descent> method.
        Args:
            x (np.array[n,m]): n m-dimensional samples of parameters
            y (np.array[n]): n results.
        Returns:
            self.
        """
        # add bias column to the x matrix.
        x_w_bias_col = np.c_[np.ones((x.shape[0], 1)), x]

        n = x_w_bias_col.shape[1]

        # identity matrix A added
        a = np.identity(n) * self.alpha
        a[0, 0] = 0

        # according to the provided algorithm.
        self.weight = np.linalg.inv(x_w_bias_col.T.dot(
            x_w_bias_col) + a).dot(x_w_bias_col.T).dot(y)
        return self

    def predict(self, x):
        """ Find weight vector minimizing quadratic loss function <self._cost> using <self._gradient_descent> method.
        Args:
            x (np.array[n,m]): n m-dimensional samples of parameters
        Returns:
            y (np.array[n]): n results. Calculated using self.weight
        """
        return x.dot(self.weight[1:]) + self.weight[0]


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
