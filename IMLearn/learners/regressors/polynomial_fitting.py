from __future__ import annotations
from typing import NoReturn
from IMLearn.learners.regressors import LinearRegression
from IMLearn.base import BaseEstimator
import numpy as np
from IMLearn.metrics import mean_square_error


class PolynomialFitting(BaseEstimator):
    """
    Polynomial Fitting using Least Squares estimation
    """

    def __init__(self, k: int) -> PolynomialFitting:
        """
        Instantiate a polynomial fitting estimator

        Parameters
        ----------
        k : int
            Degree of polynomial to fit
        """
        super().__init__()
        self.degree = k
        self.coef = None

    def _fit(self, X: np.ndarray, y: np.ndarray) -> NoReturn:
        """
        Fit Least Squares model to polynomial transformed samples

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to fit an estimator for

        y : ndarray of shape (n_samples, )
            Responses of input data to fit to
        """
        # vandermonde_of_x = []
        # for i in range(self.degree + 1):
        #     vandermonde_of_x.append(X ** i)
        vandermonde_of_x = self.__transform(X)  # np.array(vandermonde_of_x).T
        l_reg = LinearRegression(include_intercept=False)
        l_reg.fit(vandermonde_of_x, y)
        self.coef = l_reg.coefs_

    def _predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict responses for given samples using fitted estimator

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to predict responses for

        Returns
        -------
        responses : ndarray of shape (n_samples, )
            Predicted responses of given samples
        """
        return self.__transform(X) @ self.coef

    def _loss(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Evaluate performance under MSE loss function

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Test samples

        y : ndarray of shape (n_samples, )
            True labels of test samples

        Returns
        -------
        loss : float
            Performance under MSE loss function
        """
        mse = mean_square_error(y, self._predict(X))
        return mse

    def __transform(self, X: np.ndarray) -> np.ndarray:
        """
        Transform given input according to the univariate polynomial transformation

        Parameters
        ----------
        X: ndarray of shape (n_samples,)

        Returns
        -------
        transformed: ndarray of shape (n_samples, k+1)
            Vandermonde matrix of given samples up to degree k
        """
        return np.vander(X, self.degree + 1, increasing=True)


# test = np.array([2, 3, 4, 5])
# w = PolynomialFitting(k=3)
# print(len(test))
# w.fit(test, np.array([]))
