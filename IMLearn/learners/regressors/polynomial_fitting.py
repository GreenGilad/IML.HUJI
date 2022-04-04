from __future__ import annotations
from typing import NoReturn
from IMLearn.learners.regressors.linear_regression import LinearRegression
from ...base import BaseEstimator
import numpy as np
from IMLearn.metrics.loss_functions import mean_square_error


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
        self.k_ = k
        self.fitted_, self.coefs_ = False, None

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
        van_mtx = self.__transform(X)
        lin = LinearRegression(include_intercept=False).fit(van_mtx, y)
        self.coefs_ = lin.coefs_

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
        pre_van_mtx = self.__transform(X)
        return pre_van_mtx @ self.coefs_

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
        y_predicted = self._predict(X)
        return mean_square_error(y, y_predicted)

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
        return np.vander(X, N=self.k_ + 1)
