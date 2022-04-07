from __future__ import annotations
from typing import NoReturn

from ...base import BaseEstimator
import numpy as np
from numpy.linalg import pinv
from IMLearn.metrics.loss_functions import mean_square_error


class LinearRegression(BaseEstimator):
    """
    Linear Regression Estimator

    Solving Ordinary Least Squares optimization problem
    """

    def __init__(self, include_intercept: bool = True) -> LinearRegression:
        """
        Instantiate a linear regression estimator

        Parameters
        ----------
        include_intercept: bool, default=True
            Should fitted model include an intercept or not

        Attributes
        ----------
        include_intercept_: bool
            Should fitted model include an intercept or not

        coefs_: ndarray of shape (n_features,) or (n_features+1,)
            Coefficients vector fitted by linear regression. To be set in
            `LinearRegression.fit` function.
        """
        super().__init__()
        self.include_intercept_, self.coefs_ = include_intercept, None

    def _fit(self, X: np.ndarray, y: np.ndarray) -> NoReturn:
        """
        Fit Least Squares model to given samples

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to fit an estimator for

        y : ndarray of shape (n_samples, )
            Responses of input data to fit to

        Notes
        -----
        Fits model with or without an intercept depending on value of `self.include_intercept_`
        """
        # x = X if self.include_intercept_ else X[:,1:]
        x = np.hstack((np.ones((X.shape[0], 1)), X)) if self.include_intercept_ else X

        self.coefs_ = pinv(x) @ y

        # the calculation "by hand":

        # xtx = x.T @ x
        #
        # if np.linalg.det(xtx) != 0:
        #     print('short')
        #     self.coefs_ = np.array(pinv(xtx) @ x.T @ y).reshape(-1, 1)
        # else:
        #     print('long')
        #     u, sigma, vt = np.linalg.svd(x)
        #     sigma_cross = np.zeros((vt.shape[0], u.shape[0]))
        #     sigma_cross[:vt.shape[0], :vt.shape[0]] = np.diag(sigma)
        #     sigma_cross = pinv(sigma_cross)
        #     x_cross = u @ sigma_cross @ vt
        #     self.coefs_ = x_cross.T @ y

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
        x = np.hstack((np.ones((X.shape[0], 1)), X)) if self.include_intercept_ else X
        return np.array(x @ self.coefs_).reshape((x.shape[0], 1))



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
        return mean_square_error(self.predict(X), y)
