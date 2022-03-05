"""
The following file contains base classes for all estimators.
Class design and part of the code is taken and/or influenced
by the Python scikit-learn package, and specifically the
BaseEstimator.py file

# Author: Gilad Green <iml@mail.huji.ac.il>
# License: BSD 3 clause
"""
from __future__ import annotations
from abc import ABC, abstractmethod
from typing import NoReturn
import numpy as np


class BaseEstimator(ABC):
    """
    Base class of supervised estimators (classifiers and regressors)
    """

    def __init__(self) -> BaseEstimator:
        """
        Initialize a supervised estimator instance

        Attributes
        ----------
        fitted_ : bool
            Indicates if estimator has been fitted. Set by ``self.fit`` function
        """
        self.fitted_ = False

    def fit(self, X: np.ndarray, y: np.ndarray) -> BaseEstimator:
        """
        Fit estimator for given input samples and responses

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to fit an estimator for
        y : ndarray of shape (n_samples, )
            Responses of input data to fit to

        Returns
        -------
        self : returns an instance of self.

        Notes
        -----
        After fitting sets ``self.fitted_`` attribute to `True`
        """
        self._fit(X, y)
        self.fitted_ = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
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

        Raises
        ------
        ValueError is raised if ``self.predict`` was called before calling ``self.fit``
        """
        if not self.fitted_:
            raise ValueError("Estimator must first be fitted before calling ``predict``")
        return self._predict(X)

    def loss(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Evaluate performance under loss function specified for estimator

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Test samples

        y : ndarray of shape (n_samples, )
            True labels of test samples

        Returns
        -------
        loss : float
            Performance under loss function specified for estimator

        Raises
        ------
        ValueError is raised if ``self.loss`` was called before calling ``self.fit``
        """
        if not self.fitted_:
            raise ValueError("Estimator must first be fitted before calling ``loss``")
        return self._loss(X, y)

    @abstractmethod
    def _fit(self, X: np.ndarray, y: np.ndarray) -> NoReturn:
        """
        Fit estimator for given input samples and responses

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to fit an estimator for

        y : ndarray of shape (n_samples, )
            Responses of input data to fit to
        """
        raise NotImplementedError()

    @abstractmethod
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
        raise NotImplementedError()

    @abstractmethod
    def _loss(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Evaluate performance under loss function specified for estimator

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Test samples

        y : ndarray of shape (n_samples, )
            True labels of test samples

        Returns
        -------
        loss : float
            Performance under loss function specified for estimator
        """
        raise NotImplementedError()

    def fit_predict(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Fit an estimator over given input data and predict responses for given samples

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to fit an estimator for

        y : ndarray of shape (n_samples, )
            Responses of input data to fit to

        Returns
        -------
        responses : ndarray of shape (n_samples, )
            Predicted responses of given samples
        """
        self.fit(X, y)
        return self.predict(X)
