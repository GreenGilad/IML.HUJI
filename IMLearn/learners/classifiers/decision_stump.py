from __future__ import annotations

from itertools import product
from typing import Tuple, NoReturn

import numpy as np

from ...base import BaseEstimator


class DecisionStump(BaseEstimator):
    """
    A decision stump classifier for {-1,1} labels according to the CART algorithm

    Attributes
    ----------
    self.threshold_ : float
        The threshold by which the data is split

    self.j_ : int
        The index of the feature by which to split the data

    self.sign_: int
        The label to predict for samples where the value of the j'th feature is about the threshold
    """

    def __init__(self) -> DecisionStump:
        """
        Instantiate a Decision stump classifier
        """
        super().__init__()
        self.threshold_, self.j_, self.sign_ = None, None, None

    def _fit(self, X: np.ndarray, y: np.ndarray) -> NoReturn:
        """
        fits a decision stump to the given data

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to fit an estimator for

        y : ndarray of shape (n_samples, )
            Responses of input data to fit to
        """
        self.sign_ = 1
        self.j_ = 0
        self.threshold_ = 0
        X_T = X.T
        min_err = 2

        for i, sign in product(range(X.shape[1]), [-1, 1]):
            thr, thr_err = self._find_threshold(X_T[i], y, sign)
            if min_err > thr_err:
                self.threshold_ = thr
                min_err = thr_err
                self.j_ = i
                self.sign_ = sign

    def _predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict responses for given samples using fitted estimator

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to predict responses for

        y : ndarray of shape (n_samples, )
            Responses of input data to fit to

        Returns
        -------
        responses : ndarray of shape (n_samples, )
            Predicted responses of given samples

        Notes
        -----
        Feature values strictly below threshold are predicted as `-sign` whereas values which equal
        to or above the threshold are predicted as `sign`
        """
        X_T = X.T
        return np.where(X_T[self.j_] >= self.threshold_, self.sign_,
                        -self.sign_)

    def _find_threshold(self, values: np.ndarray, labels: np.ndarray,
                        sign: int) -> Tuple[float, float]:
        """
        Given a feature vector and labels, find a threshold by which to perform a split
        The threshold is found according to the value minimizing the misclassification
        error along this feature

        Parameters
        ----------
        values: ndarray of shape (n_samples,)
            A feature vector to find a splitting threshold for

        labels: ndarray of shape (n_samples,)
            The labels to compare against

        sign: int
            Predicted label assigned to values equal to or above threshold

        Returns
        -------
        thr: float
            Threshold by which to perform split

        thr_err: float between 0 and 1
            Misclassificaiton error of returned threshold

        Notes
        -----
        For every tested threshold, values strictly below threshold are predicted as `-sign` whereas values
        which equal to or above the threshold are predicted as `sign`
        """
        sorted_indices = np.argsort(values)
        sorted_val = values[sorted_indices]
        sorted_labels = labels[sorted_indices]
        signed_labels = np.sign(sorted_labels)
        errors = np.sum(
            (signed_labels != sign) * np.abs(sorted_labels))
        sum = errors - (np.cumsum((sign * sorted_labels)[::-1])[::-1])
        index = np.argmin(sum)
        y_pred = np.arange(sorted_labels.shape[0])
        y_pred = np.where(y_pred < index, -sign, sign)
        a = np.sum(np.abs(values))
        mse = float(
            np.sum((y_pred != signed_labels) * np.abs(
                sorted_labels)))\
              # / np.sum(np.abs(values))
        thr = float(sorted_val[index])
        return thr, mse

    def _loss(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Evaluate performance under misclassification loss function

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Test samples

        y : ndarray of shape (n_samples, )
            True labels of test samples

        Returns
        -------
        loss : float
            Performance under missclassification loss function
        """
        signed_labels = np.where(y > 0, 1, -1)
        y_pred = self.predict(X)
        mse = float(
            np.sum((y_pred != signed_labels) * np.abs(y)))
        return mse
