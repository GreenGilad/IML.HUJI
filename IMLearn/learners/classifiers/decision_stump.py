from __future__ import annotations
from typing import Tuple, NoReturn
from ...base import BaseEstimator
import numpy as np
from itertools import product


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
        min_loss = None
        x_T = X.T
        for i in range(X.shape[1]):
            threshold_1, loss_1 = self._find_threshold(x_T[i], y, 1)
            threshold_2, loss_2 = self._find_threshold(x_T[i], y, -1)
            if min_loss is None or loss_1 < min_loss:
                min_loss = loss_1
                self.j_ = i
                self.threshold_ = threshold_1
                self.sign_ = 1

            if loss_2 < min_loss:
                min_loss = loss_2
                self.j_ = i
                self.threshold_ = threshold_2
                self.sign_ = -1

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
        x_t = X.T
        return np.where(x_t[self.j_] >= self.threshold_, self.sign_, -self.sign_)

    def _find_threshold(self, values: np.ndarray, labels: np.ndarray, sign: int) -> Tuple[float, float]:
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
        x_sorted = np.argsort(values)
        values_sorted = values[x_sorted]
        labels_sorted = labels[x_sorted]
        labels_by_sign = np.sign(labels_sorted)

        # getting the numbers of labels different from sign with weight
        errors_amount = np.sum(np.abs(labels_sorted) * (sign != labels_by_sign))

        # creating a new array influenced by both sign and labels_sorted
        total = errors_amount - (np.cumsum((labels_sorted * sign)[::-1])[::-1])

        threshold_index = np.argmin(total)
        # making an array of index's
        y_predict = np.arange(labels_sorted.shape[0])
        y_predict = np.where(y_predict < threshold_index, -sign, sign)

        loss = float(np.sum(np.abs(labels_sorted) * (y_predict != labels_by_sign)))
        normalize = np.sum(labels_sorted)
        if normalize != 0:
            loss /= normalize
        threshold = float(values_sorted[threshold_index])
        return threshold, loss


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
        y_predict = self.predict(X)
        labels_by_sign = np.where(y > 0, 1, -1)
        return float(np.sum(np.abs(y) * (y_predict != labels_by_sign)))
