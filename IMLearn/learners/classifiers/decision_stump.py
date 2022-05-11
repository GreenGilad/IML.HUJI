from __future__ import annotations
from typing import Tuple, NoReturn
from IMLearn.base import BaseEstimator
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
        signs = [1, -1]
        min_error=np.inf
        features = list(range(X.shape[1]))
        combines_signs_features = product(signs, features)
        for sign, feature in combines_signs_features:
            curr_threshold, curr_error = self._find_threshold(X[:, feature], y, sign)
            if curr_error < min_error:
                self.threshold_ = curr_threshold
                self.j_ = feature
                self.sign_ = sign
                min_error = curr_error


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
        x_j=X[:,self.j_]
        responses=np.full(X.shape[0], self.sign_)
        responses[x_j<self.threshold_]=-self.sign_
        return responses

    @staticmethod
    def misclassification_error(y, y_pred):
        """
        return: the weighted loss of the predicted labels
        """
        weighted_loss = np.sum(np.where(np.sign(y) != np.sign(y_pred), np.abs(y), 0))
        return float(weighted_loss)


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
        arr=np.vstack([values,labels])
        arr=arr[:,arr[0,:].argsort()]
        best_threshold=0
        min_error=np.inf

        for i in range(len(values)):
            pred=np.zeros(values.shape)
            pred[:i]=-sign
            pred[i:]=sign
            curr_error=DecisionStump.misclassification_error(arr[1], pred)
            if curr_error<=min_error:
                min_error=curr_error
                best_threshold=arr[0][i]
        return best_threshold,min_error


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
        return DecisionStump.misclassification_error(y, self.predict(X))

