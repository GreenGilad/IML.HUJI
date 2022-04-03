from __future__ import annotations
from typing import NoReturn
from IMLearn.base import BaseEstimator
import numpy as np
from sklearn import svm
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsRegressor

from sklearn import metrics


class AgodaCancellationEstimator(BaseEstimator):
    """
    An estimator for solving the Agoda Cancellation challenge
    """

    def __init__(self) -> AgodaCancellationEstimator:
        """
        Instantiate an estimator for solving the Agoda Cancellation challenge

        Parameters
        ----------


        Attributes
        ----------

        """
        super().__init__()
        self.model_svm = svm.SVC(kernel='linear', C=1.0, max_iter=5000)
        self.model_lr = LogisticRegression(solver='liblinear', random_state=0)
        self.model_knn = KNeighborsRegressor(n_neighbors=15)

    def _fit(self, X: np.ndarray, y: np.ndarray) -> NoReturn:
        """
        Fit an estimator for given samples

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to fit an estimator for

        y : ndarray of shape (n_samples, )
            Responses of input data to fit to

        Notes
        -----

        """
        self.model_svm.fit(X, y)
        self.model_lr.fit(X, y)
        self.model_knn.fit(X, y)

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
        self.pre_svm = self.model_svm.predict(X)
        self.pre_lr = self.model_lr.predict(X)
        self.pre_knn = np.round(self.model_knn.predict(X))
        # print("uniq lr:", np.unique(self.pre_lr), "svm:", np.unique(self.pre_svm), "knn: ", np.unique(self.pre_knn))
        return self.pre_lr

    def _loss(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Evaluate performance under loss function

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Test samples

        y : ndarray of shape (n_samples, )
            True labels of test samples

        Returns
        -------
        loss : float
            Performance under loss function
        """
        # print("SVM:", self.model.score(X, y))
        # print("LogisticRegression:", self.model2.score(X, y))
        # print("% of 1 = ", round(np.sum(y) / len(y), 3))
        # print("lr: ", round(lose_fun(self.pre_lr, y), 3), "svm: ", round(lose_fun(self.pre_svm, y), 3), "knn: ",
        #       round(lose_fun(self.pre_knn, y), 3))
        return 1 - self.model_lr.score(X, y)


def lose_fun(y_esty, y_reL):
    return np.sum(np.power(y_esty - y_reL, 2)) / len(y_esty)
