from typing import NoReturn
from ...base import BaseEstimator
import numpy as np
from numpy.linalg import det, inv
from IMLearn.metrics.loss_functions import misclassification_error


class LDA(BaseEstimator):
    """
    Linear Discriminant Analysis (LDA) classifier

    Attributes
    ----------
    self.classes_ : np.ndarray of shape (n_classes,)
        The different labels classes. To be set in `LDA.fit`

    self.mu_ : np.ndarray of shape (n_classes,n_features)
        The estimated features means for each class. To be set in `LDA.fit`

    self.cov_ : np.ndarray of shape (n_features,n_features)
        The estimated features covariance. To be set in `LDA.fit`

    self._cov_inv : np.ndarray of shape (n_features,n_features)
        The inverse of the estimated features covariance. To be set in `LDA.fit`

    self.pi_: np.ndarray of shape (n_classes)
        The estimated class probabilities. To be set in `GaussianNaiveBayes.fit`
    """

    def __init__(self):
        """
        Instantiate an LDA classifier
        """
        super().__init__()
        self.classes_, self.mu_, self.cov_, self._cov_inv, self.pi_ = None, None, None, None, None

    def _fit(self, X: np.ndarray, y: np.ndarray) -> NoReturn:
        """
        fits an LDA model.
        Estimates gaussian for each label class - Different mean vector, same covariance
        matrix with dependent features.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to fit an estimator for

        y : ndarray of shape (n_samples, )
            Responses of input data to fit to
        """
        self.classes_ = np.unique(y)

        mu_matrix = []
        for i in self.classes_:
            index = np.where(y == i)
            mu_matrix.append(np.mean(X[index], axis=0))
        self.mu_ = np.array(mu_matrix)

        self.cov_ = np.zeros(shape=(X.shape[1], X.shape[1]))
        for i in range(len(X)):
            index = np.where(y[i] == self.classes_)
            temp_mat = (X[i] - self.mu_[index]).transpose() @ (X[i] - self.mu_[index])
            self.cov_ += temp_mat
        self.cov_ = 1 / (X.shape[0] - self.classes_.shape[0]) * self.cov_

        self._cov_inv = inv(self.cov_)

        pi_lst = []
        for i in self.classes_:
            pi_lst.append((y == i).sum() / len(y))
        self.pi_ = np.array(pi_lst)
        self.fitted_ = True

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
        res_lst = []
        for row in X:
            cur = []
            for i in range(len(self.classes_)):
                a = self._cov_inv @ self.mu_[i]
                b = np.log(self.pi_[i]) - 0.5 * self.mu_[i] @ self._cov_inv @ self.mu_[i]
                cur.append(a.transpose() @ row + b)
            res_lst.append(float(self.classes_[np.where(np.max(cur) == cur)]))
        return np.array(res_lst)

    def likelihood(self, X: np.ndarray) -> np.ndarray:
        """
        Calculate the likelihood of a given data over the estimated model

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Input data to calculate its likelihood over the different classes.

        Returns
        -------
        likelihoods : np.ndarray of shape (n_samples, n_classes)
            The likelihood for each sample under each of the classes
        """
        if not self.fitted_:
            raise ValueError("Estimator must first be fitted before calling `likelihood` function")
        lkhd_lst = []
        for row in X:
            row_lkhd_lst = []
            for k in range(len(self.classes_)):
                exp_part = -0.5 * (row - self.mu_[k]) @ self._cov_inv @ (row - self.mu_[k]).transpose()
                f_x_y = (1 / np.sqrt((2 * np.pi) ** (X.shape[1]) * det(self.cov_))) * np.exp(exp_part)
                row_lkhd_lst.append(f_x_y * self.pi_[k])
            lkhd_lst.append(row_lkhd_lst)
        return np.array(lkhd_lst)

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
        return misclassification_error(y, self.predict(X))
