from typing import NoReturn

import scipy.linalg

from ...base import BaseEstimator
import numpy as np
from numpy.linalg import det, inv

# todo maybe remove
from IMLearn.learners.gaussian_estimators import MultivariateGaussian


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
        # initialize estimator:
        prob = MultivariateGaussian()

        # initialize class labels:
        self.classes_ = np.unique(y).astype(int)

        prob.fit(X)
        self.mu_ = prob.mu_
        self.cov_ = prob.cov_
        self._cov_inv = inv(self.cov_)

        # # initialize arrays to fill:
        self.pi_ = np.zeros(len(self.classes_))
        self.mu_ = np.zeros((len(self.classes_), X.shape[1]))
        # self.cov_ = np.zeros((len(self.classes_), X.shape[1], X.shape[1]))
        # self._cov_inv = np.zeros((len(self.classes_), X.shape[1], X.shape[1]))
        #
        for _class in self.classes_:
            # Get the indices of the samples belonging to the current class
            class_indices = np.where(y == _class)[0]

            # Calculate the class probability
            self.pi_[_class] = len(class_indices) / len(y)

            #     # Get the samples belonging to the current class
            #
            # prob.fit(class_samples)
            #
            #     # Calculate the mean of the current class
            class_samples = X[class_indices]
            self.mu_[_class] = np.mean(class_samples, axis=0)
        #     self.mu_[_class] = prob.mu_
        #
        #     # Calculate the covariance of the current class
        #     # self.cov_[_class] = np.cov(class_samples.T)
        #     self.cov_[_class] = prob.cov_
        #
        #     # Calculate the inverse of the covariance of the current class
        #     self._cov_inv[_class] = inv(self.cov_[_class])

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
        res = np.zeros(X.shape[0])
        log_pi = np.log(self.pi_)
        X = X.reshape((X.shape[0], X.shape[1], 1))
        for i, x in enumerate(X):
            res[i] = np.argmax(
                [log_pi[k] +
                 x.T @ self._cov_inv @ self.mu_[k] -
                 0.5 * self.mu_[k] @ self._cov_inv @ self.mu_[k]
                 for k in self.classes_])

        return res


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
            raise ValueError(
                "Estimator must first be fitted before calling `likelihood` function")

        raise NotImplementedError()

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
        raise NotImplementedError()
