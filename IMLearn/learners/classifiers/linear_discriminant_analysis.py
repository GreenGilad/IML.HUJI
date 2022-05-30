from typing import NoReturn

import numpy as np
from numpy.linalg import det, inv
from ...base import BaseEstimator


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
        n_features = X.shape[1]
        n_samples = X.shape[0]
        self.classes_ = np.unique(y)
        n_classes = self.classes_.shape[0]

        self.pi_ = np.zeros(n_classes)
        self.mu_ = np.zeros((n_classes, n_features))
        self.cov_ = np.zeros((n_features, n_features))

        for i in range(n_classes):
            self.mu_[i] = np.mean(X[y == self.classes_[i]], axis=0)
            self.pi_[i] = np.sum(y==self.classes_[i]) / n_samples
        for i in range(n_samples):
            self.cov_ += np.outer(X[i] - self.mu_[y[i]], X[i] - self.mu_[y[i]])

        self.cov_ /= (n_samples- n_classes)
        self._cov_inv = inv(self.cov_)

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
        likelihood = self. likelihood(X)
        return self.classes_[np.argmax(likelihood, axis=1)]


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
        likelihoods = np.zeros((X.shape[0], self.classes_.shape[0]))
        for k in range(self.classes_.shape[0]):
            a = self.mu_[k] @ self._cov_inv
            b = np.log(self.pi_[k]) - 0.5* self.mu_[k] @ self._cov_inv @ self.mu_[k].T
            for x in range(X.shape[0]):
                likelihoods[x,k] = a@X[x].T +b
        return likelihoods


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
        from ...metrics import loss_functions
        return loss_functions.misclassification_error(y, self.predict(X))