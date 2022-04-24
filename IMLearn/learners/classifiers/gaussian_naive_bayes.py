from typing import NoReturn
from ...base import BaseEstimator
import numpy as np
from IMLearn.metrics.loss_functions import misclassification_error


class GaussianNaiveBayes(BaseEstimator):
    """
    Gaussian Naive-Bayes classifier
    """

    def __init__(self):
        """
        Instantiate a Gaussian Naive Bayes classifier

        Attributes
        ----------
        self.classes_ : np.ndarray of shape (n_classes,)
            The different labels classes. To be set in `GaussianNaiveBayes.fit`

        self.mu_ : np.ndarray of shape (n_classes,n_features)
            The estimated features means for each class. To be set in `GaussianNaiveBayes.fit`

        self.vars_ : np.ndarray of shape (n_classes, n_features)
            The estimated features variances for each class. To be set in `GaussianNaiveBayes.fit`

        self.pi_: np.ndarray of shape (n_classes)
            The estimated class probabilities. To be set in `GaussianNaiveBayes.fit`
        """
        super().__init__()
        self.classes_, self.mu_, self.vars_, self.pi_, self.fitted_ = None, None, None, None, False

    def _fit(self, X: np.ndarray, y: np.ndarray) -> NoReturn:
        """
        fits a gaussian naive bayes model

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to fit an estimator for

        y : ndarray of shape (n_samples, )
            Responses of input data to fit to
        """
        self.classes_ = np.unique(y)
        m = np.shape(X)[0]
        k = len(self.classes_)

        mu = []
        for i in self.classes_:
            index = np.where(y == i)
            mu.append(np.mean(X[index], axis=0))
        self.mu_ = np.array(mu)
        pi = []
        for i in self.classes_:
            pi.append(len(np.where(y == i)[0]) / m)
        self.pi_ = np.array(pi)

        self.vars_ = np.zeros(shape=(k, np.shape(X)[1]))
        for i in range(k):
            indexes = np.where(y == self.classes_[i])
            for j in range(np.shape(X)[1]):
                self.vars_[i, j] = np.var(X[indexes[0], int(j)], ddof=1)
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
        likelihood_mat = self.likelihood(X)
        y_hat = []
        for sample_like in likelihood_mat:
            y_hat.append(self.classes_[np.where(sample_like == max(sample_like))][0])
        return np.array(y_hat)

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
        gaussian_pdf = lambda x, k: np.prod(
            [self.pi_[k] * (1 / np.sqrt(2 * np.pi * self.vars_[k, j])) * np.exp(
                -((x[j] - self.mu_[k, j]) ** 2) / (2 * self.vars_[k, j]))
             for j in range(x.size)])
        likelihood_array = []
        for x in X:
            sample_likelihood = []
            for k in range(len(self.classes_)):
                sample_likelihood.append(gaussian_pdf(x, k))
            likelihood_array.append(sample_likelihood)
        return np.array(likelihood_array)

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
        y_pred = self._predict(X)
        return misclassification_error(y, y_pred)
