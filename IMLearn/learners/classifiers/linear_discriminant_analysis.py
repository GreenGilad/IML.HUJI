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
        mu = []
        for i in self.classes_:
            index = np.where(y == i)
            mu.append(np.mean(X[index], axis=0))
        self.mu_ = np.array(mu)
        self.cov_ = np.zeros(shape=(X.shape[1], X.shape[1]))
        m = np.shape(X)[0]
        k = len(self.classes_)
        for i in range(m):
            mu_i = self.mu_[np.where(self.classes_ == y[i])]
            self.cov_ += (X[i] - mu_i).T @ (X[i] - mu_i)
        self.cov_ = (1 / (m - k)) * self.cov_
        self._cov_inv = inv(self.cov_)
        pi = []
        for i in self.classes_:
            # print(len(np.where(y == i)[0]) == (y == i).sum())
            pi.append(len(np.where(y == i)[0]) / m)
        self.pi_ = np.array(pi)

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
        m = np.shape(X)[0]
        y_hat = []
        for sample in range(m):
            all_k = []
            c = 0
            for mu_k in self.mu_:
                akt_x = (self._cov_inv @ mu_k)
                bk = np.log(self.pi_[c]) - 0.5 * mu_k @ self._cov_inv @ mu_k
                all_k.append(akt_x @ X[sample] + bk)
                c += 1
            y_hat.append(self.classes_[all_k.index(max(all_k))])
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
        k = len(self.classes_)
        d = np.shape(X)[1]
        likelihood_array = []
        for sample in X:
            sample_likelihood = []
            for i in range(k):
                in_exp = -0.5 * (sample - self.mu_[i]) @ self._cov_inv @ (sample - self.mu_[i]).T
                fx_y = (1 / np.sqrt(((2 * np.pi) ** d) * det(self.cov_))) * np.exp(in_exp)
                fy = self.pi_[i]
                sample_likelihood.append(fx_y * fy)
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
