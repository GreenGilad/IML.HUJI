from typing import NoReturn
from ...base import BaseEstimator
import numpy as np
from numpy.linalg import det, inv
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
        # fit the linear discriminant analysis model
        self.classes_ = np.unique(y)
        self.pi_ = np.zeros(self.classes_.shape)
        self.mu_ = np.zeros((self.classes_.shape[0], X.shape[1]))
        self.cov_ = np.zeros((X.shape[1], X.shape[1]))
        # calculate the mean vector for each class
        for i in range(self.classes_.shape[0]):
            self.mu_[i] = np.mean(X[y == self.classes_[i]], axis=0)
            self.pi_[i] = np.sum(y == self.classes_[i]) / y.shape[0]
        for i in range(X.shape[0]):
            self.cov_ += np.outer(X[i] - self.mu_[y[i]], X[i] - self.mu_[y[i]])
        self.cov_ /= X.shape[0] - self.classes_.shape[0]
        self._cov_inv = inv(self.cov_)
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
        # predict the responses for given samples
        responses = np.zeros(X.shape[0])
        likelihood = np.zeros((X.shape[0], self.classes_.shape[0])) # different option
        likelihood = self.likelihood(X)
        responses = self.classes_[np.argmax(likelihood, axis=1)]
        # for i in range(X.shape[0]):
        #     prob = np.zeros(self.classes_.shape[0])
        #     for j in range(self.classes_.shape[0]):
        #         a_k = np.dot(self._cov_inv, self.mu_[j]).T
        #         b_k = np.log(self.pi_[j]) - 0.5 * np.dot(self.mu_[j], np.dot(self._cov_inv, self.mu_[j]))
        #         prob[j] = a_k @ X[i] + b_k
        #     responses[i] = self.classes_[np.argmax(prob)]
        return responses

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
        # calculate the likelihood of a given data over the estimated model
        if not self.fitted_:
            raise ValueError("Estimator must first be fitted before calling `likelihood` function")
        likelihoods = np.zeros((X.shape[0], self.classes_.shape[0]))
        for i in range(X.shape[0]):
            for j in range(self.classes_.shape[0]): # check if the pdf is right
                mahalanobis = (X[i] - self.mu_[j]) @ self._cov_inv @ (X[i] - self.mu_[j]).T
                gauss_pdf = np.exp(-.5 * mahalanobis) /  np.sqrt((2 * np.pi) ** len(X[i]) * det(self.cov_))
                likelihoods[i, j] = self.pi_[j] * gauss_pdf
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
        from ...metrics import misclassification_error
        return misclassification_error(self.predict(X), y)



