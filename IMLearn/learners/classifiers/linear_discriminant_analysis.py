from typing import NoReturn
from ...base import BaseEstimator
import numpy as np
from numpy.linalg import det, inv


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

    def create_mean_array(self, X: np.ndarray, y: np.ndarray):
        """
        Returns the expected values matrix for each of the classes
        """
        expected_values = np.ndarray((self.classes_.shape[0], X.shape[1]))
        for c in self.classes_:
            samples_with_c_label = X[y == c]
            expected_values[c, :] = np.mean(samples_with_c_label, axis=0)
        return expected_values


    def create_covar_matrix(self, X: np.ndarray, y: np.ndarray):
        """
        Returns the covariance matrix for the given sample data
        """
        cov_dimension = X.shape[1]
        sum_matrix = np.zeros((cov_dimension, cov_dimension))
        for c in self.classes_:
            samples_with_c_label = X[y == c]
            sum_matrix += (samples_with_c_label - self.mu_[c]).T @ (samples_with_c_label - self.mu_[c])
        return sum_matrix / y.shape[0]

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
        classes, counts = np.unique(y, return_counts=True)

        self.classes_ = classes
        self.pi_ = counts / y.shape[0]
        self.mu_ = self.create_mean_array(X, y)
        self.cov_ = self.create_covar_matrix(X, y)
        self._cov_inv = np.linalg.inv(self.cov_)

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
        likelihood_matrix = self.likelihood(X)
        return np.argmax(likelihood_matrix, axis=1)

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

        sample_num = X.shape[0]
        likelihoods = np.ndarray((sample_num, self.classes_.shape[0]))

        # Calculating the likelihood for each of the classes
        for c in self.classes_:
            current_mean = self.mu_[c]
            a_c = self._cov_inv @ current_mean
            b_c = np.log(self.pi_[c]) - (1 / 2) * (current_mean @ a_c)
            likelihoods[:, c] = X @ a_c + b_c
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
        predictions = self.predict(X)
        return misclassification_error(y, predictions)
