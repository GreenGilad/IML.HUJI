from typing import NoReturn
from ...base import BaseEstimator
import numpy as np
from IMLearn.metrics import misclassification_error


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
        self.classes_, self.mu_, self.vars_, self.pi_ = None, None, None, None

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
        num_classes = self.classes_.shape[0]
        num_features = X.shape[1]
        num_samples = X.shape[0]

        # creating pi, vars, mu as an empty np.array in the fit size
        self.pi_ = np.zeros(num_classes)
        self.vars_ = np.zeros((num_classes, num_features))
        self.mu_ = np.zeros((num_classes, num_features))

        for index, class_name in enumerate(self.classes_):
            x_class = X[y == class_name]
            self.vars_[index] = np.var(x_class, axis=0, ddof=1)
            self.mu_[index] = np.mean(x_class, axis=0)
            self.pi_[index] = x_class.shape[0] / num_samples

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
        return self.classes_[np.argmax(self.likelihood(X), axis=1)]

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

        likelihood = np.zeros((X.shape[0], self.classes_.shape[0]))
        for k in range(self.classes_.shape[0]):
            total_val = self.pi_[k]
            for d in range(X.shape[1]):
                a = -np.power(X[:, d] - self.mu_[k, d], 2) / (2 * self.vars_[k, d])
                b = 1 / ((np.sqrt(2 * np.pi)) * np.sqrt(self.vars_[k, d]))
                total_val *= b * np.exp(a)
            likelihood[:, k] = total_val
        return likelihood

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
        return misclassification_error(y, self._predict(X))
