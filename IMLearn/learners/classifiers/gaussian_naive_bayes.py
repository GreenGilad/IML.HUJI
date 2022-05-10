from typing import NoReturn
from ...base import BaseEstimator
import numpy as np

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
        classes_count = len(self.classes_)
        feature_count = X.shape[1]
        self.mu_ = np.zeros((classes_count, feature_count))
        self.vars_ = np.zeros((classes_count, feature_count))
        self.pi_ = np.zeros(classes_count)
        # Fit values
        # TODO: Try to do this without loop. Maybe with pandas df?
        for index, cls in enumerate(self.classes_):
            X_class_idx = X[y==cls]
            self.mu_[index] = np.mean(X_class_idx, axis=0)
            self.vars_[index] = np.var(X_class_idx, axis=0)
            self.pi_[index] = np.count_nonzero(y==cls) / X.shape[0]

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

    def _calc_lh_by_sample(self, sample):
        classes_count = len(self.classes_)
        sample_lh = np.zeros(classes_count)
        for index, _ in enumerate(self.classes_):
            sum_ = np.sum(((sample - self.mu_[index])**2) / (self.vars_[index]*2))
            sample_lh[index] = np.log(self.pi_[index]) - sum_
        return sample_lh

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

        classes_count = len(self.classes_)
        samples_count = X.shape[0]

        # TODO: Try to do this without loop
        lh = np.zeros((samples_count, classes_count))
        for sample_idx in range(samples_count):
            lh[sample_idx] = self._calc_lh_by_sample(X[sample_idx])
        return lh

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
        return misclassification_error(y, self.predict(X))
