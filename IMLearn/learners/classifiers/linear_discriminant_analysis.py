from typing import NoReturn
from IMLearn.base import BaseEstimator
# from ...base import BaseEstimator
import numpy as np
import pandas as pd
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
        # Initializing empty arrays to be filled by fit
        self.classes_ = np.unique(y)
        classes_count = len(self.classes_)
        feature_count = X.shape[1]
        self.mu_ = np.zeros((classes_count, feature_count))
        self.cov_ = np.zeros((feature_count, feature_count))
        self.pi_ = np.zeros(classes_count)
        # Fit values
        # TODO: Try to do this without loop. Maybe with pandas df?
        for index, cls in enumerate(self.classes_):
            X_class_idx = X[y==cls]
            self.mu_[index] = X_class_idx.mean(axis=0)
            cls_cov = X_class_idx - self.mu_[index, :]
            self.cov_ += cls_cov.transpose() @ cls_cov
            # print(self.cov_)
            self.pi_[index] = np.count_nonzero(y==cls) / X.shape[0]
        self.cov_ = self.cov_ / (X.shape[0] - classes_count)
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
        return self.classes_[np.argmax(self.likelihood(X), axis=1)]

    def _calc_lh_by_sample(self, sample):
        classes_count = len(self.classes_)
        sample_lh = np.zeros(classes_count)
        for index, _ in enumerate(self.classes_):
            curr_mu, curr_pi = self.mu_[index], self.pi_[index]
            sample_lh[index] = (self._cov_inv @ curr_mu).transpose() @ sample
            sample_lh[index] += (np.log(curr_pi) - 0.5 * (curr_mu @ self._cov_inv @ curr_mu))
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