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
        self.fitted_ = True
        self.classes_ = np.array(np.unique(y))
        self.mu_ = np.zeros(np.unique(y).size * np.shape(X)[1]).reshape(np.unique(y).size, np.shape(X)[1])
        self.pi_ = np.zeros(self.classes_.size)
        self.cov_ = np.cov(np.transpose(X),ddof=self.classes_.size)
        self._cov_inv = np.linalg.inv(self.cov_)

        class_counter = np.zeros(self.classes_.size)

        for x, clas in zip(X, y):
            self.mu_[clas] += x
            class_counter[clas] += 1
            self.pi_[clas] += 1

        for i in range(0, class_counter.size):
            self.mu_[i] = self.mu_[i]*(1/class_counter[i])
            self.pi_[i] = self.pi_[i]*(1/X.shape[0])


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
        probMat = self.likelihood(X)
        response = np.zeros(X.shape[0])

        for sample, loc in zip(probMat, range(0, probMat.size)):
            response[loc] = np.argmax(sample)

        return response

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
        matrix = np.zeros(X.shape[0]*self.classes_.size).reshape(X.shape[0], self.classes_.size)
        det = np.linalg.det(self.cov_)

        for sample, i in zip(X, range(0, X.shape[0])):
            for k in self.classes_:
                coeff = (1 / np.sqrt(np.power(2 * np.pi, X.shape[1]) * det))
                expVal = -(1/2)*((sample-self.mu_[k])@self._cov_inv@(sample-self.mu_[k]))
                expVal = np.exp(expVal)
                matrix[i][k] = (coeff * expVal)
        return matrix

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
