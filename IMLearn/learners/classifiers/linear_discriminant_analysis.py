from typing import NoReturn
from ...base import BaseEstimator
import numpy as np
from IMLearn.metrics import misclassification_error


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
        num_classes = self.classes_.shape[0]
        num_features = X.shape[1]
        num_samples = X.shape[0]

        # creating pi, vars, mu as an empty np.array in the fit size
        self.pi_ = np.zeros(num_classes)
        self.cov_ = np.zeros((num_features, num_features))
        self.mu_ = np.zeros((num_classes, num_features))

        for index, class_name in enumerate(self.classes_):
            x_class = X[y == class_name]
            self.mu_[index] = np.mean(x_class, axis=0)
            self.pi_[index] = x_class.shape[0] / num_samples
            self.cov_ += (x_class - self.mu_[index]).T.dot(x_class - self.mu_[index])

        # asked for unbiased
        self.cov_ /= num_samples - num_classes
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
        z = np.power(np.pi * 2, X.shape[1])

        for k in range(self.classes_.shape[0]):
            likelihood[:, k] = self.pi_[k] * (1 / z) * \
                              np.exp(np.diag(-0.5 * (X - self.mu_[k]) @ self._cov_inv @ (X - self.mu_[k]).T))
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
<<<<<<< HEAD
        from ...metrics import misclassification_error
        raise NotImplementedError()
=======
        return misclassification_error(y, self._predict(X))
>>>>>>> 31ed4e80e7298ea676f090fc9fc1cbe965ba35ec
