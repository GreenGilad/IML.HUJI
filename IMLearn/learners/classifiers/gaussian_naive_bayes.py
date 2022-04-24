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

        mu_matrix = []
        for i in self.classes_:
            index = np.where(y == i)
            mu_matrix.append(np.mean(X[index], axis=0))
        self.mu_ = np.array(mu_matrix)

        var_matrix = []
        for i in range(len(self.classes_)):
            index = np.where(y == self.classes_[i])
            class_var_lst = []
            for feature in range(X.shape[1]):
                class_var_lst.append(np.var(X[index, feature], ddof=1))
            var_matrix.append(class_var_lst)
        self.vars_ = np.array(var_matrix)

        pi_lst = []
        for i in self.classes_:
            pi_lst.append((y == i).sum() / len(y))
        self.pi_ = np.array(pi_lst)
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
        res_lst = []
        for sample in X:
            class_likly_lst = []
            for k in range(len(self.classes_)):
                multiplied = 1
                for feature in range(len(sample)):
                    nrml = (1 / np.sqrt(2 * np.pi * self.vars_[k, feature])) * np.exp(
                        -(sample[feature] - self.mu_[k, feature]) ** 2 / (2 * self.vars_[k, feature]))
                    multiplied = multiplied * nrml * self.pi_[k]
                class_likly_lst.append(multiplied)
            res_lst.append(class_likly_lst)
        return np.array(res_lst)

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
