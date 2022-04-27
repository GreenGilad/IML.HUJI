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


    def create_var_array(self, X: np.ndarray, y: np.ndarray):
        """
        Returns the variance array for features by class
        """
        expected_variances = np.ndarray((self.classes_.shape[0], X.shape[1]))
        for c in self.classes_:
            samples_with_c_label = X[y == c]
            expected_variances[c, :] = np.var(samples_with_c_label, axis=0)
        return expected_variances

    def create_mean_array(self, X: np.ndarray, y: np.ndarray):
        """
        Returns the expected values matrix for each of the classes
        """
        expected_values = np.ndarray((self.classes_.shape[0], X.shape[1]))
        for c in self.classes_:
            samples_with_c_label = X[y == c]
            expected_values[c, :] = np.mean(samples_with_c_label, axis=0)
        return expected_values

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
        classes, counts = np.unique(y, return_counts=True)
        self.classes_ = classes
        self.pi_ = counts / y.shape[0]
        self.mu_ = self.create_mean_array(X,y)
        self.vars_ = self.create_var_array(X,y)


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


    def calculate_likelihood_for_sample(self,sample:np.array, c: int):
        """
        Calculates the likelihood for each of the given samples for input class c
        """
        # For the calculation we start with the initial prob set to the relevant pi_k
        the_prob = self.pi_[c]
        for feature in range(len(sample)):
            sigma_c = self.vars_[c][feature]
            mu_c = self.mu_[c][feature]
            lower_term = (1 / np.sqrt(2 * np.pi * sigma_c))
            exponent_term = (-1 / (2 * sigma_c)) * ((sample[feature] - mu_c) ** 2)
            the_prob *= lower_term*np.exp(exponent_term)
        return the_prob



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

        sample_num, feature_num = X.shape[0], X.shape[1]
        likelihoods = np.ndarray((sample_num, self.classes_.shape[0]))

        # Calculating the likelihood for each of the classes
        for c in self.classes_:
             likelihoods[:, c] = np.apply_along_axis(self.calculate_likelihood_for_sample,1,X, c)
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
