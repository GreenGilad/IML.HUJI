from __future__ import annotations
import numpy as np
from numpy.linalg import inv, det, slogdet


class UnivariateGaussian:
    """
    Class for univariate Gaussian Distribution Estimator
    """

    def __init__(self, biased_var: bool = False) -> UnivariateGaussian:
        """
        Estimator for univariate Gaussian mean and variance parameters

        Parameters
        ----------
        biased_var : bool, default=False
            Should fitted estimator of variance be a biased or unbiased estimator

        Attributes
        ----------
        fitted_ : bool
            Initialized as false indicating current estimator instance has not been fitted.
            To be set as True in `UnivariateGaussian.fit` function.

        mu_: float
            Estimated expectation initialized as None. To be set in `UnivariateGaussian.fit`
            function.

        var_: float
            Estimated variance initialized as None. To be set in `UnivariateGaussian.fit`
            function.
        """
        self.biased_ = biased_var
        self.fitted_, self.mu_, self.var_ = False, None, None

    def fit(self, X: np.ndarray) -> UnivariateGaussian:
        """
        Estimate Gaussian expectation and variance from given samples

        Parameters
        ----------
        X: ndarray of shape (n_samples, )
            Training data

        Returns
        -------
        self : returns an instance of self.

        Notes
        -----
        Sets `self.mu_`, `self.var_` attributes according to calculated estimation (where
        estimator is either biased or unbiased). Then sets `self.fitted_` attribute to `True`
        """

        # estimate the expected value and standard deviation of the samples
        # than set the instance parameters accordingly:
        self.mu_ = np.mean(X)  # todo make sure
        self.var_ = np.std(X) ** 2

        if self.biased_:
            self.var_ = (self.var_ * (X.shape[0] - 1)) / X.shape[0] # todo oposite

        # self.var_ = (1/m or m-1) * ((X-mu)**2).sum()

        self.fitted_ = True
        return self

    def pdf(self, X: np.ndarray) -> np.ndarray:
        """
        Calculate PDF of observations under Gaussian model with fitted estimators

        Parameters
        ----------
        X: ndarray of shape (n_samples, )
            Samples to calculate PDF for

        Returns
        -------
        pdfs: ndarray of shape (n_samples, )
            Calculated values of given samples for PDF function of N(mu_, var_)

        Raises
        ------
        ValueError: In case function was called prior fitting the model
        """
        if not self.fitted_:
            raise ValueError(
                "Estimator must first be fitted before calling `pdf` function")

        # calculate and return the pdf array:
        return (1 / np.sqrt(2 * np.pi * self.var_)) * np.exp(
            -0.5 * ((X - self.mu_) / np.sqrt(self.var_)) ** 2)

    @staticmethod
    def log_likelihood(mu: float, sigma: float, X: np.ndarray) -> float:
        """
        Calculate the log-likelihood of the data under a specified Gaussian model

        Parameters
        ----------
        mu : float
            Expectation of Gaussian
        sigma : float
            Variance of Gaussian
        X : ndarray of shape (n_samples, )
            Samples to calculate log-likelihood with

        Returns
        -------
        log_likelihood: float
            log-likelihood calculated
        """

        # because of the log, the product becomes sum.
        return np.log((1 / np.sqrt(2 * np.pi * sigma)) * np.exp(
            -0.5 * ((X - mu) / np.sqrt(sigma)) ** 2)).sum()

        # return (-0.5 * sigma) * ((X - mu) ** 2).sum() - (X.shape[0] / 2) * np.log(2 * np.pi * sigma)


class MultivariateGaussian:
    """
    Class for multivariate Gaussian Distribution Estimator
    """

    def __init__(self):
        """
        Initialize an instance of multivariate Gaussian estimator

        Attributes
        ----------
        fitted_ : bool
            Initialized as false indicating current estimator instance has not been fitted.
            To be set as True in `MultivariateGaussian.fit` function.

        mu_: ndarray of shape (n_features,)
            Estimated expectation initialized as None. To be set in `MultivariateGaussian.fit`
            function.

        cov_: ndarray of shape (n_features, n_features)
            Estimated covariance initialized as None. To be set in `MultivariateGaussian.fit`
            function.
        """
        self.mu_, self.cov_ = None, None
        self.fitted_ = False

    def fit(self, X: np.ndarray) -> MultivariateGaussian:
        """
        Estimate Gaussian expectation and covariance from given samples

        Parameters
        ----------
        X: ndarray of shape (n_samples, n_features)
            Training data

        Returns
        -------
        self : returns an instance of self

        Notes
        -----
        Sets `self.mu_`, `self.cov_` attributes according to calculated estimation.
        Then sets `self.fitted_` attribute to `True`
        """

        self.mu_ = np.mean(X, axis=0)  # todo check bias
        self.cov_ = (1 / (X.shape[0] - 1)) * np.dot((X - self.mu_).T, (X - self.mu_))
        self.fitted_ = True
        return self

    def pdf(self, X: np.ndarray):
        """
        Calculate PDF of observations under Gaussian model with fitted estimators

        Parameters
        ----------
        X: ndarray of shape (n_samples, n_features)
            Samples to calculate PDF for

        Returns
        -------
        pdfs: ndarray of shape (n_samples, )
            Calculated values of given samples for PDF function of N(mu_, cov_)

        Raises
        ------
        ValueError: In case function was called prior fitting the model
        """
        if not self.fitted_:
            raise ValueError(
                "Estimator must first be fitted before calling `pdf` function")

        # calculate and return the pdf array:

        return (1 / np.sqrt(((2 * np.pi) ** X.shape[1]) * np.linalg.det(
                self.cov_))) * np.exp(-0.5 * (
                    (X - self.mu_) * np.matrix(self.cov_.I) * (X - self.mu_).T))

    @staticmethod
    def log_likelihood(mu: np.ndarray, cov: np.ndarray,
                       X: np.ndarray) -> float:
        """
        Calculate the log-likelihood of the data under a specified Gaussian model

        Parameters
        ----------
        mu : ndarray of shape (n_features,)
            Expectation of Gaussian
        cov : ndarray of shape (n_features, n_features)
            covariance matrix of Gaussian
        X : ndarray of shape (n_samples, n_features)
            Samples to calculate log-likelihood with

        Returns
        -------
        log_likelihood: float
            log-likelihood calculated over all input data and under given parameters of Gaussian
        """
        # todo ok?

        # return np.log((1 / np.sqrt(((2 * np.pi) ** X.shape[1]) * np.linalg.det(
        #     cov))) * np.exp(-0.5 * ((X - mu) * np.matrix(cov).I * (X - mu).T))).sum()

        # return -np.log((np.sqrt(((2 * np.pi) ** X.shape[1]) * np.linalg.det(cov)))) * (X.shape[0] / 2) + (-0.5 * ((X - mu) * np.matrix(cov).I * (X - mu).T)).sum()
        return (-0.5) * (np.log((np.sqrt(((2 * np.pi) ** X.shape[1]) * np.linalg.det(cov)))) * (X.shape[0]) + (np.dot((X - mu), inv(cov)) * (X - mu)).sum())

        # return multivariate_normal.logpdf(X, mu, cov).sum()
