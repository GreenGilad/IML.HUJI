#################################################################
# FILE : gaussian_estimators.py
# WRITER : Bar Melinarskiy
# EXERCISE : Intro to Machine Learning - 67577 - Exercise 1
# DESCRIPTION: UnivariateGaussian and MultivariateGaussian classes.
#################################################################

from __future__ import annotations
import numpy as np


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
        self.mu_ = UnivariateGaussian._calc_mu(X)
        if self.biased_:
            self.var_ = UnivariateGaussian._calc_unbiased_var(self.mu_, X)
        else:
            self.var_ = UnivariateGaussian._calc_biased_var(self.mu_, X)

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
            raise ValueError("Estimator must first be fitted before calling `pdf` function")
        return UnivariateGaussian._calc_pdf(self.mu_, self.var_, X)

    @staticmethod
    def _calc_pdf(mu: float, var: float, X: np.ndarray) -> np.ndarray:
        """
        Calculate PDF of observations under Gaussian model if random variable X~N(mu, var)
        Parameters
        ----------
        mu : float
            Expectation of Gaussian
        var : float
            Variance of Gaussian
        X: ndarray of shape (n_samples, )
            Samples to calculate PDF for
        Returns
        -------
        pdfs: ndarray of shape (n_samples, )
            Calculated values of given samples for PDF function of N(mu_, var_)
        """
        scalar1 = 1 / np.sqrt(2 * np.pi * var)
        scalar2 = -2 * var
        return scalar1 * np.exp((np.power((X - mu), 2)) / scalar2)

    @staticmethod
    def _calc_mu(X: np.ndarray) -> float:
        """
        Calculate mean estimator of observations under Gaussian model

        Parameters
        ----------
        X: ndarray of shape (n_samples, )
            Samples to calculate mean for

        Returns
        -------
        mean: float
           mean calculated
        """
        if len(X) == 0:  # Don't divide by 0, if the array is empty return mean = 0
            return 0
        return X.sum() / len(X)

    @staticmethod
    def _calc_biased_var(mu: float, X: np.ndarray) -> float:
        """
        Calculate variance estimator of observations under Gaussian model for biased estimator

        Parameters
        ----------
        mu : float
            Expectation of Gaussian
        X: ndarray of shape (n_samples, )
            Samples to calculate mean for

        Returns
        -------
        variance: float
           variance calculated
        """
        if len(X) == 0:  # Don't divide by 0, if the array is empty return var = 0
            return 0
        scalar = 1 / len(X)
        return scalar * np.sum(np.power((X - mu), 2))

    @staticmethod
    def _calc_unbiased_var(mu: float, X: np.ndarray) -> float:
        """
        Calculate variance estimator of observations under Gaussian model for unbiased estimator

        Parameters
        ----------
        mu : float
            Expectation of Gaussian
        X: ndarray of shape (n_samples, )
            Samples to calculate mean for

        Returns
        -------
        variance: float
           variance calculated
        """
        if len(X) <= 1:  # Don't divide by 0, if the array is empty or single sample return var = 0
            return 0
        scalar = 1 / (len(X) - 1)
        return scalar * np.sum(np.power((X - mu), 2))

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
        return np.float(np.sum(np.log(UnivariateGaussian._calc_pdf(mu, sigma, X))))


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

        self.mu_ = MultivariateGaussian._calc_mu(X)
        self.cov_ = MultivariateGaussian._calc_unbiased_cov(self.mu_, X)
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
            raise ValueError("Estimator must first be fitted before calling `pdf` function")
        return MultivariateGaussian._calc_pdf(self.mu_, self.cov_, X)

    @staticmethod
    def log_likelihood(mu: np.ndarray, cov: np.ndarray, X: np.ndarray) -> float:
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

        return np.float(np.sum(np.log(MultivariateGaussian._calc_pdf(mu, cov, X))))

    @staticmethod
    def _calc_mu(X: np.ndarray) -> np.ndarray:
        """
        Calculate multivariate sample mean estimator under Gaussian model

        Parameters
        ----------
        X: ndarray of shape (n_samples, n_features)
            Training data

        Returns
        -------
        mean: ndarray of shape (n_features,)
           mean calculated
        """
        if len(X) == 0:  # Don't divide by 0, if the array is empty return mean = 0
            raise ValueError("Non valid samples were given to fit function, samples size must be > 1")
        return np.mean(X, axis=0)

    @staticmethod
    def _calc_unbiased_cov(mu: np.ndarray, X: np.ndarray) -> np.ndarray:
        """
        Calculate covariance multivariate estimator of observations under Gaussian model for unbiased estimator

        Parameters
        ----------
        mu : ndarray of shape (n_features,)
            Expectation of Gaussian
        X: ndarray of shape (n_samples, n_features)
            Training data

        Returns
        -------
        covariance: ndarray of shape (n_features,)
           covariance calculated
        """
        if len(X) <= 1:  # Don't divide by 0, if the array is empty or single sample return var = 0
            raise ValueError("Non valid samples were given to fit function, samples size must be > 1")
        scalar = 1 / (len(X) - 1)
        m = X - mu
        try:
            cov = np.cov(X.T)  # Reminder: come back here and decide which implementation to choose
            # cov = scalar * m.T @ m
        except:
            try:
                cov = scalar * m @ m.T
            except:
                raise ValueError("Non valid samples were given to pdf")
        return cov

    @staticmethod
    def _calc_pdf(mu: np.ndarray, cov: np.ndarray, X: np.ndarray) -> np.ndarray:
        """
        Calculate PDF of observations under Gaussian model if random variable X~N(mu, var)
        Parameters
        ----------
        mu : ndarray of shape (n_features,)
            Expectation of Gaussian
        cov : ndarray of shape (n_features, n_features)
            covariance matrix of Gaussian
        X: ndarray of shape (n_samples, n_features)
            Training data
        Returns
        -------
        pdfs: ndarray of shape (n_samples, )
            Calculated values of given samples for PDF function of N(mu_, var_)
        """
        m = X - mu
        scalar = 1 / (np.sqrt(np.power(2 * np.pi, len(mu)) * np.linalg.det(cov)))
        pdf = np.zeros(len(X))
        for i in range(len(pdf)):
            pdf[i] = scalar * np.exp((-0.5) * (m[i].T @ np.linalg.inv(cov) @ m[i]))
        return pdf


