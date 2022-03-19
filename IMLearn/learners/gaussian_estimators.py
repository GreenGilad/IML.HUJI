from __future__ import annotations

import math

import numpy
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
        self.mu_ = np.mean(X)


        # Now calculate the expected variance based on whether the sample is biased or not
        the_divisor = X.size if self.biased_ == True else X.size - 1
        val_minus_mu_squared = lambda t: (t - self.mu_)**2
        sum_of_values_minus_mu_squared = 0
        for val in X:
            sum_of_values_minus_mu_squared += val_minus_mu_squared(val)
        self.var_ = (1/the_divisor)*sum_of_values_minus_mu_squared

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

        calc_pdf = lambda t:\
            (1/math.sqrt(2*math.pi*self.var_))*math.pow(math.e, (-1/2*self.var_)*((t - self.mu_)**2))

        # Apply the lambda function over all the sample values
        map_pdf = np.vectorize(calc_pdf)
        return map_pdf(X)

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
        sum_of_vars_minus_mu_squared = 0
        for var in X:
            sum_of_vars_minus_mu_squared += (var-mu)**2

        return math.log((1/((2*math.pi*(sigma))**(X.size/2)))\
               *math.pow(math.e,(-1/(2*(sigma)))*sum_of_vars_minus_mu_squared))


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

        # Calculating the expectation vector
        self.mu_ = []
        for i in range(X[0].size):
            row_i = X[:, i]
            self.mu_.append(np.mean(row_i))


        # Calculating covariance matrix
        self.cov_ = numpy.cov(X, rowvar=False)

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

        # Calculates the pdf for a multivariate sample vector
        num_of_features = X[0].size
        calc_pdf = lambda t: (1/math.sqrt(((2*math.pi)**num_of_features)*np.linalg.det(self.cov_)))\
        *math.pow(math.e,(-1/2)*(numpy.matmul(numpy.matmul(numpy.transpose(t - self.mu_),
                                                           numpy.linalg.inv(self.cov_)), t-self.mu_)))
        # Map this function over input array
        map_pdf = np.vectorize(calc_pdf)
        return map_pdf(X)

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
        n = X.size
        d = X[0].size
        helper_calculation = lambda t: np.matmul(np.matmul(np.transpose(t - mu), np.linalg.inv(cov)), t - mu)

        the_sum = 0
        for var in X:
            the_sum += helper_calculation(var)

        return (-n)*(d/2)*math.log(2*math.pi) - (n/2)*math.log(np.linalg.det(cov)) - (1/2)*the_sum