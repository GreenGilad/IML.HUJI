from __future__ import annotations

import math

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
        # raise NotImplementedError()
        self.mu_ = np.mean(X)
        self.var_ = X.var(ddof=1) if not self.biased_ else X.var() # default unbiased
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
        resArr = np.ndarray(X.size,)
        i = 0
        for sample in X:
            powerToTheVar = self.var_**2
            divFactor = 1 / (math.sqrt(2*math.pi*powerToTheVar))
            exp = math.exp((-(sample-self.mu_)**2)/(2*powerToTheVar))
            resArr[i] = divFactor*exp
            i+=1
        return resArr

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
        sum = 0
        # calculate sum of (xi - mu )**2:
        for sample in X:
            sum += (sample - mu)**2
        multFactor = 1 / (2 * sigma)
        logOfAddfactor = X.size * math.log((1 / (math.sqrt(2*math.pi*sigma)))
                                           , math.e)
        return logOfAddfactor - multFactor*sum



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

        mu_: float
            Estimated expectation initialized as None. To be set in `MultivariateGaussian.ft`
            function.

        cov_: float
            Estimated covariance initialized as None. To be set in `MultivariateGaussian.ft`
            function.
        """
        self.mu_, self.cov_ = None, None
        self.fitted_ = False

    def fit(self, X: np.ndarray) -> MultivariateGaussian:
        """
        Estimate Gaussian expectation and covariance from given samples

        Parameters
        ----------
        X: ndarray of shape (n_samples, )
            Training data

        Returns
        -------
        self : returns an instance of self.

        Notes
        -----
        Sets `self.mu_`, `self.cov_` attributes according to calculated estimation.
        Then sets `self.fitted_` attribute to `True`
        """
        self.mu_ = X.mean(axis=0)
        self.cov_ = np.cov(X.T)
        self.fitted_ = True
        return self

    def pdf(self, X: np.ndarray):
        """
        Calculate PDF of observations under Gaussian model with fitted estimators

        Parameters
        ----------
        X: ndarray of shape (n_samples, )
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
        pdfs = np.ndarray((X.size//X[0].size)) # initialize output array
        sizeOfSingleSample = X.shape[1]
        detCov = np.linalg.det(self.cov_)
        inverseCov = np.linalg.inv(self.cov_)

        multFactor =  1 / (math.sqrt((2*math.pi)**(sizeOfSingleSample)*detCov))
        i = 0
        for singleSample in X:
            xMinusMu = singleSample - self.mu_
            xMinusMuTranspose = xMinusMu.reshape(xMinusMu.shape + (1,))
            exp = math.exp(-0.5*(xMinusMu @ inverseCov @ xMinusMuTranspose))
            pdfs[i] = multFactor * exp
            i += 1
        return pdfs

    @staticmethod
    def log_likelihood(mu: np.ndarray, cov: np.ndarray, X: np.ndarray) -> float:
        """
        Calculate the log-likelihood of the data under a specified Gaussian model

        Parameters
        ----------
        mu : float
            Expectation of Gaussian
        cov : float
            covariance matrix of Gaussian
        X : ndarray of shape (n_samples, )
            Samples to calculate log-likelihood with

        Returns
        -------
        log_likelihood: float
            log-likelihood calculated
        """
        # each row in the X array represents a single sample.

        # cov_inverse = np.linalg.inv(cov)
        # cov_det = np.linalg.det(cov)
        # e_power = 0
        # pi_d = (2 * math.pi) ** X.shape[1]
        # for x in X:
        #     x_mu = x - mu
        #     mat_multiply = x_mu @ cov_inverse @ (x_mu.reshape(x_mu.shape + (1,)))
        #     e_power += -0.5 * mat_multiply
        #
        # return X.shape[0] * math.log(1 / math.sqrt(pi_d * cov_det)) + e_power[0]



        numberOfSamples = X.shape[0]
        sizeOfSingleSample = X.shape[1]
        detCov = np.linalg.det(cov)
        inverseCov = np.linalg.inv(cov)
        # logArg = numberOfSamples * math.log( 1 /
        #                                      (math.sqrt
        #                                       ((2*math.pi)**sizeOfSingleSample
        #                                        *(detCov))),math.e)
        twoTimesPi = (2 * math.pi) ** sizeOfSingleSample
        logArg = numberOfSamples * math.log(1 /
                                            (math.sqrt
                                             (twoTimesPi * (detCov))), math.e)
        sum = 0
        for i in range (numberOfSamples):
            xMinusMu = X[i] - mu
            xMinusMuTranspose = xMinusMu.reshape(xMinusMu.shape+(1,))
            sum += xMinusMu @ inverseCov @ xMinusMuTranspose
        sum *= -0.5
        return logArg + sum[0]