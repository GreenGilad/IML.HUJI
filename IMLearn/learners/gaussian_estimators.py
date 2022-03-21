from __future__ import annotations
import math
import numpy as np
from numpy.linalg import inv, det, slogdet

import plotly.graph_objects as go
import plotly.io as pio


class UnivariateGaussian:
    """
    Class for univariate Gaussian Distribution Estimator
    """

    def __init__(self, biased_var: bool = True, ) -> UnivariateGaussian:
        """
        Estimator for univariate Gaussian mean and variance parameters

        Parameters
        ----------
        biased_var : bool, default=True
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
        self.fitted_ = True
        self.mu_ = np.mean(X)
        self.var_ = np.var(X)
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
        pdf_func = lambda x: (1 / (self.mu_ * math.sqrt(2 * math.pi))) \
                             * math.exp(-math.pow(x - self.mu_, 2) / (2 * self.var_))
        return np.array(list(map(pdf_func, X)))

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
        return X.size / 2 * math.log(sigma * sigma) - 1 / (2 * sigma * sigma) * (np.sum(math.pow(X - mu, 2)))


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
        self.fitted_ = True
        self.mu_ = np.mean(X, axis=0)
        X_centered = X - self.mu_
        self.cov_ = 1 / (X.shape[0] - 1) * np.dot(np.transpose(X), X_centered)
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

        pdf_func = lambda x: math.exp(self.log_likelihood(self.mu_, self.cov_, X))
        X_centered = X - self.mu_
        return math.pow(np.linalg.det(self.cov_ * 2 * math.pi), -0.5) \
               * math.exp(-0.5 * (np.linalg.multi_dot([X_centered.transpose(),
                                                       np.linalg.inv(self.cov_),
                                                       X_centered])))

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
        X_centered = X - mu
        m = X.shape[0]
        d = X.shape[1]
        return -m * d / 2 * math.log(2 * math.pi) - m / 2 * math.log(np.linalg.det(cov)) \
               - 0.5 * (np.linalg.multi_dot([X_centered.transpose(),
                                             np.linalg.inv(cov),
                                             X_centered]))


def targil_3_1_1():
    univariate_normal = UnivariateGaussian()
    data = np.random.normal(10, 1, 1000)
    univariate_normal.fit(data)
    print("(" + str(univariate_normal.mu_) + "," + str(univariate_normal.var_) + ")")


def targil_3_1_2():
    sample_sizes = range(10, 1000, 10)
    distances = []
    univariate_normal = UnivariateGaussian()
    mu = 10
    for sample_size in sample_sizes:
        data = np.random.normal(mu, 1, sample_size)
        univariate_normal.fit(data)
        distances.append(univariate_normal.mu_ - mu)

    fig = go.Figure(data=go.Scatter(x=np.array(range(10, 1000, 10)), y=distances))
    fig.update_layout(title="Distance from estimated expectation to real expectation as an output of sample size",
                      xaxis_title="Sample size",
                      yaxis_title="Distance from estimated expectation to real expectation")
    fig.show()


def targil_3_1_3():
    data = np.random.normal(10, 1, 1000)
    univariate_normal = UnivariateGaussian()
    univariate_normal.fit(data)
    y = univariate_normal.pdf(data)
    fig = go.Figure(go.Scatter(x=data, y=y,  mode='markers'))
    fig.update_layout(
                title="PDFs values as an output of sample values for Univariate Gaussian with expectation 10, variance 1",
                xaxis_title="Sample values",
                yaxis_title="PDFs values")
    fig.show()


def targil_3_2_4():
    data = np.random.multivariate_normal(np.array([0, 0, 4, 0]),
                                         np.array([[1, 0.2, 0, 0.5],
                                                   [0.2, 2, 0, 0],
                                                   [0, 0, 1, 0],
                                                   [0.5, 0, 0, 1]]),
                                         1000)
    multivariate_normal = MultivariateGaussian()
    multivariate_normal.fit(data)
    print(multivariate_normal.mu_)
    print(multivariate_normal.cov_)


def targil_3_2_5():
    f1 = np.linspace(-10, 10, 200)
    f3 = np.linspace(-10, 10, 200)
    sigma = np.array([[1, 0.2, 0, 0.5],
                      [0.2, 2, 0, 0],
                      [0, 0, 1, 0],
                      [0.5, 0, 0, 1]])
    # data = np.random.multivariate_normal(np.array([0, 0, 4, 0]),
    #                                      np.array([[1, 0.2, 0, 0.5],
    #                                                [0.2, 2, 0, 0],
    #                                                [0, 0, 1, 0],
    #                                                [0.5, 0, 0, 1]]),
    #                                      1000)
    # results = [MultivariateGaussian.log_likelihood(np.array([i, 0, j, 0]).transpose(), sigma, data)
    #            for i in f1 for j in f3]
    # results = np.array(results)
    # fig = go.imshow(results,
    #                 labels=dict(x="f1", y="f3", color="Productivity"),
    #                 x=range_f1,
    #                 y=range_f3)
    # fig.update_xaxes(side="top")
    # fig.show()


# targil_3_1_1()
# targil_3_1_2()
# targil_3_1_3()
# targil_3_2_4()
#targil_3_2_5()
print("finished!")
