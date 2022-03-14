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
        self.mu_ = np.mean(X)
        self.var_ = np.var(X)  # todo where does bias play out
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
        return (1 / np.sqrt(2 * np.pi * self.var_ ** 2)) * np.exp(-(X - self.mu_) ** 2 / (2 * self.var_ ** 2))

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
        # applying pdf to all of the n samples
        applied_pdf = (1 / np.sqrt(2 * np.pi * sigma ** 2)) * np.exp(-(X - mu) ** 2 / (2 * sigma ** 2))

        # taking the log
        return np.log(applied_pdf)


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
        raise NotImplementedError()

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
        raise NotImplementedError()

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
        raise NotImplementedError()


def practical_3_1():
    # 1000 samples taken from ~ N(10, 1)
    X = np.random.normal(10, 1, 1000)
    univariate_gaussian = UnivariateGaussian()
    univariate_gaussian.fit(X)
    print("Expectation: " + str(univariate_gaussian.mu_) + ", Variance: " + str(univariate_gaussian.var_))


def practical_3_2():
    mu = 10
    univariate_gaussian = UnivariateGaussian()
    absolute_distance = []
    for sample_size in range(10, 1010, 10):
        X = np.random.normal(mu, 1, sample_size)
        univariate_gaussian.fit(X)
        absolute_distance.append(np.abs(univariate_gaussian.mu_ - mu))

    import plotly.express as px
    fig = px.line(x=range(10, 1010, 10),
                  y=np.asarray(absolute_distance),
                  labels={"x": "Sample Size", "y": "Distance of Estimated Expectation from True Value"},
                  title='Distance between the estimated and true value of the Expectation', )
    fig.show()


def practical_3_3():
    X = np.sort(np.random.normal(10, 1, 1000))
    univariate_gaussian = UnivariateGaussian()
    univariate_gaussian.fit(X)
    import matplotlib.pyplot as plt
    from matplotlib import rcParams
    plt.scatter(x=X, y=univariate_gaussian.pdf(X), s=rcParams['lines.markersize'] ** 2 / 10)
    mu, sigma = univariate_gaussian.mu_, univariate_gaussian.var_
    x = np.linspace(mu - 3 * sigma, mu + 3 * sigma, 100)
    plt.plot(x, (1 / np.sqrt(2 * np.pi * sigma ** 2)) * np.exp(-(x - mu) ** 2 / (2 * sigma ** 2)), 'r-')
    plt.show()
    print("hello")


practical_3_3()
