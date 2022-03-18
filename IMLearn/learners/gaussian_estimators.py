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
        my_sum = 0
        my_sum_squared = 0
        count = 0
        for v in X:
            my_sum += v
            my_sum_squared += v * v
            count += 1
        if not self.biased_:
            self.mu_ = my_sum / count
            self.var_ = (my_sum_squared / count) - pow(self.mu_, 2)
        else:
            self.mu_ = my_sum / (count - 1)
            self.var_ = (my_sum_squared / (count - 1)) - pow(self.mu_, 2)
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

        count = 0
        ndarr = np.zeros(X.size)
        for i in X:
            ndarr[count] = (1 / np.sqrt(2 * np.pi * self.var_)
                            ) * np.power(np.e, -np.power(i - self.mu_, 2
                                                         ) / self.var_ * 2)
            count += 1
        return ndarr

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
        times = (-1 / (2 * sigma ** 2)) * np.sum(np.power(np.subtract(X, mu), 2))
        coef = 1 / ((2 * np.pi * sigma ** 2) ** (np.shape(X)[0] / 2))
        return np.log(coef) + times


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
        count, sigma = 0, 0
        row, col = np.shape(X)
        covMat = np.ndarray((col, col))
        self.mu_ = np.ndarray(col, )

        for i in np.transpose(X):
            sigma = np.sum(i)
            self.mu_[count] = sigma / row
            count += 1

        for i in range(0, col):
            for j in range(0, col):
                sijma = 0
                for k in range(0, row):
                    sijma += (X[k][i] - self.mu_[i]) * (X[k][j] - self.mu_[j])
                covMat[i][j] = sijma / (row - 1)

        self.cov_ = covMat
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

        covDet = np.abs(np.linalg.det(self.cov_))
        row, col = np.shape(X)
        coef = 1 / np.sqrt(np.power(2 * np.pi, col)*covDet)
        ansArr = np.ndarray(row)
        count = 0
        for i in X:
            vec = i - self.mu_
            val = np.dot(np.dot(vec, np.invert(covDet)), np.transpose(vec))
            val = np.exp(-0.5 * val) * coef
            ansArr[count] = val
            count += 1

        return ansArr

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
        covDet = np.linalg.det(cov)
        row, col = np.shape(X)
        # coef = 1 / np.sqrt(np.power(2 * np.pi, col) * covDet)
        ansArr = np.ndarray(row)
        count = 0
        ans = 0
        invMat = np.linalg.inv(cov)
        for i in X:
            pdf_vec = np.subtract(i, mu)
            val = np.dot(np.dot(pdf_vec, invMat), pdf_vec)
            ansArr[count] = val
            count += 1

        for i in ansArr:
            ans += i
        ans *= -0.5
        return ans
