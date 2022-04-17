from __future__ import annotations
from typing import Callable
from typing import NoReturn
from ...base import BaseEstimator

import random
import numpy as np


from IMLearn.metrics.loss_functions import misclassification_error

def default_callback(fit: Perceptron, x: np.ndarray, y: int):
    pass


class Perceptron(BaseEstimator):
    """
    Perceptron half-space classifier

    Finds a separating hyperplane for given linearly separable data.

    Attributes
    ----------
    include_intercept: bool, default = True
        Should fitted model include an intercept or not

    max_iter_: int, default = 1000
        Maximum number of passes over training data

    coefs_: ndarray of shape (n_features,) or (n_features+1,)
        Coefficients vector fitted by Perceptron algorithm. To be set in
        `Perceptron.fit` function.

    training_loss_: array of floats
        holds the loss value of the algorithm during training.
        training_loss_[i] is the loss value of the i'th training iteration.
        to be filled in `Perceptron.fit` function.

    """
    def __init__(self,
                 include_intercept: bool = True,
                 max_iter: int = 1000,
                 callback: Callable[[Perceptron, np.ndarray, int], None] = default_callback):
        """
        Instantiate a Perceptron classifier

        Parameters
        ----------
        include_intercept: bool, default=True
            Should fitted model include an intercept or not

        max_iter: int, default = 1000
            Maximum number of passes over training data

        callback: Callable[[Perceptron, np.ndarray, int], None]
            A callable to be called after each update of the model while fitting to given data
            Callable function should receive as input a Perceptron instance, current sample and current response

        Attributes
        ----------
        include_intercept_: bool
            Should fitted model include an intercept or not

        max_iter): int, default = 1000
            Maximum number of passes over training data

        callback_: Callable[[Perceptron, np.ndarray, int], None]
            A callable to be called after each update of the model while fitting to given data
            Callable function should receive as input a Perceptron instance, current sample and current response

        coefs_: ndarray of shape (n_features,) or (n_features+1,)
            Coefficients vector fitted by Perceptron. To be set in `Perceptron.fit` function.
        """
        super().__init__()
        self.include_intercept_ = include_intercept
        self.max_iter_ = max_iter
        self.callback_ = callback
        self.coefs_ = None

    def _fit(self, X: np.ndarray, y: np.ndarray) -> NoReturn:
        """
        Fit a halfspace to to given samples. Iterate over given data as long as there exists a sample misclassified
        or that did not reach `self.max_iter_`

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to fit an estimator for

        y : ndarray of shape (n_samples, )
            Responses of input data to fit to

        Notes
        -----
        Fits model with or without an intercept depending on value of `self.fit_intercept_`
        """
        self.fitted_ = True

        # Initialize weights vector:
        self.coefs_ = np.ones(X.shape[1] + int(self.include_intercept_))
        X = np.c_[np.ones((X.shape[0], 1)), X] if self.include_intercept_ else X

        # samples = np.c_[X, y.reshape(-1, 1)]
        # np.random.shuffle(samples)

        # X = samples[:, :-1]
        # y = samples[:, -1]


        # calculate weights vector:
        # Iterate until no weights change or max_iter is reached
        for t in range(self.max_iter_):
            print(f'iter: {t}')

            # iterate over samples and check if they are misclassified
            misclassifications = \
            [i for i in range(X.shape[0]) if y[i] * self.predict(X[i]) <= 0]

            random.shuffle(misclassifications)
            if not misclassifications:
                print("No misclassifications found, stopping training")
                return

            print(f'\t- misclassifications: {len(misclassifications)}')

            for i in misclassifications:
                if (y[i] * self.predict(X[i])) <= 0:
                    # if misclassified, update weights vector

                    self.coefs_ += (y[i] * X[i])
                    self.callback_(self, X, y)
                    break

                #
                #
                # # calculate predicted response
                # y_hat = np.sign(np.dot(self.coefs_, x_i))
                #
                # # if misclassified, update weights
                # if y_hat != y[i]:
                #     self.coefs_ += y[i] * x_i

            # samples = np.c_[X, y.reshape(-1, 1)]
            # np.random.shuffle(samples)
            #
            # # for i, (x_, y_) in enumerate(zip(X, y)):
            # for i, sample in enumerate(samples):
            #     x_ = sample[:-1]
            #     y_ = sample[-1]
            #     # print(f"Iteration {t}/{self.max_iter_}: Checking sample {i}, x: {[round(s, 3) for s in x_]}, response {y_}")
            #     # if y_ * self.predict(x_) <= 0:
            #     if np.dot(y, np.dot(X, self.coefs_)) <= 0:
            #         print(f'self.coefs_ {self.coefs_}, y * x {y_ * x_}')
            #         self.coefs_ += y_ * x_
            #         # self.callback_(self, x_, y_)
            #         self.callback_(self, X, y)
            #         continue
            #         # continue
            #
            #     if i == len(X) - 1:
            #         # Stop if no misclassification is found
            #         return

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
        # todo: intercept, activation function ?
        # X = np.hstack((np.ones((X.shape[0], 1)), X)) if self.include_intercept_ else X
        return np.sign(np.dot(X, self.coefs_))

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
        return misclassification_error(self.predict(X), y)
