from __future__ import annotations
from copy import deepcopy
from typing import Tuple, Callable
import numpy as np
from IMLearn import BaseEstimator
from IMLearn.utils import split_train_test


def cross_validate(estimator: BaseEstimator, X: np.ndarray, y: np.ndarray,
                   scoring: Callable[[np.ndarray, np.ndarray, ...], float], cv: int = 5) -> Tuple[float, float]:
    """
    Evaluate metric by cross-validation for given estimator

    Parameters
    ----------
    estimator: BaseEstimator
        Initialized estimator to use for fitting the data

    X: ndarray of shape (n_samples, n_features)
       Input data to fit

    y: ndarray of shape (n_samples, )
       Responses of input data to fit to

    scoring: Callable[[np.ndarray, np.ndarray, ...], float]
        Callable to use for evaluating the performance of the cross-validated model.
        When called, the scoring function receives the true- and predicted values for each sample
        and potentially additional arguments. The function returns the score for given input.

    cv: int
        Specify the number of folds.

    Returns
    -------
    train_score: float
        Average train score over folds

    validation_score: float
        Average validation score over folds
    """
    train_loss_lst = []
    test_loss_lst = []
    m = len(y)
    for i in range(cv):
        train_x = np.concatenate((X[:int(m * (i / cv))], X[int(m * ((i + 1) / cv)):]), axis=0)
        train_y = np.concatenate((y[:int(m * (i / cv))], y[int(m * ((i + 1) / cv)):]), axis=0)
        test_x = X[int(m * (i / cv)):int(m * ((i + 1) / cv))]
        test_y = y[int(m * (i / cv)):int(m * ((i + 1) / cv))]

        estimator.fit(train_x, train_y)
        train_loss_lst.append(scoring(train_y, estimator.predict(train_x)))
        test_loss_lst.append(scoring(test_y, estimator.predict(test_x)))
    return np.average(train_loss_lst), np.average(test_loss_lst)
