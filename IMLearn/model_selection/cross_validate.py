from __future__ import annotations
from copy import deepcopy
from typing import Tuple, Callable
import numpy as np
from IMLearn import BaseEstimator


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
    # indexes = np.array_split(np.arange(X.shape[0]), cv, axis=0)
    # train_score = 0
    # validation_score = 0
    # for i in range(cv):
    #     x_train = np.delete(X, indexes[i])
    #     y_train = np.delete(y, indexes[i])
    #     s_x = np.array(X[indexes[i][0]: indexes[i][-1]]).flatten()
    #     s_y = y[indexes[i][0]: indexes[i][-1]]
    #     estimator.fit(x_train, y_train)
    #     train_score += scoring(y_train, estimator.predict(x_train))
    #     validation_score += scoring(s_y, estimator.predict(s_x))
    # return (train_score / cv), (validation_score / cv)

    indexes = np.array_split(np.arange(X.shape[0]), cv, axis=0)
    train_score = 0
    validation_score = 0
    for i in range(cv):
        y_train = np.delete(y, indexes[i])
        s_x = np.array(X[indexes[i][0]: indexes[i][-1]])
        if X.shape[1] > 1:
            x_train = np.delete(X, indexes[i], axis=0)
            s_x = s_x.reshape(indexes[i][-1] - indexes[i][0], X.shape[1])
        else:
            x_train = np.delete(X, indexes[i])
            s_x = s_x.flatten()
        s_y = y[indexes[i][0]: indexes[i][-1]]
        estimator.fit(x_train, y_train)
        train_score += scoring(y_train, estimator.predict(x_train))
        validation_score += scoring(s_y, estimator.predict(s_x))
    return (train_score / cv), (validation_score / cv)




