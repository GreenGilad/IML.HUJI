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

    # indexes = np.array_split(np.arange(X.shape[0]), cv, axis=0)
    # train_score = 0
    # validation_score = 0
    # for i in range(cv):
    #     y_train = np.delete(y, indexes[i])
    #     s_x = np.array(X[indexes[i][0]: indexes[i][-1]])
    #     if X.shape[1] > 1:
    #         x_train = np.delete(X, indexes[i], axis=0)
    #         s_x = s_x.reshape(indexes[i][-1] - indexes[i][0], X.shape[1])
    #     else:
    #         x_train = np.delete(X, indexes[i])
    #         s_x = s_x.flatten()
    #     s_y = y[indexes[i][0]: indexes[i][-1]]
    #     estimator.fit(x_train, y_train)
    #     train_score += scoring(y_train, estimator.predict(x_train))
    #     validation_score += scoring(s_y, estimator.predict(s_x))
    # return (train_score / cv), (validation_score / cv)

    train_scores, validation_scores = [], []
    index_to_split = [i for i in range(0, X.shape[0])]
    index_of_groups = np.array_split(index_to_split, cv)
    groups_data = [X[v.flatten().tolist()] for v in index_of_groups]
    groups_response = [y[v.flatten().tolist()] for v in index_of_groups]
    for j in range(0, cv):
        cur_data = np.array([])
        cur_response = np.array([])
        for k in range(0, cv):
            if k == j:
                continue
            if cur_data.size == 0:
                cur_data = groups_data[0]
                cur_response = groups_response[0]
            else:
                cur_data = np.concatenate((cur_data, groups_data[k]))
                cur_response = np.concatenate((cur_response, groups_response[k]))
        cur_data = np.array(cur_data)
        cur_response = np.array(cur_response)
        estimator.fit(cur_data, cur_response)
        train_scores.append(scoring(cur_response, estimator.predict(cur_data)))
        validation_scores.append(scoring(np.array(groups_response[j]),
                                         estimator.predict(np.array(groups_data[j]))))

    avg_train_score = np.sum(train_scores) / cv
    avg_val_score = np.sum(validation_scores) / cv
    return avg_train_score, avg_val_score


