import numpy as np
from typing import Tuple

import pandas as pd
import matplotlib.pyplot as plt

import utils
from IMLearn.metalearners.adaboost import AdaBoost
from IMLearn.learners.classifiers import DecisionStump
from utils import *
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def generate_data(n: int, noise_ratio: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate a dataset in R^2 of specified size

    Parameters
    ----------
    n: int
        Number of samples to generate

    noise_ratio: float
        Ratio of labels to invert

    Returns
    -------
    X: np.ndarray of shape (n_samples,2)
        Design matrix of samples

    y: np.ndarray of shape (n_samples,)
        Labels of samples
    """
    '''
    generate samples X with shape: (num_samples, 2) and labels y with shape (num_samples).
    num_samples: the number of samples to generate
    noise_ratio: invert the label for this ratio of the samples
    '''
    X, y = np.random.rand(n, 2) * 2 - 1, np.ones(n)
    y[np.sum(X ** 2, axis=1) < 0.5 ** 2] = -1
    y[np.random.choice(n, int(noise_ratio * n))] *= -1
    return X, y


def fit_and_evaluate_adaboost(noise, n_learners=250, train_size=5000, test_size=500, accurecy=None):
    (train_X, train_y), (test_X, test_y) = generate_data(train_size, noise), generate_data(test_size, noise)

    # Question 1: Train- and test errors of AdaBoost in noiseless case
    adaboost = AdaBoost(DecisionStump, 250)
    adaboost.fit(train_X, train_y)

    train_error = []
    test_error = []
    for i in range(1, n_learners + 1):
        train_error.append(adaboost.partial_loss(train_X, train_y, i))
        test_error.append(adaboost.partial_loss(test_X, test_y, i))
    scale = list(range(1, n_learners))
    fig = go.Figure([go.Scatter(x=scale, y=train_error, mode="lines+markers", name="Train Error"),
                     go.Scatter(x=scale, y=test_error, mode="lines+markers", name="Test Error")],
                     layout=go.Layout(
                         title=rf"$\textbf{{The training & test errors as a function of the number of fitted learners}}$",
                         xaxis_title="The number of fitted learners",
                         yaxis_title="training & test errors",
                         height=500))
    fig.show()

    # Question 2: Plotting decision surfaces
    T = [5, 50, 100, 250]
    lims = np.array([np.r_[train_X, test_X].min(axis=0), np.r_[train_X, test_X].max(axis=0)]).T + np.array([-.1, .1])

    fig = make_subplots(rows=2, cols=2,
                        subplot_titles=[rf"$\textbf{{{t} num of adaboost week learners}}$" for t in T],
                        horizontal_spacing=0.03, vertical_spacing=0.08)
    for index, num_of_iter in enumerate(T):
        def predict_function(X: np.ndarray):
            return adaboost.partial_predict(X, num_of_iter)

        fig.add_traces([utils.decision_surface(predict_function, lims[0], lims[1], showscale=True),
                        go.Scatter(x=test_X[:, 0], y=test_X[:, 1], mode="markers",
                                   showlegend=False,
                                   marker=dict(color=test_y, size=3,
                                   line=dict(color="black", width=0.2)))],
                       rows=(index // 2) + 1, cols=(index % 2) + 1)
        fig.update_layout(xaxis1_range=[-1, 1], yaxis1_range=[-1, 1],
                          xaxis2_range=[-1, 1], yaxis2_range=[-1, 1],
                          xaxis3_range=[-1, 1], yaxis3_range=[-1, 1],
                          xaxis4_range=[-1, 1], yaxis4_range=[-1, 1])


    fig.update_layout(
        title=rf"$\textbf{{The decision boundaries of the test by different adaboost iteration noise is {noise}}}$")
    fig.show()

    # Question 3: Decision surface of best performing ensemble
    # first I will find the best accuracy
    best_t = int(np.argmin(np.array(test_error)))
    best_accuracy = 1 - test_error[best_t]
    best_t += 1

    def predict_function(X: np.ndarray):
        return adaboost.partial_predict(X, best_t)

    go.Figure([utils.decision_surface(predict_function, lims[0], lims[1]),
              go.Scatter(x=test_X[:, 0], y=test_X[:, 1], mode="markers", showlegend=False,
                         marker=dict(color=test_y, size=8,
                         line=dict(color="black", width=0.2)))]).update_layout(xaxis_range=[-1, 1], yaxis_range=[-1, 1],
                         title=
                         rf"$\textbf{{The decision boundaries of the test by of adaboost with "
                         rf"{best_t} iteration the best accuracy {best_accuracy} noise {noise}}}$").show()

    # Question 4: Decision surface with weighted samples
    normalize = 5 * adaboost.D_ / np.max(adaboost.D_)
    colors = np.array(['blue', 'red'])
    normalize_D = np.array(normalize)
    train_y = np.where(train_y > 0, 1, 0).astype(int)

    go.Figure([utils.decision_surface(predict_function, lims[0], lims[1], showscale=False),
              go.Scatter(x=train_X[:, 0], y=train_X[:, 1], mode="markers", showlegend=False,
                         marker=dict(color=train_y, size=normalize_D * 2,
                         line=dict(color=colors[train_y], width=0.2)))]).update_layout(
        xaxis_range=[-1, 1], yaxis_range=[-1, 1],
        title=rf"$\textbf{{The decision boundaries of the train by of adaboost with "
              rf"{best_t} iteration the size of each instance influenced by the fit D noise {noise}}}$").show()


if __name__ == '__main__':
    np.random.seed(0)
    fit_and_evaluate_adaboost(0)
    fit_and_evaluate_adaboost(0.4)
