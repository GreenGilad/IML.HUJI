import numpy as np
from typing import Tuple
from IMLearn.learners.classifiers import DecisionStump
from IMLearn.metalearners.adaboost import AdaBoost
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


def my_decision_surface(t, predict, xrange, yrange, density=120, dotted=False, colorscale=custom, showscale=True):
    xrange, yrange = np.linspace(*xrange, density), np.linspace(*yrange, density)
    xx, yy = np.meshgrid(xrange, yrange)
    pred = predict(np.c_[xx.ravel(), yy.ravel()], t)

    if dotted:
        return go.Scatter(x=xx.ravel(), y=yy.ravel(), opacity=1, mode="markers",
                          marker=dict(color=pred, size=1, colorscale=colorscale, reversescale=False), hoverinfo="skip",
                          showlegend=False)
    return go.Contour(x=xrange, y=yrange, z=pred.reshape(xx.shape), colorscale=colorscale, reversescale=False,
                      opacity=.7, connectgaps=True, hoverinfo="skip", showlegend=False, showscale=showscale)


def fit_and_evaluate_adaboost(noise, n_learners=250, train_size=5000, test_size=500):
    # Question 1: Train- and test errors of AdaBoost in noiseless case
    (train_X, train_y), (test_X, test_y) = generate_data(5000, noise), generate_data(500, noise)
    adab = AdaBoost(DecisionStump, 250)
    adab.fit(train_X, train_y)

    train_loss_lst = []
    test_loss_lst = []
    for i in range(1, adab.iterations_ + 1):
        train_loss_lst.append(adab.partial_loss(train_X, train_y, i))
        test_loss_lst.append(adab.partial_loss(test_X, test_y, i))

    fig = make_subplots(rows=1, cols=1, subplot_titles=['loss over iterations'])
    fig.add_trace(go.Scatter(x=list(range(1, 251)), y=train_loss_lst, name='train_loss', mode="lines",
                             marker=dict(color='black')), row=1, col=1)
    fig.add_trace(
        go.Scatter(x=list(range(1, 251)), y=test_loss_lst, name='test_loss', mode="lines", marker=dict(color='red')),
        row=1, col=1)
    fig.write_html('Q1_' + str(noise) + '.html', auto_open=True)

    # Question 2: Plotting decision surfaces
    T = [5, 50, 100, 250]
    model_names = ['adaboost T=5', 'adaboost T=50', 'adaboost T=100', 'adaboost T=250']

    lims = np.array([np.r_[train_X, test_X].min(axis=0), np.r_[train_X, test_X].max(axis=0)]).T + np.array([-.1, .1])

    fig = make_subplots(rows=2, cols=2, subplot_titles=[m for m in model_names],
                        horizontal_spacing=0.04, vertical_spacing=0.07)

    for i, t in enumerate(T):
        fig.add_traces([my_decision_surface(t, adab.partial_predict, lims[0], lims[1], showscale=False),
                        go.Scatter(x=test_X[:, 0], y=test_X[:, 1], mode="markers", showlegend=False,
                                   marker=dict(color=test_y, colorscale=[custom[0], custom[-1]],
                                               line=dict(color="black", width=1)))],
                       rows=(i // 2) + 1, cols=(i % 2) + 1)
    fig.write_html('Q2_' + str(noise) + '.html', auto_open=True)

    # Question 3: Decision surface of best performing ensemble
    min_loss_iter = np.argmin(test_loss_lst)
    fig = make_subplots(rows=1, cols=1, subplot_titles=[
        'Size: ' + str(min_loss_iter + 1) + ', Accuracy: ' + str(1 - test_loss_lst[min_loss_iter])])

    fig.add_traces([my_decision_surface(min_loss_iter + 1, adab.partial_predict, lims[0], lims[1], showscale=False),
                    go.Scatter(x=test_X[:, 0], y=test_X[:, 1], mode="markers", showlegend=False,
                               marker=dict(color=test_y, colorscale=[custom[0], custom[-1]],
                                           line=dict(color="black", width=1)))], rows=1, cols=1)
    fig.write_html('Q3_' + str(noise) + '.html', auto_open=True)

    # Question 4: Decision surface with weighted samples
    d_norm = (adab.D_ / np.max(adab.D_)) * 5
    fig = make_subplots(rows=1, cols=1, subplot_titles=['weighted plot & decision surface'])

    fig.add_traces([decision_surface(adab.predict, lims[0], lims[1], showscale=False),
                    go.Scatter(x=train_X[:, 0], y=train_X[:, 1], mode="markers", showlegend=False,
                               marker=dict(color=train_y, colorscale=[custom[0], custom[-1]],
                                           line=dict(color="black", width=1), size=d_norm))], rows=1, cols=1)
    fig.write_html('Q4_' + str(noise) + '.html', auto_open=True)


if __name__ == '__main__':
    np.random.seed(0)
    fit_and_evaluate_adaboost(0)
    fit_and_evaluate_adaboost(0.4)
