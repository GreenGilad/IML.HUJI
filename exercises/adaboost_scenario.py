import numpy as np
from typing import Tuple
from IMLearn.metalearners.adaboost import AdaBoost
from IMLearn.learners.classifiers import DecisionStump
from utils import *
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px

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


def fit_and_evaluate_adaboost(noise, n_learners=250, train_size=5000, test_size=500):
    (train_X, train_y), (test_X, test_y) = generate_data(train_size, noise), generate_data(test_size, noise)

    # Question 1: Train- and test errors of AdaBoost in noiseless case
    Ada = AdaBoost(DecisionStump, n_learners)
    Ada.fit(train_X, train_y)

    graph = px.line(x=np.arange(n_learners), y=[Ada.partial_loss(test_X, test_y, t) for t in range(n_learners)],
                    title="AdaBoost Loss by Number of Learners", labels={'x': "no. learners", 'y': "misclassification error"})
    graph.show()
    graph.write_image(f"ex4_plots/AdaBoost_nlearners_noise={noise}.png")

    # Question 2: Plotting decision surfaces
    T = [5, 50, 100, 250]
    lims = np.array([np.r_[train_X, test_X].min(axis=0), np.r_[train_X, test_X].max(axis=0)]).T + np.array([-.1, .1])
    symbols = np.array(["circle", "x"])
    fig = make_subplots(rows=2, cols=2,
                        subplot_titles=[rf"$\textbf{{{t}}}$ iterations" for t
                                        in T],
                        horizontal_spacing=0.01, vertical_spacing=.03)
    lowest_err_t = (0, np.inf)
    for i, t in enumerate(T):
        fig.add_traces([decision_surface(lambda X: Ada.partial_predict(X, t), lims[0], lims[1],
                              showscale=False),
             go.Scatter(x=test_X[:, 0], y=test_X[:, 1], mode="markers",
                        showlegend=False,
                        marker=dict(color=test_y.astype(int),
                                    symbol=symbols[np.array((test_y + 1) / 2).astype(int)],
                                    colorscale=[custom[0],
                                                custom[-1]],
                                    line=dict(color="black",
                                              width=1)))],
            rows=(i // 2) + 1, cols=(i % 2) + 1)
        cur_err = Ada.partial_loss(test_X, test_y, t)
        if cur_err <= lowest_err_t[1]:
            lowest_err_t = (t, cur_err)

    fig.update_layout(title=rf"$\textbf{{Decision Boundaries}}$",
        margin=dict(t=100)) \
        .update_xaxes(visible=False).update_yaxes(visible=False)
    fig.write_image(f"ex4_plots/decision_surfaces_noise={noise}.png")

    # Question 3: Decision surface of best performing ensemble
    fig = go.Figure(
        [decision_surface(lambda X: Ada.partial_predict(X, int(lowest_err_t[0])),
                          lims[0], lims[1], showscale=False),
         go.Scatter(x=test_X[:, 0], y=test_X[:, 1], mode="markers",
                    showlegend=False,
                    marker=dict(color=test_y.astype(int),
                                symbol=symbols[np.array((test_y + 1) / 2).astype(int)],
                                colorscale=[custom[0],
                                            custom[-1]],
                                line=dict(color="black",
                                          width=1)))])
    fig.update_layout(title=f"Least Error ensemble size={lowest_err_t[0]}, accuracy={1 - lowest_err_t[1]}",
        margin=dict(t=100)) \
        .update_xaxes(visible=False).update_yaxes(visible=False)
    fig.write_image(f"ex4_plts/best_performing_noise={noise}.png")

    # Question 4: Decision surface with weighted samples
    D = (Ada.D_ * 20) / np.max(Ada.D_)
    fig = go.Figure([decision_surface(
            lambda X: Ada._predict(X),
            lims[0], lims[1], showscale=False),
            go.Scatter(x=train_X[:, 0], y=train_X[:, 1], mode='markers', showlegend=False,
                       marker=dict(color=train_y, colorscale=class_colors(2),
                                   size=D),
                       xaxis="x", yaxis="y")])
    fig.update_layout(
        title="Training sample weights",
        margin=dict(t=100)) \
        .update_xaxes(visible=False).update_yaxes(visible=False)
    fig.write_image(f"ex4_plots/weighted_samples_noise={noise}.png")


if __name__ == '__main__':
    np.random.seed(0)
    for n in [0, 0.4]:
        fit_and_evaluate_adaboost(n)
