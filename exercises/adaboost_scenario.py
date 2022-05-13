import numpy as np
from typing import Tuple
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


def fit_and_evaluate_adaboost(noise, n_learners=250, train_size=5000, test_size=500):
    (train_X, train_y), (test_X, test_y) = generate_data(train_size, noise), generate_data(test_size, noise)

    # Question 1: Train- and test errors of AdaBoost in noiseless case
    ada_boost=AdaBoost(DecisionStump, n_learners)
    ada_boost.fit(train_X, train_y)

    losses_train=np.zeros(n_learners)
    losses_test = np.zeros(n_learners)
    for t in range(n_learners):
        losses_train[t]=ada_boost.partial_loss(train_X,train_y,t)
        losses_test[t] = ada_boost.partial_loss(test_X, test_y, t)


    arr_iterations=np.arange(n_learners)
    title=f"The training- and test errors as a function of the number of fitted learners, noise={noise}"
    fig1 = go.Figure(layout=go.Layout(title=title, margin=dict(t=100)))
    fig1.add_trace(go.Scatter(x=arr_iterations[1:], y=losses_train[1:], mode='lines', name="Train"))
    fig1.add_trace(go.Scatter(x=arr_iterations[1:], y=losses_test[1:], mode='lines', name="Test"))
    fig1.show()

    lims = np.array([np.r_[train_X, test_X].min(axis=0), np.r_[train_X, test_X].max(axis=0)]).T + np.array([-.1, .1])
    # Question 2: Plotting decision surfaces
    if noise==0:
        T = [5, 50, 100, 250]
        fig2 = make_subplots(rows=2, cols=2, subplot_titles=[f"{t} iterations" for t in T],
                            horizontal_spacing=0.01, vertical_spacing=.03)
        for i,t in enumerate(T):
            fig2.add_traces([decision_surface(lambda x:ada_boost.partial_predict(x,t), lims[0], lims[1], showscale=False),
                            go.Scatter(x=test_X[:,0], y=test_X[:,1], mode="markers", showlegend=False,
                                       marker=dict(color=test_y, symbol="circle", colorscale=[custom[0], custom[-1]],
                                                   line=dict(color="black", width=1)))],
                           rows=(i//2) + 1, cols=(i%2)+1)

        fig2.update_layout(title="The decision boundary obtained by using the the ensemble up to iteration 5, 50, 100 and 250, noise=0", margin=dict(t=100)) \
            .update_xaxes(visible=False).update_yaxes(visible=False)
        fig2.show()

    # Question 3: Decision surface of best performing ensemble
    if noise==0:
        min_t =  int(np.argmin(losses_test))
        fig3 = make_subplots(rows=1, cols=1, subplot_titles=[f"Plot of the decision surface of the ensemble size achieved the lowest test error is: {min_t}, noise=0 "],
                             horizontal_spacing=0.01, vertical_spacing=.03)
        fig3.add_traces([decision_surface(lambda x: ada_boost.partial_predict(x, min_t), lims[0], lims[1], showscale=False),
             go.Scatter(x=test_X[:, 0], y=test_X[:, 1], mode="markers", showlegend=False,
                        marker=dict(color=test_y, symbol="circle", colorscale=[custom[0], custom[-1]],
                                    line=dict(color="black", width=1)))], rows=1, cols=1)
        fig3.update_layout(margin=dict(t=100)).update_xaxes(visible=False).update_yaxes(visible=False)
        fig3.show()


    # Question 4: Decision surface with weighted samples

    D =ada_boost.D_[n_learners-1]/ np.max(ada_boost.D_[n_learners-1]) * 5
    fig4 = make_subplots(rows=1, cols=1, subplot_titles=[
        f"Plot of the training set with a point size proportional to it’s weight, noise={noise}"],
                         horizontal_spacing=0.01, vertical_spacing=.03)
    fig4.add_traces([decision_surface(lambda x: ada_boost.partial_predict(x, n_learners), lims[0], lims[1], showscale=False),
                     go.Scatter(x=train_X[:, 0], y=train_X[:, 1], mode="markers", showlegend=False,
                                marker=dict(color=test_y, symbol="circle",size=D, colorscale=[custom[0], custom[-1]],
                                            line=dict(color="black", width=1)))], rows=1, cols=1)

    fig4.update_layout(margin=dict(t=100)) \
        .update_xaxes(visible=False).update_yaxes(visible=False)
    fig4.show()


if __name__ == '__main__':
    np.random.seed(0)
    fit_and_evaluate_adaboost(0)
    fit_and_evaluate_adaboost(0.4)
