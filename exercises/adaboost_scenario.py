from typing import Tuple

from IMLearn.learners.classifiers import DecisionStump
from IMLearn.metalearners.adaboost import AdaBoost
from utils import *

pio.templates.default = "simple_white"
pio.renderers.default = "browser"


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


def fit_and_evaluate_adaboost(noise, n_learners=250, train_size=5000,
                              test_size=500):
    (train_X, train_y), (test_X, test_y) = generate_data(train_size,
                                                         noise), generate_data(
        test_size, noise)

    # Question 1: Train- and test errors of AdaBoost in noiseless case
    ada = AdaBoost(DecisionStump, 250)
    ada.fit(train_X, train_y)

    train_loss = []
    test_loss = []
    for t in range(1, n_learners + 1):
        train_loss.append(ada.partial_loss(train_X, train_y, t))
        test_loss.append(ada.partial_loss(test_X, test_y, t))
    x = list(range(1, n_learners + 1))
    fig = go.Figure(
        [go.Scatter(x=x, y=test_loss, mode='lines+markers',
                    name=r'Test Error'),
         go.Scatter(x=x, y=train_loss, mode='lines+markers',
                    name=r'Train Error')],
        layout=go.Layout(
            title=fr"$(1) text{{Training and Test Errors As Function Of Number Of Fitted Learners With Noise= {noise}}}$",
            xaxis_title="$\\text{number of fitted learners}$", font = dict(size= 24),
            yaxis_title="error value",
            height=600))

    fig.show()

    # # Question 2: Plotting decision surfaces
    T = [5, 50, 100, 250]
    lims = np.array([np.r_[train_X, test_X].min(axis=0),
                     np.r_[train_X, test_X].max(axis=0)]).T + np.array(
        [-.1, .1])

    symbols = np.array(["square", "x", "circle"])


    # for j in range(2):
    fig = make_subplots(rows=2, cols=2,
                        subplot_titles=[rf"$\textbf{{{m} learners}}$" for m
                                        in T],
                        horizontal_spacing=0.01, vertical_spacing=.05)
    for t, iter in enumerate(T):
        def partial_p(data: np.ndarray):
            return ada.partial_predict(data, iter)

        row = (t // 2) + 1
        col = (t % 2) + 1

        fig.add_traces(
            [decision_surface(partial_p, lims[0], lims[1],
                              showscale=True),
             go.Scatter(x=test_X[:, 0], y=test_X[:, 1], mode="markers",
                        showlegend=False,
                        marker=dict(color=test_y,
                                    symbol=symbols[test_y.astype(int)],
                                    # colorscale=[custom[0], custom[-1]],
                                    size=10,
                                    line=dict(color="black", width=0.2)))],
            rows=row, cols=col)
    fig.update_layout(
        title=rf"$\textbf{{(2) AdaBoost Decision Boundaries With Noise= {noise}}}$",
        yaxis1_range=[-1, 1], yaxis2_range=[-1, 1], yaxis3_range=[-1, 1],
        yaxis4_range=[-1, 1],
        xaxis1_range=[-1, 1], xaxis2_range=[-1, 1], xaxis3_range=[-1, 1],
        xaxis4_range=[-1, 1])

    fig.show()

    # Question 3: Decision surface of best performing ensemble
    # find best loss:
    arr = np.array(test_loss)
    min_it = np.argmin(np.array(test_loss)) + 1

    # calculate accuracy:
    accuracy = 1 - test_loss[min_it - 1]

    def partial_(data: np.ndarray):
        return ada.partial_predict(data, min_it)

    go.Figure([decision_surface(partial_, lims[0], lims[1]),
               go.Scatter(x=test_X[:, 0], y=test_X[:, 1], mode="markers",
                          showlegend=False,
                          marker=dict(color=test_y,
                                      symbol=symbols[test_y.astype(int)],
                                      size=8,
                                      line=dict(color="black",
                                                width=0.2)))]).update_layout(
        title=rf"$\textbf{{(3) AdaBoost Decision Boundaries of {min_it} Learners On Test Dataset With Noise= {noise}. accuracy = {accuracy}}}$",
        yaxis_range=[-1, 1], xaxis_range=[-1, 1]).show()


    # Question 4: Decision surface with weighted samples
    colors = np.array(["greenyellow", "aquamarine"])
    # y_pred = ada.predict(test_X).astype(int)
    D = ada.D_ / np.max(ada.D_) * 15
    D = np.array(D)
    train_y = np.where(train_y > 0, 1, 0).astype(int)
    go.Figure(
        [decision_surface(ada.predict, lims[0], lims[1], showscale=False),
         go.Scatter(x=train_X[:, 0], y=train_X[:, 1], mode="markers",
                    showlegend=False,
                    marker=dict(color=colors[train_y],
                                symbol=symbols[train_y],
                                # colorscale=colors[train_y],
                                size=D,
                                line=dict(color=colors[train_y],
                                          width=0.2)))]).update_layout(
        title=rf"$\textbf{{(4) AdaBoost: Samples Proportional to Weights With Noise= {noise}}}$",
        font=dict(size=24),
        yaxis_range=[-1, 1], xaxis_range=[-1, 1]).show()




if __name__ == '__main__':
    np.random.seed(0)
    fit_and_evaluate_adaboost(0)
    # fit_and_evaluate_adaboost(noise=0.4)
