from math import atan2, pi
from typing import Tuple

import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots

from IMLearn.learners.classifiers import Perceptron, LDA, GaussianNaiveBayes
from IMLearn.metrics import accuracy

pio.templates.default = "simple_white"
pio.renderers.default = "browser"


def load_dataset(filename: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load dataset for comparing the Gaussian Naive Bayes and LDA classifiers. File is assumed to be an
    ndarray of shape (n_samples, 3) where the first 2 columns represent features and the third column the class

    Parameters
    ----------
    filename: str
        Path to .npy data file

    Returns
    -------
    X: ndarray of shape (n_samples, 2)
        Design matrix to be used

    y: ndarray of shape (n_samples,)
        Class vector specifying for each sample its class

    """
    data = np.load(filename)
    return data[:, :2], data[:, 2].astype(int)


def run_perceptron():
    """
    Fit and plot fit progression of the Perceptron algorithm over both the linearly separable and inseparable datasets

    Create a line plot that shows the perceptron algorithm's training loss values (y-axis)
    as a function of the training iterations (x-axis).
    """
    for n, f in [("Linearly Separable", "linearly_separable.npy"),
                 ("Linearly Inseparable", "linearly_inseparable.npy")]:
        # Load dataset
        X, y = load_dataset(
            r"C:\Users\User\Documents\CSE_2\IML\IML.HUJI\datasets\\" + f)

        # Fit Perceptron and record loss in each fit iteration

        losses = []

        def callback(per: Perceptron, a: np.ndarray, b: int) -> None:
            losses.append(per.loss(X, y))

        per = Perceptron(callback=callback)
        per.fit(X, y)
        per.predict(X)
        # print(losses)

        # Plot figure
        iters = np.arange(0, len(losses))

        go.Figure([go.Scatter(x=iters, y=np.array(losses), mode='lines')],
                  layout=go.Layout(
                      title=f"The perceptrone loss as a function of iterations"
                            f" number for a {n} data",
                      xaxis_title="number of iterations",
                      yaxis_title="loss", height=400))
        # go.show()


def get_ellipse(mu: np.ndarray, cov: np.ndarray):
    """
    Draw an ellipse centered at given location and according to specified covariance matrix

    Parameters
    ----------
    mu : ndarray of shape (2,)
        Center of ellipse

    cov: ndarray of shape (2,2)
        Covariance of Gaussian

    Returns
    -------
        scatter: A plotly trace object of the ellipse
    """
    l1, l2 = tuple(np.linalg.eigvalsh(cov)[::-1])
    theta = atan2(l1 - cov[0, 0], cov[0, 1]) if cov[0, 1] != 0 else (
        np.pi / 2 if cov[0, 0] < cov[1, 1] else 0)
    t = np.linspace(0, 2 * pi, 100)
    xs = (l1 * np.cos(theta) * np.cos(t)) - (l2 * np.sin(theta) * np.sin(t))
    ys = (l1 * np.sin(theta) * np.cos(t)) + (l2 * np.cos(theta) * np.sin(t))

    return go.Scatter(x=mu[0] + xs, y=mu[1] + ys, mode="lines",
                      marker_color="black")


def compare_gaussian_classifiers():
    """
    Fit both Gaussian Naive Bayes and LDA classifiers on both gaussians1 and gaussians2 datasets
    """
    lda = LDA()
    naive_bayes = GaussianNaiveBayes()
    for f in ["gaussian1.npy", "gaussian2.npy"]:
        # Load dataset
        X, y = load_dataset(
            r"C:\Users\User\Documents\CSE_2\IML\IML.HUJI\datasets\\" + f)

        # Fit models and predict over training set
        lda.fit(X, y)
        naive_bayes.fit(X, y)

        # Plot a figure with two suplots, showing the Gaussian Naive Bayes predictions on the left and LDA predictions
        # on the right. Plot title should specify dataset used and subplot titles should specify algorithm and accuracy
        symbols = np.array(["triangle-up", "star", "octagon"])

        fig = make_subplots(rows=1, cols=2,
                            subplot_titles=(r"$\textbf{LDA}$",
                                            r"$\textbf{Gaussian Naive Bayes}$"),
                            horizontal_spacing=0.1, vertical_spacing=.1,
                            specs=[
                                [{"type": "scatter"}, {"type": "scatter"}]], )

        # add LDA sub-plot:
        fig.add_trace(
            go.Scatter(x=X[:, 0], y=X[:, 1], mode='markers', showlegend=False,
                       marker=dict(color=lda.predict(X), symbol=symbols[y],
                                   line=dict(color=lda.predict(X),
                                             width=3)), ), 1, 1)

        # add Gaussian Naive Bayes sub-plot:
        fig.add_trace(
            go.Scatter(x=X[:, 0], y=X[:, 1], mode='markers', showlegend=False,
                       marker=dict(color=naive_bayes.predict(X),
                                   symbol=symbols[y],
                                   line=dict(color=naive_bayes.predict(X),
                                             width=3))), 1, 2)
        fig.update_layout(
            title=rf"$\textbf{{Decisions Of The Models on {f} Dataset}}$",
            margin=dict(t=100))

        # add ellipsis to plot:

        fig.add_trace(get_ellipse(lda.mu_[0], lda.cov_), 1, 1)
        fig.add_trace(get_ellipse(lda.mu_[1], lda.cov_), 1, 1)
        fig.add_trace(get_ellipse(lda.mu_[2], lda.cov_), 1, 1)

        fig.add_trace(
            get_ellipse(naive_bayes.mu_[0], np.diag(naive_bayes.vars_[0])), 1,
            2)
        fig.add_trace(
            get_ellipse(naive_bayes.mu_[1], np.diag(naive_bayes.vars_[1])), 1,
            2)
        fig.add_trace(
            get_ellipse(naive_bayes.mu_[2], np.diag(naive_bayes.vars_[2])), 1,
            2)

        # add accuracy:

        a = accuracy(lda.predict(X), y)
        b = accuracy(naive_bayes.predict(X), y)
        fig.layout.annotations[0].update(
            text=f"LDA accuracy: {accuracy(lda.predict(X), y)}")
        fig.layout.annotations[1].update(
            text=f"Gaussian Naive Bayes accuracy: {accuracy(naive_bayes.predict(X), y)}")

        # add center:
        fig.add_trace(
            go.Scatter(x=np.array(lda.mu_[:, 0]), y=np.array(lda.mu_[:, 1]),
                       mode="markers", marker_symbol="x", marker_color="black",
                       marker_size=20), 1, 2)
        fig.add_trace(
            go.Scatter(x=np.array(naive_bayes.mu_[:, 0]),
                       y=np.array(naive_bayes.mu_[:, 1]),
                       mode="markers", marker_symbol="x", marker_color="black",
                       marker_size=20), 1, 1)

        # fig.show()



if __name__ == '__main__':
    np.random.seed(0)
    run_perceptron()
    compare_gaussian_classifiers()

