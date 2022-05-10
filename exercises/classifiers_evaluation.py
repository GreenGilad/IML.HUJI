from math import atan2

from IMLearn.learners.classifiers import Perceptron, LDA, GaussianNaiveBayes
import numpy as np
from typing import Tuple
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots
pio.templates.default = "simple_white"


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
    for n, f in [("Linearly Separable", "linearly_separable.npy"), ("Linearly Inseparable", "linearly_inseparable.npy")]:
        # Load dataset
        X, y = load_dataset("..\\datasets\\" + f)

        # Fit Perceptron and record loss in each fit iteration
        def callback(perc: Perceptron, x1: np.ndarray, y1: int):
            losses.append(perc.loss(X, y))

        losses = []
        perceptron = Perceptron(callback=callback)
        perceptron.fit(X, y)

        # Plot figure
        iters = np.arange(0, len(losses))

        go.Figure([go.Scatter(x=iters, y=np.array(losses), mode='lines')],
                  layout=go.Layout(title="Loss of Perceptron as function of iterations for database " + n,
                                   xaxis_title="number of iterations",
                                   yaxis_title="loss", height=500)).show()


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
    theta = atan2(l1 - cov[0, 0], cov[0, 1]) if cov[0, 1] != 0 else (np.pi / 2 if cov[0, 0] < cov[1, 1] else 0)
    t = np.linspace(0, 2 * np.pi, 100)
    xs = (l1 * np.cos(theta) * np.cos(t)) - (l2 * np.sin(theta) * np.sin(t))
    ys = (l1 * np.sin(theta) * np.cos(t)) + (l2 * np.cos(theta) * np.sin(t))

    return go.Scatter(x=mu[0] + xs, y=mu[1] + ys, mode="lines", marker_color="black")


def compare_gaussian_classifiers():
    """
    Fit both Gaussian Naive Bayes and LDA classifiers on both gaussians1 and gaussians2 datasets
    """
    for n, f in [("gaussian1", "gaussian1.npy"), ("gaussian2", "gaussian2.npy")]:
        # Load dataset
        X, y = load_dataset("..\\datasets\\" + f)

        # Fit models and predict over training set
        lda = LDA()
        lda.fit(X, y)
        lad_predict = lda.predict(X)

        gaussian_naive_bayes = GaussianNaiveBayes()
        gaussian_naive_bayes.fit(X, y)
        gaussian_predict = gaussian_naive_bayes.predict(X)

        # Plot a figure with two suplots, showing the Gaussian Naive Bayes predictions on the left and LDA predictions
        # on the right. Plot title should specify dataset used and subplot titles should specify algorithm and accuracy
        from IMLearn.metrics import accuracy
        models_and_accuracy = [("LDA", accuracy(y, lad_predict)),
                               ("Gaussian Naive Bayes", accuracy(y, gaussian_predict))]

        fig = make_subplots(rows=1, cols=2,
                            subplot_titles=[rf"$\textbf{{The model- {m},The accuracy- {a}}}$"
                                            for m, a in models_and_accuracy],
                            horizontal_spacing=0.01, vertical_spacing=0.03)

        # show LDA prediction
        fig.add_trace(
            go.Scatter(x=X[:, 0], y=X[:, 1], mode="markers",
                       marker=go.scatter.Marker(color=gaussian_predict, symbol=y)), row=1, col=1)

        # show Gaussian prediction
        fig.add_trace(
            go.Scatter(x=X[:, 0], y=X[:, 1], mode="markers",
                       marker=go.scatter.Marker(color=gaussian_predict, symbol=y)), row=1, col=2)

        fig.update_layout(
            title=rf"$\textbf{{The dataset is {n}}}$", showlegend=False, margin=dict(t=100))

        fig.add_trace(go.Scatter(x=gaussian_naive_bayes.mu_[:, 0],
                                 y=gaussian_naive_bayes.mu_[:, 1], mode='markers',
                                 marker={'symbol': 'x', 'color': 'black', 'size': 10}), row=1, col=2)

        fig.add_trace(go.Scatter(x=lda.mu_[:, 0],
                                 y=lda.mu_[:, 1], mode='markers',
                                 marker={'symbol': 'x', 'color': 'black', 'size': 10}), row=1, col=1)

        for i in range(gaussian_naive_bayes.classes_.shape[0]):
            fig.add_trace(trace=get_ellipse(gaussian_naive_bayes.mu_[i], np.diag(gaussian_naive_bayes.vars_[i])),
                          row=1, col=2)
            fig.add_trace(trace=get_ellipse(lda.mu_[i], lda.cov_), row=1, col=1)
        fig.show()


if __name__ == '__main__':
    np.random.seed(0)
    run_perceptron()
    compare_gaussian_classifiers()


    # quiz
    X1 = np.array([[0], [1], [2], [3], [4], [5], [6], [7]])
    y1 = np.array([0, 0, 1, 1, 1, 1, 2, 2])
    model1 = GaussianNaiveBayes()
    model1.fit(X1, y1)
    print("quiz question 2")
    print("pi[0]:", model1.pi_[0])
    print("mu[1]:", model1.mu_[1])

    X2 = np.array([[1, 1], [1, 2], [2, 3], [2, 4], [3, 3], [3, 4]])
    y2 = np.array([0, 0, 1, 1, 1, 1])
    model2 = GaussianNaiveBayes()
    model2.fit(X2, y2)
    print("\nquiz question 2")
    print("var[0, 0]:", model2.vars_[0][0])
    print("var[0, 1]:", model2.vars_[1][0])


