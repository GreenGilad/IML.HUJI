from IMLearn.learners.classifiers import Perceptron, LDA, GaussianNaiveBayes
import numpy as np
from typing import Tuple
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots
from IMLearn.metrics.loss_functions import accuracy
from math import atan2, pi

pio.templates.default = "simple_white"


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
    t = np.linspace(0, 2 * pi, 100)
    xs = (l1 * np.cos(theta) * np.cos(t)) - (l2 * np.sin(theta) * np.sin(t))
    ys = (l1 * np.sin(theta) * np.cos(t)) + (l2 * np.cos(theta) * np.sin(t))

    return go.Scatter(x=mu[0] + xs, y=mu[1] + ys, mode="lines", marker_color="black")


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
    arr = np.load(filename)
    return (arr[:, 0:-1], arr[:, -1])


def run_perceptron():
    """
    Fit and plot fit progression of the Perceptron algorithm over both the linearly separable and inseparable datasets

    Create a line plot that shows the perceptron algorithm's training loss values (y-axis)
    as a function of the training iterations (x-axis).
    """
    res = make_subplots(rows=2, cols=1, subplot_titles=("Linearly Separable", "Linearly Inseparable"))
    counter = 1
    for n, f in [("Linearly Separable", "linearly_separable.npy"),
                 ("Linearly Inseparable", "linearly_inseparable.npy")]:
        # Load dataset
        X, y = load_dataset(r'C:\Users\Lenovo\Documents\GitHub\IML.HUJI\datasets//' + f)
        # Fit Perceptron and record loss in each fit iteration

        losses = []
        callfunc = lambda fit, x, i: losses.append(fit.loss(X, y))
        my_perc = Perceptron(callback=callfunc,include_intercept=False).fit(X, y)
        # Plot figure
        res.add_trace(go.Scatter(x=list(range(1, len(losses) + 1)), y=losses), row=counter, col=1).update_layout(
            title="Loss as a function of percpectron iteration", xaxis_title="Percpectron iteration",
            yaxis_title="Loss")
        counter = counter + 1
    res.write_html('percpectron.html')


def compare_gaussian_classifiers():
    """
    Fit both Gaussian Naive Bayes and LDA classifiers on both gaussians1 and gaussians2 datasets
    """
    for f in ["gaussian1.npy", "gaussian2.npy"]:
        # Load dataset
        X, y = load_dataset(r'C:\Users\Lenovo\Documents\GitHub\IML.HUJI\datasets//' + f)
        # Fit models and predict over training set
        my_lda = LDA().fit(X, y)
        lda_y_hat = my_lda.predict(X)
        lda_acc = accuracy(y, lda_y_hat)

        my_gdb = GaussianNaiveBayes().fit(X, y)
        gdb_y_hat = my_gdb.predict(X)
        gdb_acc = accuracy(y, gdb_y_hat)

        # Plot a figure with two suplots, showing the Gaussian Naive Bayes predictions on the left and LDA predictions
        # on the right. Plot title should specify dataset used and subplot titles should specify algorithm and accuracy
        fig = make_subplots(rows=1, cols=2, subplot_titles=(
            f[:-4] + ', GaussianNaiveBayes, accuracy: ' + str(gdb_acc), f[:-4] + ', LDA, accuracy: ' + str(lda_acc)))
        # Add traces for data-points setting symbols and colors
        fig.add_trace(go.Scatter(x=X[:, 0], y=X[:, 1], mode="markers",
                                 marker=dict(color=gdb_y_hat, symbol=y, line=dict(color="black", width=0.5))),
                      row=1, col=1)
        fig.add_trace(go.Scatter(x=X[:, 0], y=X[:, 1], mode="markers",
                                 marker=dict(color=lda_y_hat, symbol=y, line=dict(color="black", width=0.5))),
                      row=1, col=2)
        # Add `X` dots specifying fitted Gaussians' means
        fig.add_trace(go.Scatter(mode='markers', x=my_gdb.mu_[:, 0], y=my_gdb.mu_[:, 1],
                                 marker=dict(color='black', size=12, symbol='x')), row=1, col=1)
        fig.add_trace(go.Scatter(mode='markers', x=my_lda.mu_[:, 0], y=my_lda.mu_[:, 1],
                                 marker=dict(color='black', size=12, symbol='x')), row=1, col=2)
        # add elipsis
        for i in range(len(my_lda.classes_)):
            gdb_index = np.where(y == my_lda.classes_[i])
            gdb_covv = np.cov(X[gdb_index].transpose())

            fig.add_trace(get_ellipse(my_gdb.mu_[i], np.diag(my_gdb.vars_[i])), row=1, col=1)
            fig.add_trace(get_ellipse(my_lda.mu_[i], my_lda.cov_), row=1, col=2)
        fig.write_html(f + '_plot.html')


if __name__ == '__main__':
    np.random.seed(0)
    run_perceptron()
    compare_gaussian_classifiers()
