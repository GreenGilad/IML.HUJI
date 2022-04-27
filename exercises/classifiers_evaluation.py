import numpy as np
import plotly.graph_objects

from IMLearn.learners.classifiers import Perceptron, LDA, GaussianNaiveBayes
from typing import Tuple
from utils import *
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from math import atan2, pi

from IMLearn.metrics.loss_functions import accuracy


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
        X, y = load_dataset(f"../datasets/{f}")

        # Fit Perceptron and record loss in each fit iteration
        losses = []

        # First we will define a callback function that returns the loss for a given iteration
        def callback(fit: Perceptron, x: np.ndarray, value: int):
            losses.append(fit.loss(X, y))

        perceptron_fit = Perceptron(callback=callback)
        perceptron_fit.fit(X, y)

        # Plot figure of loss as function of fitting iteration
        fig = px.line(losses,
                      title=f"Loss as function of iterations for {n} data",
                      x=[i for i in range(len(losses))],
                      y=losses,
                      labels={
                          'x': 'Iterations',
                          'y': 'Loss'
                      })
        fig.show()


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


def plot_classifier_predictions(X: np.ndarray, predictions: np.ndarray, true_values: np.ndarray):
    """
    Plots the scatter of predictions of datasets for given classifier
    """
    return go.Scatter(
        x=X[:, 0],
        y=X[:, 1],
        mode='markers',
        marker=dict(
            size=20,
            color=predictions,
            symbol=true_values,
            line_width=1
        ))


def adding_mean_class_value_crosses(mu_values: np.ndarray):
    """
    Adds x markers for mean values of each class to scatter plot
    """
    return go.Scatter(
        x=mu_values[:, 0],
        y=mu_values[:, 1],
        mode='markers',
        marker=dict(
            size=30,
            color='black',
            symbol='x'
        ))

def compare_gaussian_classifiers():
    """
    Fit both Gaussian Naive Bayes and LDA classifiers on both gaussians1 and gaussians2 datasets
    """

    for f in ["gaussian1.npy", "gaussian2.npy"]:
        # Load dataset
        X, y = load_dataset(f"../datasets/{f}")

        dataset_name = f[:-4]

        # Fit models and predict over training set

        # Fitting the Naive Gaussian bayes model
        naive_gauss = GaussianNaiveBayes()
        naive_gauss.fit(X,y)

        # Fitting the LDA model
        lda = LDA()
        lda.fit(X, y)

        # Predicting classes with Naive Gaussian
        naive_gauss_predictions = naive_gauss.predict(X)
        naive_gauss_accuracy = round(accuracy(y, naive_gauss_predictions), 4)

        # Predicting classes with LDA
        lda_predictions = lda.predict(X)
        lda_accuracy = round(accuracy(y, lda_predictions), 4)

        # Plot a figure with two suplots, showing the Gaussian Naive Bayes predictions on the left and LDA predictions
        # on the right. Plot title should specify dataset used and subplot titles should specify algorithm and accuracy
        # Create subplots
        fig = make_subplots(
            rows=1, cols=2, subplot_titles=[f"Naive Gaussian model with accuracy {naive_gauss_accuracy}"
                , f"LDA model with accuracy {lda_accuracy}"])

        # Add traces for data-points setting symbols and colors

        # Adding the LDA trace
        fig.add_trace(
            plot_classifier_predictions(X, lda_predictions, y),
            row=1, col=2
        )

        # Adding the naive Gaussian trace
        fig.add_trace(
            plot_classifier_predictions(X, naive_gauss_predictions, y),
            row=1, col=1
        )

        # Add `X` dots specifying fitted Gaussians' means

        # Adding 'X' for lda classifier
        fig.add_trace(
            adding_mean_class_value_crosses(lda.mu_),
            row=1, col=2
        )

        # Adding 'X' for naive gauss classifier
        fig.add_trace(
            adding_mean_class_value_crosses(naive_gauss.mu_),
            row=1, col=1
        )

        # Add ellipses depicting the covariances of the fitted Gaussians

        # Adding ellipses for LDA
        for c in lda.classes_:
            fig.add_trace(
                get_ellipse(lda.mu_[c], lda.cov_),
                row=1, col=2
            )

        # Adding ellipses for Naive Gaussian
        for c in naive_gauss.classes_:
            fig.add_trace(
                get_ellipse(naive_gauss.mu_[c], np.diag(naive_gauss.vars_[c])),
                row=1, col=1
            )

        fig.update_layout(
            title={
                'text': f"Predicted classifications of {dataset_name} dataset using LDA and Naive Bayes classifiers",
                'x': 0.5,
                'y': 0.99,
                'font': {'size': 20, 'color': 'blue'}
            },
            showlegend=False)

        fig.show()


if __name__ == '__main__':
    np.random.seed(0)
    run_perceptron()
    compare_gaussian_classifiers()
