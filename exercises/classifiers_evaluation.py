from IMLearn.learners.classifiers import Perceptron, LDA, GaussianNaiveBayes
from typing import Tuple
from utils import *
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from math import atan2, pi


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
        # Load dataset with `load_dataset`
        X, y = load_dataset(f)



        # Fit the Perceptron algorithm with `fit` while inside the fit loop
        # use callback function to append the loss value to the list `losses`
        losses = []
        perceptron = Perceptron(max_iter=100, callback=lambda per, item1, item2: losses.append(per._loss(X, y)))
        perceptron.fit(X, y)
        # Plot loss progression
        # create a list of x-values for plotting
        num_of_iterations = np.arange(1, len(losses) + 1)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=num_of_iterations, y=losses, mode="lines", name="Loss"))
        fig.update_layout(title=f"Perceptron: {n}", xaxis_title="Iteration", yaxis_title="Loss")
        losses = []
        fig.show()
        # Plot figure of loss as function of fitting iteration


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


def compare_gaussian_classifiers():
    """
    Fit both Gaussian Naive Bayes and LDA classifiers on both gaussians1 and gaussians2 datasets
    """
    for f in ["gaussian1.npy", "gaussian2.npy"]:
        # Load dataset
        X, y = load_dataset(f)

        # Fit models and predict over training set
        lda = LDA()
        lda.fit(X, y)

        gnb = GaussianNaiveBayes()
        gnb.fit(X, y)
        # Plot a figure with two suplots, showing the Gaussian Naive Bayes predictions on the left and LDA predictions
        # on the right. Plot title should specify dataset used and subplot titles should specify algorithm and accuracy
        # Create subplots
        # plot the following:
        # - 2D scatter plot of samples with marker color indicating Gaussian Naive Bayes predicted class
        # and marker shape indicating true class
        # - 2D scatter plot of samples with marker color indicating LDA predicted class
        # and marker shape indicating true class
        # provide classifier names and accuracy as title
        from IMLearn.metrics import accuracy
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=X[:, 0], y=X[:, 1], mode="markers", marker_color=gnb.predict(X), marker_symbol=y))
        fig.add_trace(go.Scatter(x=X[:, 0], y=X[:, 1], mode="markers", marker_color=lda.predict(X), marker_symbol=y))
        # add accuracy and names to title
        fig.update_layout(title=f"Gaussian Naive Bayes: {accuracy(gnb.predict(X), y)} LDA: {accuracy(lda.predict(X), y)}")
        fig.show()
        raise NotImplementedError()

        # Add traces for data-points setting symbols and colors
        raise NotImplementedError()

        # Add `X` dots specifying fitted Gaussians' means
        raise NotImplementedError()

        # Add ellipses depicting the covariances of the fitted Gaussians
        raise NotImplementedError()


if __name__ == '__main__':
    np.random.seed(0)
    # crun_perceptron()
    compare_gaussian_classifiers()
