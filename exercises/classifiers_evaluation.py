from IMLearn.learners.classifiers import Perceptron, LDA, GaussianNaiveBayes
from typing import Tuple
from utils import *
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from math import atan2, pi
import matplotlib.pyplot as plt
import matplotlib.patches as pltpach

import utils as utl
import IMLearn.metrics.loss_functions as lf



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
        X, y = load_dataset("../datasets/" + f)

        # Fit Perceptron and record loss in each fit iteration
        losses = []
        call = lambda a, b, c: losses.append(a.loss(b, [c]))

        perceptron = Perceptron(callback=call)
        perceptron.fit(X, y)

        plt.plot(range(0, len(losses)), losses)
        plt.title("percrptrone losses as function of repetitions")
        plt.xlabel("repetitions")
        plt.ylabel("losses")
        plt.show()


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
    # Plot a figure with two suplots, showing the Gaussian Naive Bayes predictions on the left and LDA predictions
    # on the right. Plot title should specify dataset used and subplot titles should specify algorithm and accuracy
    # Create subplots

    for f in ["gaussian1.npy", "gaussian2.npy"]:
        # Load dataset
        X, y = load_dataset("../datasets/" + f)
        # Fit models and predict over training set
        lda = LDA()
        lda.fit(X, y)
        prediction = lda.predict(X)
        colors = list('rgbcynkmrg')
        shapes = list('.v1spP*hXD')

        fig, axs = plt.subplots(ncols=2)
        accuracy = lf.accuracy(y, prediction)

        for sample, pred, trueVal in zip(X, prediction, y):
            axs[0].scatter((round(sample[0], 2)), (round(sample[1], 2)), c=colors[int(pred)], marker=shapes[trueVal])
        axs[0].set_title("LDA classification , accuracy:" + str(accuracy) + " data set:" + f)
        axs[0].set_xlabel('X axis')
        axs[0].set_ylabel('Y axis')

        # Add `X` dots specifying fitted Gaussians' means
        axs[0].scatter(x=np.transpose(lda.mu_)[0], y=np.transpose(lda.mu_)[1], s=50, marker='X', color='k')


        for i in lda.classes_:
            COV = np.cov(X[np.where(prediction == i)].transpose())
            eigenvalues, eigenvectors = np.linalg.eig(COV)
            theta = np.degrees(np.arctan2(*eigenvectors[:, 0][::-1]))
            width, height = 2 * np.sqrt(eigenvalues)
            ellipsis = pltpach.Ellipse(xy=lda.mu_[i], width=width, height=height, angle=theta, facecolor='none', edgecolor='k')
            axs[0].add_artist(ellipsis)

        gnb = GaussianNaiveBayes()
        gnb.fit(X, y)
        prediction = gnb.predict(X)

        accuracy = lf.accuracy(y, prediction)

        axs[1].set_title("Gaussian Naive bayes classification , accuracy:"
                         + str(accuracy) + " data set:" + f)
        axs[1].set_xlabel('X axis')
        axs[1].set_ylabel('Y axis')

        for sample, pred, trueVal in zip(X, prediction, y):
            axs[1].scatter((round(sample[0], 2)), (round(sample[1], 2)), c=colors[int(pred)], marker=shapes[trueVal])

        axs[1].scatter(x=np.transpose(gnb.mu_)[0], y=np.transpose(gnb.mu_)[1], s=50, marker='X', color='k')

        # Add traces for data-points setting symbols and colors
        f = lambda m, c: plt.plot([], [], marker=m, color=c, ls="none")[0]
        handles = [f("s", colors[i]) for i in gnb.classes_]
        handles += [f(shapes[i], "k") for i in gnb.classes_]
        labels = ["Predicted class " + str(c) for c in gnb.classes_] + ["True class " + str(c) for c in gnb.classes_]

        # circles
        for i in gnb.classes_:
            COV = np.cov(X[np.where(prediction == i)].transpose())
            eigenvalues, eigenvectors = np.linalg.eig(COV)
            theta = np.degrees(np.arctan2(*eigenvectors[:, 0][::-1]))
            width, height = 2 * np.sqrt(eigenvalues)
            ellipsis = pltpach.Ellipse(xy=gnb.mu_[i], width=width, height=height, angle=theta, facecolor='none', edgecolor='k')
            axs[1].add_artist(ellipsis)

        # Add ellipses depicting the covariances of the fitted Gaussians
        plt.legend(handles, labels)
        plt.show()
        plt.cla()
        plt.clf()
        plt.close()

if __name__ == '__main__':
    np.random.seed(0)
    run_perceptron()
    compare_gaussian_classifiers()
