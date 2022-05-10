from IMLearn.learners.classifiers import Perceptron, LDA, GaussianNaiveBayes, perceptron
from typing import Tuple
from utils import *
import plotly.graph_objects as go
import plotly.express as px
import os
from plotly.subplots import make_subplots
from math import atan2, pi

DATASET_DIR = 'datasets'
LEFT_COL_IDX = 1
RIGHT_COL_IDX = 2
FEATURE1_IDX = 0
FEATURE2_IDX = 1

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
        X, y = load_dataset(os.path.join(DATASET_DIR, f))

        # Fit Perceptron and record loss in each fit iteration
        losses = []
        Perceptron(callback=lambda instance, X, y: losses.append(instance.loss(X, y))).fit(X, y) 
        # For Quiz:
        # Perceptron(include_intercept=False, callback=lambda instance, X, y: losses.append(instance.loss(X, y))).fit(X, y) 
        
        # Plot figure of loss as function of fitting iteration
        go.Figure(
            data = go.Scatter(x=np.arange(start=1, stop=len(losses)), y=losses, mode='lines'),
            layout = go.Layout(
                title = f'Loss Values as a function of Training Iteration: {n}',
                xaxis_title = 'Training Iteration',
                yaxis_title = 'Loss',
            )
        ).show()

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
        X, y = load_dataset(os.path.join(DATASET_DIR, f))
        
        # Fit models and predict over training set
        gnb = GaussianNaiveBayes()
        gnb.fit(X, y)
        lda = LDA()
        lda.fit(X, y)
        gnb_predict = gnb.predict(X)
        lda_predict = lda.predict(X)

        # Plot a figure with two suplots, showing the Gaussian Naive Bayes predictions on the left and LDA predictions
        # on the right. Plot title should specify dataset used and subplot titles should specify algorithm and accuracy
        # Create subplots
        from IMLearn.metrics import accuracy
        subplots_titles = [
            f'Gaussian Naive Bayes with\n Accuracy of {accuracy(y, gnb_predict)}',
            f'Linear Discriminant Analysis with\n Accuracy of {accuracy(y, lda_predict)}',
        ]
        fig = make_subplots(rows=1, cols=2, subplot_titles=subplots_titles)
        fig.update_layout(title_text=f'Data Set: {f}') 
        df = pd.DataFrame(X, columns=['x', 'y'])
        mode = 'markers'
        # Add traces for data-points setting symbols and colors
        gnb_markers = dict(color=gnb_predict.astype(int), symbol=y)
        lda_markers = dict(color=lda_predict.astype(int), symbol=y)
        fig.add_trace(
            go.Scatter(x=df.x, y=df.y, mode=mode, marker=gnb_markers, showlegend=False),
            row=1, col=LEFT_COL_IDX,
        )
        fig.add_trace(
            go.Scatter(x=df.x, y=df.y, mode=mode, marker=lda_markers, showlegend=False),
            row=1, col=RIGHT_COL_IDX,
        )

        # Add `X` dots specifying fitted Gaussians' means
        mean_markers = dict(symbol='x', color='red')
        fig.add_trace(
            go.Scatter(x=gnb.mu_[:, FEATURE1_IDX], y=gnb.mu_[:, FEATURE2_IDX], 
                mode=mode, marker=mean_markers, showlegend=False),
            row=1, col=LEFT_COL_IDX
        )
        fig.add_trace(
            go.Scatter(x=lda.mu_[:, FEATURE1_IDX], y=lda.mu_[:, FEATURE2_IDX], 
                mode=mode, marker=mean_markers, showlegend=False),
            row=1, col=RIGHT_COL_IDX
        )
        
        # Add ellipses depicting the covariances of the fitted Gaussians
        for i in range(gnb.mu_.shape[0]):
            fig.add_trace(get_ellipse(gnb.mu_[i], np.diag(gnb.vars_[i])), row=1, col=LEFT_COL_IDX)
        for i in range(lda.mu_.shape[0]):
            fig.add_trace(get_ellipse(lda.mu_[i], lda.cov_), row=1, col=RIGHT_COL_IDX)
        fig.show()

if __name__ == '__main__':
    np.random.seed(0)
    run_perceptron()
    compare_gaussian_classifiers()
