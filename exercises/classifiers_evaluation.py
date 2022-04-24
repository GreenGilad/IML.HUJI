import plotly.express

from IMLearn.learners.classifiers import Perceptron, LDA, GaussianNaiveBayes
import numpy as np
from typing import Tuple
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots
from IMLearn.metrics import loss_functions

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
    return (data[:, 0:-1], data[:, -1])


def run_perceptron():
    """
    Fit and plot fit progression of the Perceptron algorithm over both the linearly separable and inseparable datasets

    Create a line plot that shows the perceptron algorithm's training loss values (y-axis)
    as a function of the training iterations (x-axis).
    """
    for n, f in [("Linearly Separable", "linearly_separable.npy"),
                 ("Linearly Inseparable", "linearly_inseparable.npy")]:
        # Load dataset
        X, y = load_dataset(r"C:\Users\elish\OneDrive\Documents\GitHub\IML\IML.HUJI\datasets//" + f)
        # Fit Perceptron and record loss in each fit iteration
        losses = []
        collect_losses = lambda fit, x, y: losses.append(fit.loss(x, y))
        model = Perceptron(callback=collect_losses)
        fitted_model = model.fit(X, y)
        # Plot figure
        # fig = plotly.express.line(x=np.array(range(len(losses))), y=losses, title='test')
        fig = go.Figure([go.Scatter(x=np.array(range(len(losses))), y=losses)], layout=go.Layout(
            title=n + r" data, loss as function of perceptron iterations",
            height=500))
        fig.write_html(n + '.html', auto_open=True)


def compare_gaussian_classifiers():
    """
    Fit both Gaussian Naive Bayes and LDA classifiers on both gaussians1 and gaussians2 datasets
    """
    for f in ["gaussian1.npy", "gaussian2.npy"]:
        # Load dataset
        X, y = load_dataset(r"C:\Users\elish\OneDrive\Documents\GitHub\IML\IML.HUJI\datasets//" + f)
        # Fit models and predict over training set
        lda_model = LDA().fit(X, y)
        y_hat_lda = lda_model.predict(X)
        naive_model = GaussianNaiveBayes().fit(X, y)
        y_hat_naive = naive_model.predict(X)
        # Plot a figure with two suplots, showing the Gaussian Naive Bayes predictions on the left and LDA predictions
        fig = make_subplots(rows=1, cols=2, subplot_titles=[
            'dataset: ' + f + ', algorithm: Gaussian Naive Bayes, accuracy: ' + str(
                np.round(loss_functions.accuracy(y, y_hat_naive), 3)),
            'dataset: ' + f + ', algorithm: LDA, accuracy: ' + str(np.round(loss_functions.accuracy(y, y_hat_lda), 3))])
        fig.add_trace(go.Scatter(x=X.T[0], y=X.T[1], mode="markers", marker=dict(color=y_hat_naive, symbol=y,
                                                                                 line=dict(color="black", width=1))),
                      row=1, col=1)
        fig.add_trace(go.Scatter(x=X.T[0], y=X.T[1], mode="markers", marker=dict(color=y_hat_lda, symbol=y,
                                                                                 line=dict(color="black", width=1))),
                      row=1, col=2)
        fig.add_trace(go.Scatter(x=naive_model.mu_.T[0], y=naive_model.mu_.T[1], mode="markers",
                                 marker=dict(color='black', symbol='x', size= 20,
                                             line=dict(color="black", width=1))),
                      row=1, col=1)
        fig.add_trace(go.Scatter(x=lda_model.mu_.T[0], y=lda_model.mu_.T[1], mode="markers",
                                 marker=dict(color='black', symbol='x', size=20,
                                             line=dict(color="black", width=1))),
                      row=1, col=2)
        for mu in naive_model.mu_:
            fig.update_layout(shapes= [dict(type= "circle", x0=5, y0=12, x1=10, y1=18)])
        fig.write_html('test.html', auto_open=True)
        # on the right. Plot title should specify dataset used and subplot titles should specify algorithm and accuracy
        from IMLearn.metrics import accuracy


if __name__ == '__main__':
    np.random.seed(0)
    # run_perceptron()
    compare_gaussian_classifiers()
