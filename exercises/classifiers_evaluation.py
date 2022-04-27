from IMLearn.learners.classifiers import Perceptron, LDA, GaussianNaiveBayes
from typing import Tuple
from utils import *
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from math import atan2, pi
from IMLearn.metrics import accuracy, misclassification_error


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
        perceptron = Perceptron(max_iter=1000, callback=lambda per, item1, item2: losses.append(per._loss(X, y)))
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
        # compare predictions by printing all the samples
        gnb = GaussianNaiveBayes()
        gnb.fit(X, y)

        # run the decision boundary plot
        # decision_boundaries_of_models(X,gnb,lda,y)
        # Plot a figure with two suplots, showing the Gaussian Naive Bayes predictions on the left and LDA predictions
        # on the right. Plot title should specify dataset used and subplot titles should specify algorithm and accuracy
        # Create subplots
        # decision_boundaries_of_models(X, gnb, lda, y)
        fig = make_subplots(rows=1, cols=2, specs=[[{"type": "scatter"}, {"type": "scatter"}]], subplot_titles=("Gaussian Naive Bayes", "LDA"))
        fig.update_layout(showlegend=False)
        # Add scatter traces
        # add the predicted Gaussian Naive Bayes classifications with different colors
        fig.add_trace(go.Scatter(x=X[:, 0], y=X[:, 1], mode="markers",marker_symbol=class_symbols[y],
                                 marker_color=gnb.predict(X), name="Gaussian Naive Bayes", marker_size=12,), 1, 1)
        # add the predicted LDA classifications with different colors
        fig.add_trace(go.Scatter(x=X[:, 0], y=X[:, 1], mode="markers",
                                 marker_symbol=class_symbols[y] ,marker_color=lda.predict(X), name="LDA", marker_size=12), 1, 2)
        # Add ellipses

        fig.add_trace(get_ellipse(gnb.mu_[0], np.diag(gnb.vars_[0])), 1, 1)
        fig.add_trace(get_ellipse(gnb.mu_[1], np.diag(gnb.vars_[1])), 1, 1)
        fig.add_trace(get_ellipse(gnb.mu_[2], np.diag(gnb.vars_[2])), 1, 1)

        fig.add_trace(get_ellipse(lda.mu_[0], lda.cov_), 1, 2)
        fig.add_trace(get_ellipse(lda.mu_[1], lda.cov_), 1, 2)
        fig.add_trace(get_ellipse(lda.mu_[2], lda.cov_), 1, 2)

        # Add titles
        fig.update_layout(title=f"Gaussian Naive Bayes and LDA on {f}", xaxis_title="x", yaxis_title="y", title_x=0.5)
        # add to each subplot title a name of the model and accuracy
        # add accuracy to the subplot title using accuracy function and layout.annotations
        fig.layout.annotations[0].update(text=f"Gaussian Naive Bayes accuracy: {accuracy(gnb.predict(X), y)}")
        fig.layout.annotations[1].update(text=f"LDA accuracy: {accuracy(lda.predict(X), y)}")
        # Markers (color black and shaped 'X') indicating the center of each class
        # add to the graph 'X' based on the mean of each class
        # add a single dot to the graph for each class center shape 'X'
        fig.add_trace(go.Scatter(x=np.array(gnb.mu_[:, 0]), y=np.array(gnb.mu_[:, 1]),
                                 mode="markers", marker_symbol="x", marker_color="black", marker_size=23), 1, 1)
        fig.add_trace(go.Scatter(x=np.array(lda.mu_[:,0]),  y=np.array(lda.mu_[:,1]),
                                 mode="markers", marker_symbol="x",marker_color="black", marker_size=23), 1, 2)

        fig.show()
        # Add ellipses depicting the covariances of the fitted Gaussians


def decision_boundaries_of_models(X, gnb, lda, y):
    models = [gnb, lda]
    model_names = ["Gaussian Naive Bayes", "LDA"]
    lims = np.array([X.min(axis=0), X.max(axis=0)]).T + np.array([-.4, .4])
    fig = make_subplots(rows=1, cols=2, subplot_titles=[rf"$\textbf{{{m}}}$" for m in model_names],
                        horizontal_spacing=0.01, vertical_spacing=.03)
    for i, m in enumerate(models):
        fig.add_traces([decision_surface(m.fit(X, y).predict, lims[0], lims[1], showscale=False),
                        go.Scatter(x=X[:, 0], y=X[:, 1], mode="markers", showlegend=False,
                                   marker=dict(color=y, colorscale=class_colors(3),
                                               line=dict(color="black", width=1)))],
                       rows=1, cols=(i % 2) + 1)
    fig.update_layout(title=rf"$\textbf{{(2) Decision Boundaries Of Models - {model_names[0]} VS {model_names[1]}}}$",
                      margin=dict(t=100)) \
        .update_xaxes(visible=False).update_yaxes(visible=False)
    # calculate the number of missclassifications for each model
    fig.layout.annotations[0].update(text=f"Gaussian Naive Bayes misclassification: {misclassification_error(gnb.predict(X), y, False)}")
    fig.layout.annotations[1].update(text=f"LDA misclassification: {misclassification_error(lda.predict(X), y, False)}")
    fig.show()
    exit(1)

def quiz_1():
    # X∈R2, Y∈{0, 1}
    # S = {([1, 1], 0), ([1, 2], 0), ([2, 3], 1), ([2, 4], 1), ([3, 3], 1), ([3, 4], 1)}
    X = np.array([[1, 1], [1, 2], [2, 3], [2, 4], [3, 3], [3, 4]])
    y = np.array([0, 0, 1, 1, 1, 1])
    # fit naive bayes
    gnb = GaussianNaiveBayes()
    gnb.fit(X, y)
    # print the mu_ and var_ and pi_ of the fitted model
    print(f"mu_: {gnb.mu_}")
    print(f"var_: {gnb.vars_}")
    print(f"pi_: {gnb.pi_}")
    pass

if __name__ == '__main__':
    np.random.seed(0)
    run_perceptron()
    compare_gaussian_classifiers()
    #quiz_1()