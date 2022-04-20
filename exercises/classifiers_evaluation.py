from IMLearn.learners.classifiers import Perceptron, LDA, GaussianNaiveBayes
from typing import Tuple
from utils import *
import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio
from plotly.subplots import make_subplots
from os import path

pio.templates.default = "simple_white"
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
    # data = np.load(filename)
    # return data[:, :-1], data[:, -1]
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
        # Plot figure of loss as function of fitting iteration
        # Load dataset
        samples, labels = load_dataset(path.join('../datasets', f))
        losses = []

        # Fit Perceptron and record loss in each fit iteration
        perceptron_classifier = Perceptron(callback=
                                           lambda fit, x, y: losses.append(
                                               fit.loss(x, y)))
        # px.scatter(samples, color=labels).show()
        perceptron_classifier.fit(samples, labels)

        # Plot figure
        perceptron_loss_fig = px.line(x=np.arange(len(losses)), y=losses)
        perceptron_loss_fig.update_layout(title_text=n, title_x=0.5)
        perceptron_loss_fig.show()


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
    from IMLearn.metrics import accuracy

    # initialize parameters:
    symbols_dict = {0: 'circle', 1: 'square', 2: 'diamond'}
    colors_dict = {0: 'lightsalmon', 1: 'limegreen', 2: 'mediumblue'}

    # loop over datasets:
    for f in ["gaussian1.npy", "gaussian2.npy"]:
        # Load dataset
        samples, labels = load_dataset(path.join('../datasets', f))
        true_symbols = [symbols_dict[f] for f in labels]

        # initialize classifiers:
        gaussian_classifier = GaussianNaiveBayes()
        lda_classifier = LDA()

        # Fit models and predict over training set
        gaussian_classifier.fit(samples, labels)
        lda_classifier.fit(samples, labels)

        # Predict over samples:
        gc_response = gaussian_classifier.predict(samples)
        lda_response = lda_classifier.predict(samples)

        # Plot figure:
        gc_response_colors = [colors_dict[f] for f in
                              gc_response.reshape(-1, )]
        lda_response_colors = [colors_dict[f] for f in
                               lda_response.reshape(-1, )]

        # initialize figure:
        fig = make_subplots(rows=1, cols=2,
                            subplot_titles=(
                                f'Gaussian Naive Bayes, Accuracy:'
                                f' {round(accuracy(labels, gc_response), 3) * 100}%',
                                f'LDA, Accuracy:'
                                f' {round(accuracy(labels, lda_response), 3) * 100}%'))
        fig.update_layout(title_text=f, title_x=0.5)

        # Plot scatter of the gaussian classifier:
        fig.add_trace(go.Scatter(x=samples[:, 0], y=samples[:, 1],
                                 mode='markers',
                                 marker=dict(color=gc_response_colors,
                                             symbol=true_symbols,
                                             size=10,
                                             line=dict(width=2)
                                             )
                                 ),
                      row=1, col=1)

        # Plot scatter of the lda classifier:
        fig.add_trace(go.Scatter(x=samples[:, 0], y=samples[:, 1],
                                 mode='markers',
                                 marker=dict(color=lda_response_colors,
                                             symbol=true_symbols,
                                             size=10,
                                             line=dict(width=2)
                                             )
                                 ),
                      row=1, col=2)

        # set parameters:
        responses = [gc_response, lda_response]
        classifiers = [gaussian_classifier, lda_classifier]

        # add ellipses to represent the gaussian distributions:
        for i, item in enumerate(zip(classifiers, responses)):
            # extract params:
            classifier, response = item

            # plot classes results:
            for _class in classifier.classes_:
                # class_samples = samples[(response == _class).reshape(-1, )]
                #
                # coords = get_ellipse_coordinates(
                #     _class,
                #     class_samples)
                #
                # fig.add_trace(go.Scatter(
                #     x=coords[:, 0], y=coords[:, 1],
                #     line=dict(color="black", width=4),
                # ),
                #     row=1, col=1 + i)

                fig.add_trace(get_ellipse(classifier.mu_[_class],
                                          classifier.cov_),
                              row=1, col=1 + i)

                # add middle gaussian markers:
                # class_mu = class_samples.mean(axis=0)
                fig.add_trace(go.Scatter(x=[classifier.mu_[_class][0]],  # todo mu of classifier or res?
                                         y=[classifier.mu_[_class][1]],
                                         mode='markers',
                                         marker=dict(color='black',
                                                     symbol='x',
                                                     size=15,
                                                     )),
                              row=1, col=1 + i)

        fig.show()


def get_ellipse_coordinates(_class, class_samples):
    # response parameters:
    mu = class_samples.mean(axis=0)
    cov = (class_samples - mu).T @ (class_samples - mu) / (
                len(class_samples) - 1)

    # radius:
    pearson = cov[0, 1] / np.sqrt(cov[0, 0] * cov[1, 1])
    ellipse_r_x = np.sqrt(1 + pearson)
    ellipse_r_y = np.sqrt(1 - pearson)

    # base coordinates:
    step = np.linspace(0, 2 * np.pi, 100)
    ellipse_coords = np.column_stack(
        [ellipse_r_x * np.cos(step), ellipse_r_y * np.sin(step)])

    # translation:
    translation_matrix = np.tile([mu[0], mu[1]],
                                 (ellipse_coords.shape[0], 1))
    # rotation:
    quarter_pi = (np.pi / 4)
    rotation_matrix = np.array([[np.cos(quarter_pi), np.sin(quarter_pi)],
                                [-np.sin(quarter_pi), np.cos(quarter_pi)]])
    # scale:
    std_num = 2
    scale_matrix = np.array([[np.sqrt(cov[0, 0]) * std_num, 0],
                             [0, np.sqrt(cov[1, 1]) * std_num]])
    ellipse_coords = ellipse_coords @ rotation_matrix @ scale_matrix + \
                     translation_matrix

    return ellipse_coords


if __name__ == '__main__':
    np.random.seed(0)
    # run_perceptron()  # todo remove comment
    compare_gaussian_classifiers()
