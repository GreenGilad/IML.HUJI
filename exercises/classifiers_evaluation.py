import IMLearn.metrics
from IMLearn.learners.classifiers import Perceptron, LDA, GaussianNaiveBayes
import numpy as np
from typing import Tuple
import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio
from plotly.subplots import make_subplots
pio.templates.default = "simple_white"

from os import path # todo?


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
    raise NotImplementedError()

def run_perceptron():
    """
    Fit and plot fit progression of the Perceptron algorithm over both the linearly separable and inseparable datasets

    Create a line plot that shows the perceptron algorithm's training loss values (y-axis)
    as a function of the training iterations (x-axis).
    """
    for n, f in [("Linearly Separable", "linearly_separable.npy"), ("Linearly Inseparable", "linearly_inseparable.npy")]:
        # Load dataset
        data = np.load(path.join('../datasets', f))
        losses = []

        # Fit Perceptron and record loss in each fit iteration
        perceptron_classifier = Perceptron(callback=
                            lambda fit, x, y: losses.append(fit.loss(x, y)))

        perceptron_classifier.fit(data[:, :-1], data[:, -1])

        # Plot figure
        perceptron_loss_fig = px.line(x=np.arange(len(losses)), y=losses)
        perceptron_loss_fig.update_layout(title_text=n, title_x=0.5)
        perceptron_loss_fig.show()


def compare_gaussian_classifiers():
    """
    Fit both Gaussian Naive Bayes and LDA classifiers on both gaussians1 and gaussians2 datasets
    """
    for f in ["gaussian1.npy", "gaussian2.npy"]:
        symbols_dict = {0: 'circle', 1: 'square', 2: 'diamond'}
        colors_dict = {0: 'lightsalmon', 1: 'limegreen', 2: 'mediumblue'}


        # Load dataset
        data = np.load(path.join('../datasets', f))

        samples = data[:, :-1]
        labels = data[:, -1]

        # initialize classifiers:
        gaussian_classifier = GaussianNaiveBayes()
        lda_classifier = LDA()

        # Fit models and predict over training set
        gaussian_classifier.fit(samples, labels)
        lda_classifier.fit(samples, labels)

        # Predict over samples:
        gc_response = gaussian_classifier.predict(samples)
        lda_response = lda_classifier.predict(samples)

        # Plot a figure with two suplots, showing the Gaussian Naive Bayes predictions on the left and LDA predictions
        # on the right. Plot title should specify dataset used and subplot titles should specify algorithm and accuracy


        from IMLearn.metrics import accuracy

        # fig = px.scatter(data, x=0, y=1, color=2, labels={0: 'Class 0', 1: 'Class 1', 2: 'Class 2'})

        true_symbols = [symbols_dict[f] for f in labels]
        gc_response_colors = [colors_dict[f] for f in gc_response.reshape(-1, )]
        lda_response_colors = [colors_dict[f] for f in lda_response.reshape(-1, )]

        fig = make_subplots(rows=1, cols=2, subplot_titles=('Gaussian Naive Bayes', 'LDA'))

        fig.add_trace(go.Scatter(x=samples[:, 0], y=samples[:, 1],
                                 mode='markers',
                                 marker=dict(color=gc_response_colors,
                                             symbol=true_symbols,
                                             size=10,
                                             line=dict(width=2)
                                             )
                                 ),
                      row=1, col=1)

        fig.add_trace(go.Scatter(x=samples[:, 0], y=samples[:, 1],
                                 mode='markers',
                                 marker=dict(color=lda_response_colors,
                                             symbol=true_symbols,
                                             size=10,
                                             line=dict(width=2)
                                             )
                                 ),
                      row=1, col=2)

        fig.add_shape(type="line",
                      path=confidence_2d_ellipse_helper(lda_classifier.mu_[0], lda_classifier.cov_),
                      line=dict(width=2),
                      line_color="black",
                      row=1, col=1)

        fig.update_layout(title_text=f, title_x=0.5)
        fig.show()


def confidence_2d_ellipse_helper(mu, cov_matrix):

    pearson = cov_matrix[0, 1] / np.sqrt(cov_matrix[0, 0] * cov_matrix[1, 1])

    ellipse_r_x = np.sqrt(1 + pearson)
    ellipse_r_y = np.sqrt(1 - pearson)
    step = np.linspace(0, 2 * np.pi, 100)
    ellipse_coords = np.column_stack(
        [ellipse_r_x * np.cos(step), ellipse_r_y * np.sin(step)])

    x_scale = np.sqrt(cov_matrix[0, 0])
    y_scale = np.sqrt(cov_matrix[1, 1])

    translation_matrix = np.tile([mu[0], mu[1]],
                                 (ellipse_coords.shape[0], 1))

    rotation_matrix = np.array([[np.cos(np.pi / 4), np.sin(np.pi / 4)],
                                [-np.sin(np.pi / 4), np.cos(np.pi / 4)]])

    scale_matrix = np.array([[x_scale, 0],
                             [0, y_scale]])

    ellipse_coords = ellipse_coords.dot(rotation_matrix).dot(
        scale_matrix) + translation_matrix

    path = f'M {ellipse_coords[0, 0]}, {ellipse_coords[0, 1]}'
    for k in range(1, len(ellipse_coords)):
        path += f'L{ellipse_coords[k, 0]}, {ellipse_coords[k, 1]}'
    path += ' Z'
    return path


if __name__ == '__main__':
    np.random.seed(0)
    # run_perceptron() # todo remove comment
    compare_gaussian_classifiers()
