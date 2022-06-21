import time
import numpy as np
import gzip
from typing import Tuple

from IMLearn.metrics.loss_functions import accuracy
from IMLearn.learners.neural_networks.modules import FullyConnectedLayer, ReLU, CrossEntropyLoss, softmax
from IMLearn.learners.neural_networks.neural_network import NeuralNetwork
from IMLearn.desent_methods import GradientDescent, StochasticGradientDescent, FixedLR
from IMLearn.utils.utils import confusion_matrix

import plotly.figure_factory as ff
from plotly.subplots import make_subplots
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
pio.templates.default = "simple_white"


def load_mnist() -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Loads the MNIST dataset

    Returns:
    --------
    train_X : ndarray of shape (60,000, 784)
        Design matrix of train set

    train_y : ndarray of shape (60,000,)
        Responses of training samples

    test_X : ndarray of shape (10,000, 784)
        Design matrix of test set

    test_y : ndarray of shape (10,000, )
        Responses of test samples
    """

    def load_images(path):
        with gzip.open(path) as f:
            # First 16 bytes are magic_number, n_imgs, n_rows, n_cols
            raw_data = np.frombuffer(f.read(), 'B', offset=16)
        # converting raw data to images (flattening 28x28 to 784 vector)
        return raw_data.reshape(-1, 784).astype('float32') / 255

    def load_labels(path):
        with gzip.open(path) as f:
            # First 8 bytes are magic_number, n_labels
            return np.frombuffer(f.read(), 'B', offset=8)

    return (load_images('../datasets/mnist-train-images.gz'),
            load_labels('../datasets/mnist-train-labels.gz'),
            load_images('../datasets/mnist-test-images.gz'),
            load_labels('../datasets/mnist-test-labels.gz'))


def plot_images_grid(images: np.ndarray, title: str = ""):
    """
    Plot a grid of images

    Parameters
    ----------
    images : ndarray of shape (n_images, 784)
        List of images to print in grid

    title : str, default="
        Title to add to figure

    Returns
    -------
    fig : plotly figure with grid of given images in gray scale
    """
    side = int(len(images) ** 0.5)
    subset_images = images.reshape(-1, 28, 28)

    height, width = subset_images.shape[1:]
    grid = subset_images.reshape(side, side, height, width).swapaxes(1, 2).reshape(height * side, width * side)

    return px.imshow(grid, color_continuous_scale="gray")\
        .update_layout(title=dict(text=title, y=0.97, x=0.5, xanchor="center", yanchor="top"),
                       font=dict(size=16), coloraxis_showscale=False)\
        .update_xaxes(showticklabels=False)\
        .update_yaxes(showticklabels=False)


if __name__ == '__main__':
    train_X, train_y, test_X, test_y = load_mnist()
    (n_samples, n_features), n_classes = train_X.shape, 10

    # ---------------------------------------------------------------------------------------------#
    # Question 5+6+7: Network with ReLU activations using SGD + recording convergence              #
    # ---------------------------------------------------------------------------------------------#
    # Initialize, fit and test network
    raise NotImplementedError()

    # Plotting convergence process
    raise NotImplementedError()

    # Plotting test true- vs predicted confusion matrix
    raise NotImplementedError()

    # ---------------------------------------------------------------------------------------------#
    # Question 8: Network without hidden layers using SGD                                          #
    # ---------------------------------------------------------------------------------------------#
    raise NotImplementedError()

    # ---------------------------------------------------------------------------------------------#
    # Question 9: Most/Least confident predictions                                                 #
    # ---------------------------------------------------------------------------------------------#
    raise NotImplementedError()

    # ---------------------------------------------------------------------------------------------#
    # Question 10: GD vs GDS Running times                                                         #
    # ---------------------------------------------------------------------------------------------#
    raise NotImplementedError()
