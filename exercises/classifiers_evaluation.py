from IMLearn.learners.classifiers import Perceptron, LDA, GaussianNaiveBayes
import numpy as np
from typing import Tuple
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots
from IMLearn.metrics.loss_functions import misclassification_error

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
    arr = np.load(filename)
    np.insert(arr, 0, 1, axis=1)
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
        my_perc = Perceptron(callback=callfunc).fit(X, y)
        # Plot figure
        print(len(losses), losses)
        res.add_trace(go.Scatter(x=list(range(1, len(losses) + 1)), y=losses), row=counter, col=1)
        counter = counter + 1
    res.write_html('test.html', auto_open=True)


def compare_gaussian_classifiers():
    """
    Fit both Gaussian Naive Bayes and LDA classifiers on both gaussians1 and gaussians2 datasets
    """
    for f in ["gaussian1.npy", "gaussian2.npy"]:
        # Load dataset
        X, y = load_dataset(r'C:\Users\Lenovo\Documents\GitHub\IML.HUJI\datasets//' + f)

        # Fit models and predict over training set
        my_lda = LDA().fit(X, y)
        print(my_lda.predict(X))

        # Plot a figure with two suplots, showing the Gaussian Naive Bayes predictions on the left and LDA predictions
        # on the right. Plot title should specify dataset used and subplot titles should specify algorithm and accuracy
        from IMLearn.metrics import accuracy
        raise NotImplementedError()


if __name__ == '__main__':
    np.random.seed(0)
    # run_perceptron()
    compare_gaussian_classifiers()
