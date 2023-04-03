from IMLearn.learners import UnivariateGaussian, MultivariateGaussian
from typing import Iterable
from plotly.subplots import make_subplots
import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
pio.templates.default = "simple_white"


SAMPLES = 1000
STEP = 10
START = 10
MEAN = 10
VAR = 1

MULT_MEAN = np.array([0, 0, 4, 0])
MULT_MEAN.transpose()
COV = np.array([[1, 0.2, 0, 0.5],
                  [0.2, 2, 0, 0],
                  [0, 0, 1, 0],
                  [0.5, 0, 0, 1]])
HM_START = -10
HM_STOP = 10
HM_NUM = 200


def plot_univariate(samples: np.ndarray, sizes: Iterable[int], exp):
    gaussians = []
    for size in sizes:
        g = UnivariateGaussian()
        gaussians.append(g)
        g.fit(samples[:size])

    means = np.array([g.mu_ for g in gaussians])
    deviation = np.abs(means - exp)
    fig = make_subplots(rows=1, cols=1).add_traces([go.Scatter(x=list(sizes), y=deviation, mode='markers',
                                                               marker=dict(color='black'))])
    fig.update_layout(title_text="Absolute Deviation of Mean Estimator as Function of Sample Size")\
        .update_yaxes(title_text="Absolute Deviation of Mean Estimator")\
        .update_xaxes(title_text="Sample Size")
    fig.show()


def plot_pdf_univariate(g: UnivariateGaussian, samples: np.ndarray):
    pdfs = g.pdf(samples)
    sorted_copy = np.array(samples)
    np.ndarray.sort(sorted_copy)
    fig = make_subplots(rows=1, cols=1).add_traces([go.Scatter(x=sorted_copy, y=pdfs, mode='markers+lines',
                                                               marker=dict(color='black'))])
    fig.update_layout(title_text="Empirical PDF Under Fitted Model") \
        .update_yaxes(title_text="PDF") \
        .update_xaxes(title_text="Sample Value")
    fig.show()


def test_univariate_gaussian():
    # Question 1 - Draw samples and print fitted model
    uni_s = np.random.normal(MEAN, VAR, SAMPLES)
    gaussian = UnivariateGaussian()
    gaussian.fit(uni_s)
    print("Expectation, Variance: (%s, %s)" % (np.round(gaussian.mu_, 3),  np.round(gaussian.var_, 3)))

    # Question 2 - Empirically showing sample mean is consistent
    plot_univariate(uni_s, range(START, SAMPLES + 1, STEP), MEAN)

    # Question 3 - Plotting Empirical PDF of fitted model
    plot_pdf_univariate(gaussian, uni_s)


def plot_log_likelihood(cov, samples):
    likelihood_mtx = np.zeros((200, 200))
    lnspc = np.linspace(HM_START, HM_STOP, HM_NUM)
    for i, f1 in enumerate(lnspc):
        for j, f3 in enumerate(lnspc):
            likelihood_mtx[i, j] = MultivariateGaussian.log_likelihood(np.array([f1, 0, f3, 0]), cov, samples)

    go.Figure(go.Heatmap(x=lnspc, y=lnspc, z=likelihood_mtx),
              layout=dict(template='plotly_dark',
                          title='Log Likelihood of Multivariate Gaussian as Function Expectation Features 1 and 3',
                          xaxis_title='Expectation feature 3',
                          yaxis_title='Expectation feature 1')).show()

    print("Maximum likelihood achieved for [f1, f3]:\n",
          np.round(lnspc[list(np.unravel_index(likelihood_mtx.argmax(), likelihood_mtx.shape))], 3))


def test_multivariate_gaussian():
    # Question 4 - Draw samples and print fitted model
    multi_s = np.random.multivariate_normal(MULT_MEAN, COV, SAMPLES)
    gaussian = MultivariateGaussian()
    gaussian.fit(multi_s)
    print("Estimated expectation:\n", np.round(gaussian.mu_, 3))
    print("Estimated covariance:\n", np.round(gaussian.cov_, 3))

    # Question 5 - Likelihood evaluation + # Question 6 - Maximum likelihood
    plot_log_likelihood(COV, multi_s)


if __name__ == '__main__':
    np.random.seed(0)
    test_univariate_gaussian()
    test_multivariate_gaussian()
