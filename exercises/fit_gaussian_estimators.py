import math

from plotly.offline import iplot

from IMLearn.learners import UnivariateGaussian, MultivariateGaussian
import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import plotly.express as px
import pandas as pd

pio.templates.default = "simple_white"


def test_univariate_gaussian():
    # Question 1 - Draw samples and print fitted model
    univariateGaussian = UnivariateGaussian()
    data = np.random.normal(10, 1, 1000)
    univariateGaussian.fit(data)
    print((univariateGaussian.mu_, univariateGaussian.var_))

    # Question 2 - Empirically showing sample mean is consistent
    x = np.zeros(100)
    for i in range(100):
        x[i] = (i + 1) * 10
    y = np.zeros(100)
    for i in range(100):
        y[i] = abs(univariateGaussian.fit(data[:(i + 1) * 10]).mu_ - 10)
    trace = go.Scatter(x=x, y=y)
    fig = make_subplots()
    fig.append_trace(trace, 1, 1)
    fig.layout.update(title="Empirically showing sample mean is consistent")
    fig.layout["xaxis1"].update(title="Number of samples")
    fig.layout["yaxis1"].update(title="Absolute distance between the estimated and true value of the expectation")
    iplot(fig)

    # Question 3 - Plotting Empirical PDF of fitted model
    px.scatter(pd.DataFrame({"sample values": data, "pdf": univariateGaussian.pdf(data)}),
               x="sample values", y="pdf", title="Plotting Empirical PDF of fitted model").show()



def test_multivariate_gaussian():
    # Question 4 - Draw samples and print fitted model
    multivariateGaussian = MultivariateGaussian()
    m = np.array([0, 0, 4, 0])
    c = np.array([[1, 0.2, 0, 0.5], [0.2, 2, 0, 0], [0, 0, 1, 0], [0.5, 0, 0, 1]])
    data = np.random.multivariate_normal(m, c, 1000)
    multivariateGaussian.fit(data)
    print(multivariateGaussian.mu_)
    print(multivariateGaussian.cov_)

    # Question 5 - Likelihood evaluation
    mat = np.array([[0.0] * 200 for _ in range(200)])
    samples = np.linspace(-10, 10, 200)
    max_val = (0, 0, 0)
    for i, f1 in enumerate(samples):
        for j, f3 in enumerate(samples):
            mat[i][j] = MultivariateGaussian.log_likelihood(np.array([f1, 0, f3, 0]), c, data)
            if (i == j == 0) or (mat[i][j] > max_val[0]):
                max_val = (mat[i][j], i, j)
    fig = px.imshow(mat, x=samples, y=samples)
    fig.layout.update(title="log likelihood heatmap")
    fig.layout["xaxis1"].update(title="f3")
    fig.layout["yaxis1"].update(title="f1")
    fig.show()

    # Question 6 - Maximum likelihood
    print("f1 =" + str(samples[max_val[1]]), "f3 =" + str(samples[max_val[2]]))


if __name__ == '__main__':
    np.random.seed(0)
    test_univariate_gaussian()
    test_multivariate_gaussian()
