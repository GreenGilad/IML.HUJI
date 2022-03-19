from IMLearn.learners import UnivariateGaussian, MultivariateGaussian
import numpy as np
import plotly.graph_objects as go
import plotly.io as pio

import plotly.express as px
import os
pio.templates.default = "simple_white"


def test_univariate_gaussian():

    if not os.path.exists("ex1_plots"):
        os.mkdir("ex1_plots")

    # Question 1 - Draw samples and print fitted model
    expectation, variance, num_samples = 10.0, 1, 1000  # expectation, variance, number of samples
    X = np.random.normal(expectation, variance, (num_samples,))
    estimator = UnivariateGaussian()
    estimator.fit(X)
    print(estimator.mu_, estimator.var_)

    # Question 2 - Empirically showing sample mean is consistent
    num_points = 100
    step = num_samples//num_points
    estimated_mu = np.empty((num_points,))
    for i, size in enumerate(range(step, num_samples+step, step)):
        estimated_mu[i] = UnivariateGaussian().fit(X[:size]).mu_
    dist = abs(estimated_mu-expectation)
    plot = px.line(
        x=np.linspace(0,num_samples, num=num_points),
        y=dist,
        title="Univariate Gaussian Estimated Expectation Convergence",
        labels={'x':"no. samples", 'y':"dist estimated from true expectation"}
    )
    plot.show()
    plot.write_image("ex1_plots/univariate_convergence.png")

    # Question 3 - Plotting Empirical PDF of fitted model
    scatter = px.scatter(
        x=X,
        y=estimator.pdf(X),
        title="Empirical PDF -- Gaussian Estimator",
        labels={'x':"sample", 'y':"empirical PDF value"}
    )
    scatter.show()
    scatter.write_image("ex1_plots/univariate_scatterplot.png")


def test_multivariate_gaussian():

    if not os.path.exists("ex1_plots"):
        os.mkdir("ex1_plots")

    # Question 4 - Draw samples and print fitted model
    mu = np.array([0,0,4,0])
    S = np.array([
        [1.0, 0.2, 0.0, 0.5],
        [0.2, 2.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.5, 0.0, 0.0, 1.0]
    ])

    num_samples = 1000

    X = np.random.multivariate_normal(mu, S, num_samples)
    estimator = MultivariateGaussian()
    estimator.fit(X)
    print(estimator.mu_)
    print(estimator.cov_)

    # Question 5 - Likelihood evaluation
    size = 200
    rrange = (-10, 10)
    f = np.repeat(np.linspace(*rrange, size).reshape(-1,1), size, 1)
    Mu = np.dstack((f, np.zeros_like(f), f.T, np.zeros_like(f)))
    likelihood = MultivariateGaussian.log_likelihood(Mu, S, X)

    heatmap = px.imshow(
        likelihood,
        title="Multivariate Gaussian Log Likelihood -- Mu Space",
        labels={'x': "f3", 'y': 'f1'},
        x=np.linspace(*rrange, size),
        y=np.linspace(*rrange, size),
        color_continuous_scale="magma"
    )
    heatmap.show()
    heatmap.write_image("ex1_plots/multivariate_heatmap.png")

    # Question 6 - Maximum likelihood
    max_model_indices = np.unravel_index(np.argmax(likelihood), likelihood.shape)
    max_model = f[max_model_indices[0], 0], f[max_model_indices[1], 0]
    print(max_model)


if __name__ == '__main__':
    np.random.seed(0)
    test_univariate_gaussian()
    test_multivariate_gaussian()
