from IMLearn.learners import UnivariateGaussian, MultivariateGaussian
import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
pio.templates.default = "simple_white"


def test_univariate_gaussian():
    # Question 1 - Draw samples and print fitted model
    univariate_normal = UnivariateGaussian()
    data = np.random.normal(10, 1, 1000)
    univariate_normal.fit(data)
    print("(" + str(univariate_normal.mu_) + "," + str(univariate_normal.var_) + ")")

    # Question 2 - Empirically showing sample mean is consistent
    sample_sizes = range(10, 1001, 10)
    data = np.random.normal(10, 1, 1000)
    distances = []
    for sample_size in sample_sizes:
        univariate_normal.fit(data[:sample_size])
        distances.append(np.abs(univariate_normal.mu_ - 10))

    fig = go.Figure(data=go.Scatter(x=np.array(sample_sizes), y=distances))
    fig.update_layout(title="Distance from estimated expectation to real expectation as an output of sample size"
                            "for Univariate Gaussian with expectation 10, variance 1",
                      xaxis_title="Sample size",
                      yaxis_title="Distance from estimated expectation to real expectation")\
        .show()

    # Question 3 - Plotting Empirical PDF of fitted model
    univariate_normal.fit(data)
    y = univariate_normal.pdf(data)
    fig = go.Figure(go.Scatter(x=data, y=y, mode='markers'))
    fig.update_layout(
        title="PDFs values as an output of sample values for Univariate Gaussian with expectation 10, variance 1",
        xaxis_title="Sample values",
        yaxis_title="PDFs values")
    fig.show()


def test_multivariate_gaussian():
    # Question 4 - Draw samples and print fitted model
    sigma = np.array([[1, 0.2, 0, 0.5],
                      [0.2, 2, 0, 0],
                      [0, 0, 1, 0],
                      [0.5, 0, 0, 1]])
    data = np.random.multivariate_normal(np.array([0, 0, 4, 0]),
                                         sigma,
                                         1000)
    multivariate_normal = MultivariateGaussian()
    multivariate_normal.fit(data)
    print(multivariate_normal.mu_)
    print(multivariate_normal.cov_)

    # Question 5 - Likelihood evaluation
    f1, f3 = np.linspace(-10, 10, 200), np.linspace(-10, 10, 200)
    results = [[MultivariateGaussian.log_likelihood(np.array([i, 0, j, 0]).transpose(), sigma, data) for i in f1]
               for j in f3]

    results = np.array(results)
    go.Figure(go.Heatmap(x=f1, y=f3, z=results),
              layout=go.Layout(
                  title="Log likelihood As Function of [f1,0,f3,0] expectation values",
                  xaxis_title="f1",
                  yaxis_title="f3")) \
        .show()
    # Question 6 - Maximum likelihood
    f3_index, f1_index = np.where(results == np.amax(results))
    max_f1, max_f3 = f1[f1_index], f3[f3_index]
    str_max_f1 = f"{float(max_f1):.3f}"
    str_max_f3 = f"{float(max_f3):.3f}"
    print("maximum log-likelihood achieved for f1 " + str_max_f1 + ", f3 " + str_max_f3)


if __name__ == '__main__':
    np.random.seed(0)
    test_univariate_gaussian()
    test_multivariate_gaussian()
