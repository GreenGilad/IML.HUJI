import numpy as np
import plotly.graph_objects as go
import plotly.io as pio

from IMLearn.learners import *

pio.templates.default = "simple_white"


def test_univariate_gaussian():
    # Question 1 - Draw samples and print fitted model
    X = np.random.normal(10, 1, 1000)
    univ_test = UnivariateGaussian()
    univ_test.fit(X)
    print(f"({univ_test.mu_}, {univ_test.var_})")

    # Question 2 - Empirically showing sample mean is consistent
    ms = np.linspace(0, 1000, 100).astype(int)
    estimated_mean = []
    i = 10
    while i < 1000:
        Y = X[:i]
        univ_test.fit(Y)
        estimated_mean.append(abs(univ_test.mu_ - 10))
        i += 10

    go.Figure(
        [go.Scatter(x=ms, y=estimated_mean, mode='markers+lines',
                    name=r'$\widehat\mu$')],
        layout=go.Layout(
            title=r"$\text{Estimation of Expectation Differrence As Function "
                  r"Of Number Of Samples}$",
            xaxis_title="$m\\text{ - number of samples}$",
            yaxis_title="r$|\hat\mu - \mu|$",
            height=300, width=1000)).show()

    # Question 3 - Plotting Empirical PDF of fitted model
    univ_test.fit(X)
    pdf = univ_test.pdf(X)

    go.Figure(
        [go.Scatter(x=X, y=pdf, mode='markers',
                    name=r'$\widehat\mu$')],
        layout=go.Layout(
            title=r"$\text{Estimation of PDF As Function Of Number Of Samples}$",
            xaxis_title="$m\\text{ - number of samples}$",
            yaxis_title="r$pdf$",
            height=500, width=900)).show()


def test_multivariate_gaussian():
    # Question 4 - Draw samples and print fitted model
    mu = [0, 0, 4, 0]
    sigma = np.array([[1, 0.2, 0, 0.5],
                      [0.2, 2, 0, 0],
                      [0, 0, 1, 0],
                      [0.5, 0, 0, 1]])

    S = np.random.multivariate_normal(mu, sigma, 1000)

    multi_test = MultivariateGaussian()
    multi_test.fit(S)

    print(multi_test.mu_)
    print(multi_test.cov_)

    # Question 5 - Likelihood evaluation
    def response(x):
        mu = np.array([x[0], 0, x[1], 0])
        return multi_test.log_likelihood(mu, sigma, S)

    f1 = np.array(np.linspace(-10, 10, 200))
    f3 = np.array(np.linspace(-10, 10, 200))
    xy = np.array(np.meshgrid(f1, f3)).T.reshape(200, 200, 2)
    z = np.array([list(map(response, xy[i])) for i in range(200)])

    go.Figure(data=go.Heatmap(x=f1, y=f3, z=z), layout=go.Layout(
        title=r"$\text{Estimation of log-likelihood As Function Of"
              r" f1 and f3}$",
        xaxis_title="$f1$",
        yaxis_title="r$f3$",
        height=500, width=900)).show()

    # Question 6 - Maximum likelihood
    row, col = np.where(z == np.amax(z))
    print(f1[row], f3[col])


if __name__ == '__main__':
    np.random.seed(0)
    test_univariate_gaussian()
    test_multivariate_gaussian()
