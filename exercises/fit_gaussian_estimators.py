

from IMLearn.learners import UnivariateGaussian, MultivariateGaussian
import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
import plotly.express as px
pio.templates.default = "simple_white"


def test_univariate_gaussian():
    # Question 1 - Draw samples and print fitted model
    uni = UnivariateGaussian()
    mu, var = 10, 1
    normal_univariate_samples = np.random.normal(mu, var, 1000)
    uni.fit(normal_univariate_samples)
    print(tuple([uni.mu_, uni.var_]))

    # Question 2 - Empirically showing sample mean is consistent
    plot_distances(mu, normal_univariate_samples)

    # Question 3 - Plotting Empirical PDF of fitted model
    uni.fit(normal_univariate_samples)
    pdf_res = uni.pdf(normal_univariate_samples)
    plot_pdfs(normal_univariate_samples, pdf_res)


def test_multivariate_gaussian():
    multi = MultivariateGaussian()
    # Question 4 - Draw samples and print fitted model
    mu2 = [0, 0, 4, 0]
    var2 = np.matrix([[1, 0.2, 0, 0.5],
                     [0.2, 2, 0, 0],
                     [0, 0, 1, 0],
                     [0.5, 0, 0, 1]])

    normal_multivariate_samples = np.random.multivariate_normal(mu2, var2, 1000)
    multi.fit(normal_multivariate_samples)
    print("estimated expectation")
    print(multi.mu_)
    print("estimated covariance")
    print(multi.cov_)

    # Question 5 - Likelihood evaluation
    vals = np.linspace(-10, 10, 200)
    outer_log = []
    max_f1 = None
    max_f3 = None
    max_log_likelihood = None

    for i in range(0, 200):
        inner_log = []
        for j in range(200):
            tempt_mu = np.array([vals[i], 0, vals[j], 0])
            log_likelihood = MultivariateGaussian.\
                log_likelihood(tempt_mu, var2, normal_multivariate_samples)
            inner_log.append(log_likelihood)
            # get the maximal log-likelihood
            if (max_log_likelihood is None) or (max_log_likelihood < log_likelihood):
                max_log_likelihood = log_likelihood
                max_f1 = vals[i]
                max_f3 = vals[j]
        outer_log.append(inner_log)
    fig = go.Figure(data=go.Heatmap(x=vals, y=vals, z=outer_log))
    fig.update_layout(
        title="Log-likelihood of different expectation values",
        xaxis_title="f3 values",
        yaxis_title="f1 values")
    fig.show()

    # Question 6 - Maximum likelihood
    print("maximal log-likelihood:")
    print(max_log_likelihood)
    print("Maximizer f1 and f3:")
    print(tuple([max_f1, max_f3]))


def plot_pdfs(normal_samples, pdfs):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=normal_samples, y=pdfs, mode='markers',
                             marker=dict(color="Blue"),
                             legendgroup="female", name="female"))

    fig.update_xaxes(title_text="Samples values")
    fig.update_yaxes(title_text="PDFs values")
    fig.update_layout(
        title="PDFs for given samples")
    fig.show()


def plot_distances(mu, samples):
    uni = UnivariateGaussian()
    expectation_distances = np.ndarray((100,))
    samples_sizes = np.ndarray((100,))
    j = 0
    for i in range(10, 1010, 10):
        uni.fit(samples[:i])
        expectation_distances[j] = abs(mu - uni.mu_)
        samples_sizes[j] = i
        j += 1

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=samples_sizes,
        y=expectation_distances))

    fig.update_layout(
        title="Expectation estimator",
        xaxis_title="Sample size",
        yaxis_title="Expectation distances")
    fig.show()


if __name__ == '__main__':
    np.random.seed(0)
    test_univariate_gaussian()
    test_multivariate_gaussian()

