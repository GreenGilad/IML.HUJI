from IMLearn.learners import UnivariateGaussian, MultivariateGaussian
import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
pio.templates.default = "simple_white"


def test_univariate_gaussian():
    # Question 1 - Draw samples and print fitted model
    univariate_gaussian = UnivariateGaussian()
    normal_samples = np.random.normal(10,1,1000)
    univariate_gaussian.fit(normal_samples)
    print(univariate_gaussian.mu_, univariate_gaussian.var_)

    # Question 2 - Empirically showing sample mean is consistent
    sample_sizes = []
    results = []
    for i in range(1,101):
        sample_size = i*10
        sample_sizes.append(sample_size)
        univariate_gaussian.fit(normal_samples[:sample_size])
        results.append(np.abs(univariate_gaussian.mu_-10))
    go.Figure([go.Scatter(x=sample_sizes, y=results, mode='markers+lines', name=r'$\widehat\mu$')],
              layout=go.Layout(title=r"$\text{Estimation of Expectation As Function Of Number Of Samples}$",
                               xaxis_title="$m\\text{ - number of samples}$",
                               yaxis_title="$\\text{Estimated expceptions' distance from true value}$",
                               height=500)).show()

    # Question 3 - Plotting Empirical PDF of fitted model
    pdf_list = univariate_gaussian.pdf(normal_samples)
    go.Figure([go.Scatter(x=normal_samples, y=pdf_list, mode='markers', name=r'$\widehat\mu$')],
              layout=go.Layout(title=r"$\text{PDF of samples}$",
                               xaxis_title="$\\text{Sample}$",
                               yaxis_title="$\\text{PDF of Sample}$",
                               height=500)).show()

def test_multivariate_gaussian():
    # Question 4 - Draw samples and print fitted model
    multivariate_gaussian = MultivariateGaussian()
    mu = np.array([0,0,4,0]).transpose()
    sigma = np.array([1,0.2,0,0.5,0.2,2,0,0,0,0,1,0,0.5,0,0,1]).reshape((4,4))
    noraml_multi_samples = np.random.multivariate_normal(mu,sigma,1000)
    multivariate_gaussian.fit(noraml_multi_samples)
    print(multivariate_gaussian.mu_)
    print(multivariate_gaussian.cov_)

    # Question 5 - Likelihood evaluation
    f1 = np.linspace(-10,10,200)
    f3 = np.linspace(-10,10,200)
    log_likelihood_results =[]
    for x in f1:
        mid_list = []
        for y in f3:
            mu = np.array([x,0,y,0])
            mid_list.append(multivariate_gaussian.log_likelihood(mu,sigma,noraml_multi_samples))
        log_likelihood_results.append(mid_list)
    go.Figure([go.Heatmap(x=f3, y=f1, z=log_likelihood_results)],
              layout=go.Layout(title=r"$\text{Heatmap of the log likelihood of multivariate noraml distribution}$",
                               xaxis_title="$\\text{f1 values}$",
                               yaxis_title="$\\text{f2 values}$",
                               height=1000,width=1000)).show()

    # Question 6 - Maximum likelihood
    max_vals = np.unravel_index(np.argmax(log_likelihood_results),np.shape(log_likelihood_results))
    print(max_vals, log_likelihood_results[99][139])
    quiz_a = f1[max_vals[0]]
    quiz_b = f3[max_vals[1]]


if __name__ == '__main__':
    np.random.seed(0)
    test_univariate_gaussian()
    test_multivariate_gaussian()
