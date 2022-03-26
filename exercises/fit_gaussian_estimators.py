from sklearn.covariance import log_likelihood
from IMLearn.learners import UnivariateGaussian, MultivariateGaussian
import numpy as np
import plotly
import plotly.graph_objects as go
import plotly.io as pio
import math
pio.templates.default = "simple_white"


def test_univariate_gaussian():
    # Question 1 - Draw samples and print fitted model
    origin_mu = 10
    X = np.random.normal(origin_mu, 1, size=1000)
    uni = UnivariateGaussian().fit(X)
    print((uni.mu, uni.var))

    # Question 2 - Empirically showing sample mean is consistent
    sizes = np.arange(10, 1010, step=10)
    fixed_by_size = np.vectorize(lambda size: np.abs(uni.fit(X[:size]).mu-origin_mu))(sizes)
    fig = go.Figure(
        layout = go.Layout(
            title = 'Distance from Mean by Sample Size',
            xaxis = dict(title='Sample Size'),
            yaxis = dict(title='Absolute Distance of Estimated and True Expectation'),
        ),
        data = [go.Bar(x=sizes, y=fixed_by_size)],
    ).show()
    
    # Question 3 - Plotting Empirical PDF of fitted model
    fig = go.Figure(
        layout = go.Layout(
            title = 'Empirical PDF of Fitted Model',
            xaxis = dict(title='Ordered Sample Values'),
            yaxis = dict(title='PDF'),
        ),
        data = go.Scatter(x=X, y=uni.pdf(X), mode='markers'),
    ).show() 

def test_multivariate_gaussian():
    # Question 4 - Draw samples and print fitted model
    mu = np.array([0, 0, 4, 0])
    sigma = np.array([
        np.array([1, 0.2, 0, 0.5]),
        np.array([0.2, 2, 0, 0]),
        np.array([0, 0, 1, 0]),
        np.array([0.5, 0, 0, 1])
    ])
    X = np.random.multivariate_normal(mean=mu, cov=sigma, size=1000)
    multi = MultivariateGaussian().fit(X)
    print(multi.mu)
    print(multi.cov)
    
    # Question 5 - Likelihood evaluation
    max_log_lh = (None, -math.inf) # For q6: (<pair>, <log lh val>)
    log_likelihood_matrix = []
    
    f1 = np.linspace(-10, 10, 200)
    f3 = np.linspace(-10, 10, 200)
    
    for v_i in f1:
        row_values = []
        for v_j in f3:
            log_lh = multi.log_likelihood(mu=np.array([v_i, 0, v_j, 0]), cov=sigma, X=X)
            row_values.append(log_lh)
            if log_lh > max_log_lh[1]:
                max_log_lh = ((v_i, v_j), log_lh)
        log_likelihood_matrix.append(row_values)
    
    go.Figure(
        layout = go.Layout(
            title = 'Log Likelihood Heatmap',
            xaxis = dict(title='f1 (All values between -10 to 10 with steps of 200)'),
            yaxis = dict(title='f3 (All values between -10 to 10 with steps of 200)'),
        ),
        data = go.Heatmap(x=f1, y=f3, z=log_likelihood_matrix),
    ).show()
    
    # Question 6 - Maximum likelihood
    print(round(max_log_lh[0][0], 3), round(max_log_lh[0][1], 3))

if __name__ == '__main__':
    np.random.seed(0)
    test_univariate_gaussian()
    test_multivariate_gaussian()
