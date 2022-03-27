import plotly

from IMLearn.learners import UnivariateGaussian, MultivariateGaussian
import numpy as np
import plotly.graph_objects as go
import plotly.io as pio

pio.templates.default = "simple_white"


def test_univariate_gaussian():
    # Question 1 - Draw samples and print fitted model
    sample = np.random.normal(10, 1, 1000)
    uni_sample = UnivariateGaussian()
    uni_sample.fit(sample)
    print(uni_sample.mu_, uni_sample.var_)

    # Question 2 - Empirically showing sample mean is consistent
    increasing_samples = np.array([sample[:i] for i in range(10, 1001, 10)])
    increasing_fitted_param = []
    for samp in increasing_samples:
        temp_object = UnivariateGaussian()
        temp_object.fit(samp)
        increasing_fitted_param.append(abs(temp_object.mu_ - 10))

    ms = np.linspace(10, 1001, 100).astype(np.int64)
    figure = go.Figure([go.Scatter(x=ms, y=increasing_fitted_param, mode='lines', name=r'$\widehat\mu$')],
                       layout=go.Layout(title={'text': 'Estimation of Expectation As Function Of Number Of Samples'},
                                        xaxis_title={'text': 'Number of observations'},
                                        yaxis_title={'text': 'difference between the Expectation and the fitted mu'},
                                        height=500))
    figure.write_html('Q2.html', auto_open=True)
    # Question 3 - Plotting Empirical PDF of fitted model
    sample_pdf = uni_sample.pdf(sample)
    figure = plotly.subplots.make_subplots(rows=1, cols=1) \
        .add_traces(
        [go.Scatter(x=sample, y=sample_pdf, mode='markers', marker=dict(color="black"), showlegend=False), ],
        rows=1, cols=1) \
        .add_traces(
        [go.Scatter(x=sample, y=sample_pdf, mode='markers', marker=dict(color="blue")),
         ],
        rows=1, cols=1) \
        .update_layout(
        title={'text': 'Empirical PDF plot of 1000 random samples from the Gaussian distribution mu=10 var=10'},
        height=600) \
        # .show(render= browser)
    figure.write_html('Q3.html', auto_open=True)


def test_multivariate_gaussian():
    # Question 4 - Draw samples and print fitted model
    mu = np.array([0, 0, 4, 0])
    sigma = np.array([[1, 0.2, 0, 0.5], [0.2, 2, 0, 0], [0, 0, 1, 0], [0.5, 0, 0, 1]])
    m_sample = np.random.multivariate_normal(mu, sigma, 1000)
    mul_sample = MultivariateGaussian()
    mul_sample.fit(m_sample)
    print(mul_sample.mu_)
    print(mul_sample.cov_)

    # Question 5 - Likelihood evaluation
    f_1 = np.linspace(-10, 10, 200)
    f_3 = np.linspace(-10, 10, 200)
    log_l_by_list = []
    for i in range(200):
        log_l_by_list.append([])
        for j in range(200):
            temp_mu = np.array([f_1[i], 0, f_3[j], 0])
            log_l_by_list[i].append(MultivariateGaussian.log_likelihood(temp_mu, sigma, m_sample))
    log_l_by_array = np.array(log_l_by_list)
    fig = go.Figure(go.Heatmap(x=f_1, y=f_3, z=log_l_by_array),
                    layout=go.Layout(title='heatmap of log likelihood of diifrent mu vectors',
                                     xaxis_title={'text': '200 different values for mu[0] from the interval [-10,10] '},
                                     yaxis_title={'text': '200 different values for mu[3] from the interval [-10,10] '},
                                     height=1000, width=1000))
    fig.write_html('Q5.html', auto_open=True)
    # Question 6 - Maximum likelihood
    mx = np.max(log_l_by_array)
    row, col = np.where(log_l_by_array == mx)
    print('f_1:', np.round_(f_1[row], 4), 'f_3:', np.round_(f_3[col], 4))


if __name__ == '__main__':
    np.random.seed(0)
    test_univariate_gaussian()
    test_multivariate_gaussian()
