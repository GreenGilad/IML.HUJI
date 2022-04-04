import pandas as pd
from IMLearn.learners import UnivariateGaussian, MultivariateGaussian
import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
import plotly.express as px

pio.templates.default = "simple_white"


def test_univariate_gaussian():
    # Question 1 - Draw samples and print fitted model
    uni_sample = np.random.normal(10, 1, 1000)
    my_univar = UnivariateGaussian()
    my_univar.fit(uni_sample)
    print((my_univar.mu_, my_univar.var_))

    # Question 2 - Empirically showing sample mean is consistent
    # gather samples
    est_mu_dif_lst = []
    for i in range(10, 1001, 10):
        sample = uni_sample[:i]
        univar_temp = UnivariateGaussian()
        univar_temp.fit(sample)
        est_mu_dif_lst.append(abs(univar_temp.mu_ - 10))
    # plot
    for_x = np.linspace(0, 1000, 100).astype(np.int64)
    fig = go.Figure([go.Scatter(x=for_x, y=est_mu_dif_lst, mode='lines', name=r'mu bias'), ],
                    layout=go.Layout(
                        title="Distance of expected value from real parameter as a function of sample size",
                        xaxis_title={'text': "Sample size"},
                        yaxis_title={'text': "Estimate distance from real parameter"}, height=500))

    # fig.write_html('first_figure.html', auto_open=True)
    fig.show()

    # Question 3 - Plotting Empirical PDF of fitted model
    for_y = my_univar.pdf(uni_sample)
    df = pd.DataFrame({'Sample': uni_sample, 'PDF': for_y})
    fig1 = px.scatter(df, title='Plot of PDF as function of random samples from Gaussian distribution', x='Sample',
                      y='PDF', height=500)
    # fig1.write_html('second_figure.html', auto_open=True)
    fig1.show()


def test_multivariate_gaussian():
    # Question 4 - Draw samples and print fitted model
    cov = np.matrix([[1, 0.2, 0, 0.5], [0.2, 2, 0, 0], [0, 0, 1, 0], [0.5, 0, 0, 1]])
    multi_sample = np.random.multivariate_normal(np.array([0, 0, 4, 0]), cov, 1000)
    my_multivar = MultivariateGaussian()
    my_multivar.fit(multi_sample)
    print(my_multivar.mu_)
    print(my_multivar.cov_)

    # Question 5 - Likelihood evaluation
    f1_3 = np.linspace(-10, 10, 200)

    res = []
    for i in range(len(f1_3)):
        temp = []
        for j in range(len(f1_3)):
            # print loading percent
            # print('\r', (i * len(f1_3) + j) * 100 / (len(f1_3) * len(f1_3)), '%', end='')
            temp.append(MultivariateGaussian.log_likelihood(np.array([f1_3[i], 0, f1_3[j], 0]), cov, multi_sample))
        res.append(temp)
    res = np.array(res)

    # heatmap arr
    fig2 = go.Figure(go.Heatmap(x=f1_3, y=f1_3, z=res),
                     layout=go.Layout(title='heatmap of loglikelyhood as a function of f1 & f2', height=1000,
                                      width=1000, xaxis_title={'text': "f3"}, yaxis_title={'text': "f1"}))
    # fig2.write_image('pic.png')
    fig2.show()

    # Question 6 - Maximum likelihood
    mx = np.max(res)
    row, col = np.where(res == mx)
    print('Highest likelyhood values (f1, f2): ', round(float(f1_3[row]), 3), round(float(f1_3[col]), 3))


if __name__ == '__main__':
    np.random.seed(0)
    test_univariate_gaussian()
    test_multivariate_gaussian()
