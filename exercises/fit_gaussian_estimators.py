from IMLearn.learners import UnivariateGaussian, MultivariateGaussian
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio

pio.templates.default = "simple_white"


def test_univariate_gaussian():
    # initialize the estimator:
    univariate_estimator = UnivariateGaussian()

    # ----------------------------------------------------------------------
    # Question 1 - Draw samples and print fitted model
    # ----------------------------------------------------------------------

    # samples parameters:
    n_samples = 1000
    mu = 10
    sigma = 1

    # generating samples:
    samples = np.random.normal(loc=mu, scale=sigma, size=n_samples)

    # fitting a normal distribution parameters model:
    q1_model = univariate_estimator.fit(samples)

    # print the output in the "(<expectation>, <variance>)" format:
    print(f'({q1_model.mu_}, {q1_model.var_})')

    # ----------------------------------------------------------------------
    # Question 2 - Empirically showing sample mean is consistent
    # ----------------------------------------------------------------------

    # error parameters:
    size_step = 10

    # initialize error df:
    error_df = pd.DataFrame(columns=['sample_step', 'abs_error'])
    error_df.sample_step = (np.arange(0, n_samples, size_step) + size_step)
    error_df.abs_error = [univariate_estimator.fit(samples[:step]).mu_ for
                          step in error_df.sample_step]

    # plotting the error:
    q2_fig = px.line(error_df, x='sample_step', y='abs_error')
    q2_fig.update_layout(title_text='Absolute error of expected value by '
                                    'sample size in gaussian distribution',
                         title_x=0.5)
    q2_fig.update_xaxes(title_text="Sample size")
    q2_fig.update_yaxes(title_text="Absolute mu error")
    q2_fig.show()

    # ----------------------------------------------------------------------
    # Question 3 - Plotting Empirical PDF of fitted model
    # ----------------------------------------------------------------------

    # initialize df for the pdf
    pdf_df = pd.DataFrame({'samples': samples,
                           'pdf': q1_model.pdf(samples)})
    # plotting the pdf:
    q3_fig = px.scatter(pdf_df, x='samples', y='pdf')
    q3_fig.update_layout(title_text='pdf of normally distributed samples',
                         title_x=0.5)

    # What are you expecting to see in the plot?
    #   - the normal distribution (gaussian curve) of the samples.

    q3_fig.show()


def test_multivariate_gaussian():
    # initialize the estimator:
    multivariate_estimator = MultivariateGaussian()

    # ----------------------------------------------------------------------
    # Question 4 - Draw samples and print fitted model
    # ----------------------------------------------------------------------

    # samples parameters:
    n_samples = 1000
    mu = np.array([0, 0, 4, 0])
    sigma = np.array([[1, 0.2, 0, 0.5],
                      [0.2, 2, 0, 0],
                      [0, 0, 1, 0],
                      [0.5, 0, 0, 1]])

    # generating samples:
    samples = np.random.multivariate_normal(mean=mu, cov=sigma, size=n_samples)

    # fitting a normal distribution parameters model:
    q4_model = multivariate_estimator.fit(samples)

    # print the putput in the "(<expectation>, <variance>)" format:
    print(q4_model.mu_)
    print(q4_model.cov_)

    # ----------------------------------------------------------------------
    # Question 5 - Likelihood evaluation
    # ----------------------------------------------------------------------

    # parameters:
    # space = np.linspace(-10, 10, 200)
    size = 200
    space = np.linspace(-10, 10, size)
    f3, f1 = np.meshgrid(space, space, sparse=True)

    res = np.vectorize(lambda x, y: multivariate_estimator.log_likelihood(
        np.array([x, 0, y, 0]).T, sigma, samples))(f1, f3)

    q5_fig = px.imshow(res,
                       labels=dict(x='f3 (3rd mu coordinate value)',
                                   y='f1 (1st mu coordinate value)',
                                   color='log likelihood scale'),
                       x=space, y=space)
    q5_fig.update_layout(title_text='Heatmap of log likelihood by changing '
                                    'mu values, from [-10, 0, -10, 0].T to ['
                                    '10, 0, 10, 0].T',
                         title_x=0.5)

    # What are you able to learn from the plot?
    #   - it looks like the samples behave exactly like multivariate gaussian
    #     distribution.

    q5_fig.show()

    # ----------------------------------------------------------------------

    # ----------------------------------------------------------------------
    # Question 6 - Maximum likelihood
    # ----------------------------------------------------------------------

    # parameters:
    accuracy = 4

    # get the maximum value from the likelihood calculation:
    f1_max_idx, f3_max_idx = np.unravel_index(np.argmax(res), res.shape)

    # print out the rounded results:
    print(f'Max log-likelihood values for:'
          f' f1={np.round(space[f1_max_idx], accuracy)},'
          f' f3={np.round(space[f3_max_idx], accuracy)}')


if __name__ == '__main__':
    np.random.seed(0)
    test_univariate_gaussian()
    test_multivariate_gaussian()
