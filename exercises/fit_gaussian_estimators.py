from IMLearn.learners import UnivariateGaussian, MultivariateGaussian
import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
pio.templates.default = "simple_white"


def test_univariate_gaussian():
    # Question 1 - Draw samples and print fitted model
    samples = np.random.normal(10, 1, 1000)
    my_estimator = UnivariateGaussian()
    my_estimator.fit(samples)
    print("\nQuestion 1:\n" + "(" + str(my_estimator.mu_) + ", " + str(my_estimator.var_) + ")\n")

    # Question 2 - Empirically showing sample mean is consistent
    samples_num = [i for i in range(10, 1010, 10)]
    estimated_mu = []
    for i in range(100):
        my_estimator.fit(samples[:samples_num[i]])
        estimated_mu.append(abs(my_estimator.mu_ - 10))

    go.Figure([go.Scatter(x=samples_num, y=estimated_mu, mode='markers+lines')],
              layout=go.Layout(
                  title=r"$\text{Estimation of the expectation as function of samples number}$",
                  xaxis_title="$m\\text{ samples number }$",
                  yaxis_title="r\\text{ the absolute distance between the estimated"
                              "- and true value of the expectation}$",
                  height=600)).show()

    # Question 3 - Plotting Empirical PDF of fitted model
    pdf_result = my_estimator.pdf(samples)
    go.Figure([go.Scatter(x=samples, y=pdf_result, mode='markers',
                          line=dict(width=4, color="rgb(204,68,83)"))],
              layout=go.Layout(title=r"$\text{The empirical PDF function under the fitted model}$",
                               xaxis_title="$\\text{ samples values }$",
                               yaxis_title="r\\text{ PDF values }$",
                               height=500)).show()


def test_multivariate_gaussian():
    # Question 4 - Draw samples and print fitted model
    mu = [0, 0, 4, 0]
    cov = [[1, 0.2, 0, 0.5], [0.2, 2, 0, 0], [0, 0, 1, 0], [0.5, 0, 0, 1]]
    samples_multi_var = np.random.multivariate_normal(mu, cov, 1000)
    my_estimator_multi_var = MultivariateGaussian()
    my_estimator_multi_var.fit(samples_multi_var)
    print("Question 4:\n")
    print(my_estimator_multi_var.mu_)
    print(my_estimator_multi_var.cov_)
    print("\n")

    # Question 5 - Likelihood evaluation
    f_values = np.linspace(-10, 10, 200)
    likelihood_values = np.zeros((200, 200))

    for x_axis in range(200):
        for y_axis in range(200):
            likelihood_values[x_axis][y_axis] = \
                MultivariateGaussian.log_likelihood(
                np.asarray([f_values[x_axis], 0, f_values[y_axis], 0]),
                    cov, samples_multi_var)
    go.Figure(go.Heatmap(x=f_values, y=f_values, z=likelihood_values),
              layout=go.Layout(title="Likelihood Evaluation by f1 & f3", height=600, width=800,
                               xaxis_title="$\\text{f3 values }$",
                               yaxis_title="r\\text{f1 values}$")).show()

    # Question 6 - Maximum likelihood
    f1_max, f3_max = np.unravel_index(np.argmax(likelihood_values, axis=None), likelihood_values.shape)
    print("Question 6:\n" + "the maximum log-likelihood value gets from the next pair of (f1,f3):")
    print(" (" + str(round(f_values[f1_max], 4)) + ", " + str(round(f_values[f3_max], 4)) + ")\n")

if __name__ == '__main__':
    np.random.seed(0)
    test_univariate_gaussian()
    test_multivariate_gaussian()

    print("question 3 in the quiz:")
    samp_quiz = np.array([1, 5, 2, 3, 8, -4, -2, 5, 1, 10, -10, 4, 5, 2, 7, 1, 1, 3, 2, -1, -3, 1, -4, 1, 2, 1,
              -4, -4, 1, 3, 2, 6, -6, 8, 3, -6, 4, 1, -2, 3, 1, 4, 1, 4, -2, 3, -1, 0, 3, 5, 0, -2])
    print(UnivariateGaussian.log_likelihood(1, 1, samp_quiz))
    print(UnivariateGaussian.log_likelihood(10, 1, samp_quiz))
