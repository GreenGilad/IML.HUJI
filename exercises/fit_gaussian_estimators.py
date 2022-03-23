#################################################################
# FILE : fit_gaussian_estimators.py
# WRITER : Bar Melinarskiy
# EXERCISE : Intro to Machine Learning - 67577 - Exercise 1
# DESCRIPTION: Testing UnivariateGaussian and MultivariateGaussian classes.
#################################################################

from IMLearn.learners import UnivariateGaussian, MultivariateGaussian
import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
import matplotlib.pyplot as plt
pio.templates.default = "simple_white"


def test_univariate_gaussian():
    # Question 1 - Draw samples and print fitted model
    mu, sigma = 10, 1  # mean and standard deviation
    samples = np.random.normal(mu, sigma, 1000)
    g = UnivariateGaussian()
    g.fit(samples)
    print(" Question 1 - Draw samples and print fitted model:")
    print("(" + str(g.mu_) + ", " + str(g.var_) + ")")

    # Question 2 - Empirically showing sample mean is consistent
    data_q2 = np.empty((2, 100))  # Data array for plot Q2
    i = 0
    for sample_size in np.arange(10, 1010, 10):
        g.fit(samples[0:sample_size])
        # absolute distance between the estimated- and true value of the expectation
        abs_diff = np.abs(mu - g.mu_)
        data_q2[0, i] = sample_size
        data_q2[1, i] = abs_diff
        i += 1

    # Create scatter plot with our data
    plt.xlabel('Sample Size')
    plt.ylabel('Abs Expectation Distance')
    mu_text = "|" + '\u03BC' + " - " + '\u03BC' + "{}|".format('\N{LATIN SUBSCRIPT SMALL LETTER X}')
    q2_title = "Scatterplot of " + r'$\Delta$' + mu_text + " Vs. sample size"
    plt.title(q2_title)
    plt.scatter(data_q2[0, :], data_q2[1, :], label="Scatter")
    # Add trend Line to the plot
    z = np.polyfit(data_q2[0, :], data_q2[1, :], 10)
    p = np.poly1d(z)
    plt.plot(data_q2[0, :], p(data_q2[0, :]), "r--", label="Scatter trend line")
    plt.legend(loc="upper right")
    plt.tight_layout()
    # plt.savefig('part1_q2.png')  # Save the plot for insta story
    plt.show()

    # Question 3 - Plotting Empirical PDF of fitted model
    data_q3 = {
        "samples": samples,
        "pdf": g.pdf(samples)
    }
    plt.xlabel('Sample Value')
    plt.ylabel('PDF Value')
    q3_title = "Scatterplot of PDF Value Vs. Sample Value"
    plt.title(q3_title)
    plt.scatter(data_q3["samples"], data_q3["pdf"], label="Scatter")
    plt.legend(loc="upper right")
    plt.tight_layout()
    # plt.savefig('part1_q3.png')  # Save the plot for insta story
    plt.show()


def test_multivariate_gaussian():
    # Question 4 - Draw samples and print fitted model
    # mean and covariance
    mu = np.array([0, 0, 4, 0])
    cov = np.array([[1, 0.2, 0, 0.5], [0.2, 2, 0, 0], [0, 0, 1, 0], [0.5, 0, 0, 1]])
    samples = np.random.multivariate_normal(mu, cov, 1000)
    m = MultivariateGaussian()
    m.fit(samples)
    print(" Question 4 - Draw samples and print fitted model:")
    print("Estimated expectation: \n" + str(m.mu_))
    print("Estimated covariance: \n" + str(m.cov_))

    # Question 5 - Likelihood evaluation
    f_arr = np.linspace(-10, 10, 200)
    data_q5 = []
    for f1 in f_arr:
        for f3 in f_arr:
            new_mu = np.array([f1, 0, f3, 0]).T
            log_likelihood = MultivariateGaussian.log_likelihood(new_mu, cov, samples)
            data_q5.append(log_likelihood)

    data_q5 = np.array(data_q5).reshape(200, 200)
    q5_title = 'Q5: Heatmap of log_likelihood Vs. values in mu = [f1, 0, f3, 0]'
    fig = go.Figure(go.Heatmap(x=f_arr, y=f_arr, z=data_q5, colorbar=dict(title='Log Likelihood')),
                    layout=go.Layout(
                        title=q5_title,
                        xaxis=dict(title="f3 values"),
                        yaxis=dict(title="f1 values")
                    ))
    fig.show()

    # Question 6 - Maximum likelihood
    max_index = np.argmax(data_q5)
    i = int(max_index / 200)
    j = max_index % 200
    form = "{:.3f}"
    print(" Question 6 - Maximum likelihood:")
    print("The (f1, f3) pair with the max log_likelihood of: " + form.format(data_q5.max()))
    print("Is: (" + form.format(f_arr[i]) + ", " + form.format(f_arr[j]) + ")")


if __name__ == '__main__':
    np.random.seed(0)
    test_univariate_gaussian()
    test_multivariate_gaussian()
