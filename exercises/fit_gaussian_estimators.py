from IMLearn.learners import UnivariateGaussian, MultivariateGaussian
import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
import plotly.express as px
import matplotlib.pyplot as plt
pio.templates.default = "simple_white"


def test_univariate_gaussian():
    # Question 1 - Draw samples and print fitted model
    mu, sigma = 10, 1  # mean and standard deviation
    samples = np.random.normal(mu, sigma, 1000)
    g = UnivariateGaussian()
    g.fit(samples)
    print("(" + str(g.mu_) + "," + str(g.var_) + ")")
    print("PDF:" + str(g.pdf(samples)))

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
    plt.savefig('part1_q2.png')  # Save the plot for insta story
    plt.show()

    # fig = px.line(data_q2, x="sample_size", y="abs_diff", title=q2_title)
    # fig.show()

    # Question 3 - Plotting Empirical PDF of fitted model
    # raise NotImplementedError()


def test_multivariate_gaussian():
    # Question 4 - Draw samples and print fitted model
    raise NotImplementedError()

    # Question 5 - Likelihood evaluation
    raise NotImplementedError()

    # Question 6 - Maximum likelihood
    raise NotImplementedError()


if __name__ == '__main__':
    np.random.seed(0)
    test_univariate_gaussian()
    # test_multivariate_gaussian()
