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
    plt.savefig('part1_q2.png')  # Save the plot for insta story
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
    plt.savefig('part1_q3.png')  # Save the plot for insta story
    plt.show()


def test_multivariate_gaussian():
    # Question 4 - Draw samples and print fitted model
    # mean and covariance
    mu = [0, 0, 4, 0]
    cov = [[1, 0.2, 0, 0.5], [0.2, 2, 0, 0], [0, 0, 1, 0], [0.5, 0, 0, 1]]
    samples = np.random.multivariate_normal(mu, cov, 1000)
    m = MultivariateGaussian()
    m.fit(samples)
    print("Estimated expectation: \n" + str(m.mu_))
    print("Estimated covariance: \n" + str(m.cov_))

    # Question 5 - Likelihood evaluation
    f_arr = np.linspace(-10, 10, 200)
    data_q5 = {
        'X': np.empty(200 * 200),
        'Y': np.empty(200 * 200),
        'Z': np.empty(200 * 200)
    }
    i = 0
    for f1 in f_arr:
        for f3 in f_arr:
            print(i)
            new_mu = [f1, 0, f3, 0]
            log_likelihood = MultivariateGaussian.log_likelihood(new_mu, cov, samples)
            data_q5['X'][i] = f1
            data_q5['Y'][i] = f3
            data_q5['Z'][i] = log_likelihood
            i += 1

    q5_title = 'Q5: Heatmap of log_likelihood'
    fig = go.Figure(go.Heatmap(x=data_q5['X'], y=data_q5['Y'], z=data_q5['Z']),
              layout=go.Layout(title=q5_title))
    fig.update_xaxes(title_text="f1 values")
    fig.update_yaxes(title_text="f3 values")
    fig.show()

    # Question 6 - Maximum likelihood

    max_index = np.argmax(data_q5['Z'])
    form = "{:.3f}"
    print("The (f1, f3) pair with the max log_likelihood of: " + form.format(data_q5['Z'][max_index]))
    print("Is: (" + form.format(data_q5['X'][max_index]) + ", " + form.format(data_q5['Y'][max_index]) + ")")


if __name__ == '__main__':
    np.random.seed(0)
    test_univariate_gaussian()
    test_multivariate_gaussian()
