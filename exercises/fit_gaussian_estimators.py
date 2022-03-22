from IMLearn.learners import UnivariateGaussian, MultivariateGaussian
import numpy as np
import plotly.graph_objects as go
import plotly.io as pio

pio.templates.default = "simple_white"


def test_univariate_gaussian():
    # Question 1 - Draw samples and print fitted model
    # 1000 samples taken from ~ N(10, 1)
    X = np.random.normal(10, 1, 1000)
    univariate_gaussian = UnivariateGaussian().fit(X)
    print("Expectation: " + str(univariate_gaussian.mu_) + ", Variance: " + str(univariate_gaussian.var_))

    # Question 2 - Empirically showing sample mean is consistent
    univariate_gaussian = UnivariateGaussian()
    absolute_distance = []
    for sample_size in range(10, 1010, 10):
        univariate_gaussian.fit(X[:sample_size])
        absolute_distance.append(np.abs(univariate_gaussian.mu_ - 10))
    import plotly.express as px
    fig = px.line(x=range(10, 1010, 10),
                  y=np.asarray(absolute_distance),
                  labels={"x": "Sample Size", "y": "Distance of Estimated Expectation from True Value"},
                  title='Distance between the Estimated and True value of the Expectation')
    fig.show()

    # Question 3 - Plotting Empirical PDF of fitted model
    univariate_gaussian = UnivariateGaussian()
    univariate_gaussian.fit(X)
    import matplotlib.pyplot as plt
    from matplotlib import rcParams
    plt.scatter(x=X, y=univariate_gaussian.pdf(X), s=rcParams['lines.markersize'] ** 2 / 10)
    plt.xlabel("Sample Values")
    plt.ylabel("PDF of sample value")
    plt.title("Empirical PDF under Fitted Model")
    plt.show()


def test_multivariate_gaussian():
    # Question 4 - Draw samples and print fitted model
    MU = np.asarray([0, 0, 4, 0]).T
    COV = np.asarray([[1, 0.2, 0, 0.5],
                      [0.2, 2, 0, 0],
                      [0, 0, 1, 0],
                      [0.5, 0, 0, 1]])
    X = np.random.multivariate_normal(MU, COV, 1000)
    multi = MultivariateGaussian()
    multi.fit(X)
    print("Expectation: " + str(multi.mu_) + "\n Variance: \n" + str(multi.cov_))

    # Question 5 - Likelihood evaluation
    log_likelihood = [[MultivariateGaussian.log_likelihood(np.asarray([f1, 0, f3, 0]).T, COV, X)
                       for f3 in np.linspace(-10, 10, 200)]
                      for f1 in np.linspace(-10, 10, 200)]

    import plotly.express as px

    fig = px.imshow(log_likelihood,
                    x=np.linspace(-10, 10, 200),
                    labels=dict(x="f3 value", y="f1 value", color="log likelihood"),
                    y=np.linspace(-10, 10, 200),
                    title="Log-Likelihood using Different expectation vectors: [f1, 0, f3, 0]")

    fig.show()

    # Question 6 - Maximum likelihood
    max_idx = np.where(log_likelihood == np.amax(log_likelihood))
    f1 = np.linspace(-10, 10, 200)[max_idx[0]]
    f3 = np.linspace(-10, 10, 200)[max_idx[1]]

    print(f'f1: {f1[0]}, f3: {f3[0]}')


if __name__ == '__main__':
    np.random.seed(0)
    test_univariate_gaussian()
    test_multivariate_gaussian()


