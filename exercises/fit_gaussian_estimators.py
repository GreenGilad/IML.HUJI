from IMLearn.learners import UnivariateGaussian, MultivariateGaussian
import numpy as np
import plotly.graph_objects as go
import plotly.io as pio

pio.templates.default = "simple_white"


def test_univariate_gaussian():
    # Question 1 - Draw samples and print fitted model
    mu = 10
    var = 1
    x = np.random.normal(mu, var, 1000)
    ug = UnivariateGaussian()
    ug.fit(x)
    print('(' + str(ug.mu_) + ", " + str(ug.var_) + ')')

    # Question 2 - Empirically showing sample mean is consistent
    estimated_mean = []

    for m in np.linspace(10, 1000, 100).astype(np.int_):
        temp_ug = UnivariateGaussian()
        temp_ug.fit(x[:m + 1])
        estimated_mean.append(np.abs(temp_ug.mu_ - mu))

    go.Figure(go.Scatter(x=np.linspace(10, 1000, 100).astype(np.int_), y=estimated_mean, mode='markers+lines',
                         name=r'Estimated expectation distance from expectation'),
              layout=go.Layout(
                  title=r'Absolute distance between the estimated and true value of the expectation, as a function '
                        r'of the sample size',
                  xaxis_title="$m\\text{ - number of samples}$",
                  yaxis_title=r"$|\hat{\mu}-\mu|$",
                  height=300,
                  width=1000)).show()

    # Question 3 - Plotting Empirical PDF of fitted model
    sorted_x = np.sort(x)
    go.Figure(go.Scatter(x=sorted_x, y=ug.pdf(sorted_x), mode='markers+lines'),
              layout=go.Layout(title='Empirical PDF function under the fitted model',
                               xaxis_title="Samples",
                               yaxis_title="PDF(x)",
                               height=300,
                               width=1000)).show()


def test_multivariate_gaussian():
    # Question 4 - Draw samples and print fitted model
    mu = np.array([0, 0, 4, 0])
    cov = np.array([[1, 0.2, 0, 0.5],
                    [0.2, 2, 0, 0],
                    [0, 0, 1, 0],
                    [0.5, 0, 0, 1]])
    x = np.random.multivariate_normal(mu, cov, 1000)
    mug = MultivariateGaussian()
    mug.fit(x)
    print(mug.mu_)
    print(mug.cov_)

    # Question 5 - Likelihood evaluation
    x_ = np.linspace(-10, 10, 200)
    y_ = np.linspace(-10, 10, 200)
    z = []
    for i in x_:
        temp = []
        for j in y_:
            temp.append(MultivariateGaussian().log_likelihood(np.array([i, 0, j, 0]), cov, x))
        z.append(temp)

    go.Figure(go.Heatmap(x=x_, y=y_, z=z),
              layout=go.Layout(
                  title=r"$\text{Heatmap for }\mu=\left[f_{1},0,f_{3},0\right]^{T}\text{ and given Cov matrix}$",
                  xaxis_title=r"$f_{3}$",
                  yaxis_title=r"$f_{1}$",
                  height=500,
                  width=500)).show()

    # Question 6 - Maximum likelihood
    result = np.where(z == np.amax(z))
    print(round(list(x_)[result[0][0]], 3))
    print(round(list(y_)[result[1][0]], 3))


if __name__ == '__main__':
    np.random.seed(0)
    test_univariate_gaussian()
    test_multivariate_gaussian()
