from IMLearn.learners import UnivariateGaussian, MultivariateGaussian
import numpy as np
import plotly.graph_objects as go
import plotly.io as pio

pio.renderers.default = 'browser'
pio.templates.default = "simple_white"


def test_univariate_gaussian():
    # Question 1 - Draw samples and print fitted model
    question_1_samples = np.random.normal(10, 1, size=1000)
    question_1_model = UnivariateGaussian()
    question_1_model.fit(question_1_samples)
    print(question_1_model.mu_, question_1_model.var_)

    # Question 2 - Empirically showing sample mean is consistent
    question_2_x_values = []
    mean_values = []

    ug = UnivariateGaussian()
    for i in range(10, 1001, 10):
        question_2_x_values.append(i)
        ug.fit(question_1_samples[1:i])
        mean_values.append(np.abs(10-ug.mu_))

    question_2_fig = go.Figure()
    question_2_fig.update_layout(
        title="Samples from ~N(10,1) and their difference from the true mean",
        xaxis_title="Sample size",
        yaxis_title="Difference from mean"
    )

    question_2_fig.add_scatter(
        x=question_2_x_values,
        y=mean_values,
    )
    question_2_fig.show()

    # Question 3 - Plotting Empirical PDF of fitted model
    question_3_fig = go.Figure()
    question_3_fig.update_layout(
        title="Samples from ~N(10,1) and their PDFs",
        xaxis_title="Sample value",
        yaxis_title="PDF of sample"
    )

    qs_3_samples = np.sort(question_1_samples)
    question_3_fig.add_scatter(
        x=qs_3_samples,
        y=question_1_model.pdf(qs_3_samples),
    )

    """The expected result is a graph that looks like a bell,
    as the samples were taken from normal distribution. 
    """
    question_3_fig.show()



def test_multivariate_gaussian():
    # Question 4 - Draw samples and print fitted model
    mean = np.array([0, 0, 4, 0])
    cov = np.array(
        [
            [1, 0.2, 0, 0.5],
            [0.2, 2, 0, 0],
            [0, 0, 1, 0],
            [0.5, 0, 0, 1]
        ])

    question_1_samples = \
        np.random.multivariate_normal(mean, cov, size=1000)

    question_1_model = MultivariateGaussian()
    question_1_model.fit(question_1_samples)
    print("Mean:\n", question_1_model.mu_)
    print("\n")
    print("Cov:\n", question_1_model.cov_)

    # Question 5 - Likelihood evaluation
    space = np.linspace(-10, 10, 200)
    heat = np.zeros((space.size, space.size))

    for i, f1 in enumerate(space):
        for j, f3 in enumerate(space):
            heat[i, j] = MultivariateGaussian.log_likelihood(np.array([f1, 0, f3, 0]), cov, question_1_samples)

    fig = go.Figure(go.Heatmap(x=space, y=space, z=heat), layout=go.Layout(title="Log-likelihood for [f1,0,f3,0]"))
    fig.show()
    # Question 6 - Maximum likelihood]
    indices = np.unravel_index(heat.argmax(),heat.shape)
    max_log_likelihood = np.round(heat[indices[0],indices[1]],3)

    max_f1 = np.round(space[indices[0]],3)
    max_f3 = np.round(space[indices[1]],3)
    print("The (f1,f3) values with the max log-likelihood of", max_log_likelihood)
    print("are: ",max_f1, "and", max_f3)


if __name__ == '__main__':
    np.random.seed(0)
    test_univariate_gaussian()
    test_multivariate_gaussian()
