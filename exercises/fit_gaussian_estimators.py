import numpy

from IMLearn.learners import UnivariateGaussian, MultivariateGaussian
import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
import plotly.express as px

pio.templates.default = "simple_white"


def plot_pdf(X: np.ndarray, univariate_gaussian: UnivariateGaussian):
    """
    Plots the pdf
    """
    fig = px.scatter(x=X,
                     y=univariate_gaussian.pdf(X),
                     labels={
                         'x': 'Sample Value',
                         'y': 'Calculated values of given samples for PDF function'},
                     title=f"Calculated values of given samples for PDF function of N({univariate_gaussian.mu_}, {univariate_gaussian.var_})")
    fig.show()


def plot_consistency_of_expectation(sample_size, dis_from_true_exp_list):
    """
    Plots graph of the relation between sample size and the accuracy of the sample's expected
    value relative to the true expected value for the total sample data
    """
    fig = px.scatter(x=sample_size,
                     y=dis_from_true_exp_list,
                     labels={
                         'x': 'Sample size',
                         'y': 'Deviation of expected value from true expected value'},
                     title="Accuracy of expected value of sample data relative to sample size")
    fig.show()


def find_distance(X: np.ndarray, true_expectation: float, dis_from_true_exp_list: [float]):
    """
    Finds absolute distance between the expected value of a given slice of the input data
    and the true expected value of the full sample data
    """
    univar_gaus = UnivariateGaussian()
    univar_gaus.fit(X)
    dis_from_true_exp_list.append(abs(true_expectation - univar_gaus.mu_))


# This function should return the
def get_max_val_and_indexes(results_matrix, f1_sample_values, f3_sample_values):
    x = y = len(results_matrix)
    x_coord_of_max = 0
    y_coord_of_max = 0
    curr_max = results_matrix[0][0]
    for i in range(x):
        for j in range(y):
            if results_matrix[i][j] > curr_max:
                curr_max = results_matrix[i][j]
                x_coord_of_max = i
                y_coord_of_max = j
    return (curr_max, f1_sample_values[x_coord_of_max], f3_sample_values[y_coord_of_max])



def calculate_log_likelihood_for_miltivariate(
        multivar_gaussian: MultivariateGaussian,
        cov_matrix,
        f1_values,
        f3_values,
        sample_data_array):
    data_matrix = []
    for i in range(len(f1_values)):
        data_matrix_row = []
        for j in range(len(f3_values)):
            data_matrix_row.append(multivar_gaussian.log_likelihood(
                np.array([f1_values[i], 0, f3_values[j], 0]),
                cov_matrix,
                sample_data_array))
        data_matrix.append(data_matrix_row)
        print(f"done with row {i}")
    return data_matrix


def find_consistency_of_expectation(univariant_guassian: UnivariateGaussian, X: np.ndarray):
    """
    Plots the distance between the expected value of an increasing sample size from the
    true expected value of given sample data.
    """
    dist_from_true_exp_list = []
    sample_size_list = []
    for i in range(10, 1010, 10):
        sample_size_list.append(i)
        find_distance(X[:i], univariant_guassian.mu_, dist_from_true_exp_list)

    plot_consistency_of_expectation(sample_size_list, dist_from_true_exp_list)


def test_univariate_gaussian():
    """
    Calls fitting and graphing helper methods
    """
    # Question 1 - Draw samples and print fitted model

    # First draw samples ~N(10,1)
    mu, sigma = 10, 1
    samples_array = np.random.normal(mu, sigma, 1000)

    # Create univariant guassian model and estimate expectation and variance for given samples
    univariant_guassian = UnivariateGaussian()
    univariant_guassian.fit(samples_array)

    # print the estimated expectation and variance
    print(f"({univariant_guassian.mu_} , {univariant_guassian.var_})")

    # Question 2 - Empirically showing sample mean is consistent
    find_consistency_of_expectation(univariant_guassian, samples_array)

    # Question 3 - Plotting Empirical PDF of fitted model
    plot_pdf(samples_array, univariant_guassian)


def test_multivariate_gaussian():
    # Question 4 - Draw samples and print fitted model

    # First we draw the samples for given expectation and variance
    mu = np.array([0, 0, 4, 0])
    sigma = np.array([[1, 0.2, 0, 0.5], [0.2, 2, 0, 0], [0, 0, 1, 0], [0.5, 0, 0, 1]])

    samples_array = np.random.multivariate_normal(mu, sigma, size=1000)
    multivariant_gaussian = MultivariateGaussian()
    multivariant_gaussian.fit(samples_array)

    # Question 5 - Likelihood evaluation
    f1_values = np.linspace(-10, 10, 200)
    f3_values = np.linspace(-10, 10, 200)

    data_array = calculate_log_likelihood_for_miltivariate(
        multivariant_gaussian,
        sigma,
        f1_values,
        f3_values,
        samples_array)

    fig = px.imshow(data_array,
                    title="Heatmap of log-likelihood for normal multivariate distribution",
                    labels=dict(
                        x="f1 values",
                        y="f3 values",
                        color="log-likelihood"),
                    x=f1_values,
                    y=f3_values)
    fig.show()

    # Question 6 - Maximum likelihood
    results = get_max_val_and_indexes(data_array, f1_values, f3_values)
    print(f"The max value is: {results[0]} for f3 value {results[1]} and f1 value {results[2]}")


if __name__ == '__main__':
    np.random.seed(0)
    # test_univariate_gaussian()
    test_multivariate_gaussian()

