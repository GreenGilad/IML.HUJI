import math
import datetime
import os

from IMLearn.utils import split_train_test
from IMLearn.learners.regressors import LinearRegression

from typing import NoReturn
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio

pio.templates.default = "simple_white"


def adding_dist_from_central_seattle(X: pd.DataFrame):
    """
    Adds a distance from seattle column
    """
    # lat and long coordinates of central seattle
    seattle_lat, seattle_long = 47.6062, -122.3321

    # adding the new column
    X['lat'] = X['lat'].map(lambda t: ((seattle_lat - t) ** 2))
    X['long'] = X['long'].map(lambda t: ((seattle_long - t) ** 2))
    X['dist_from_seattle'] = (X['long'] + X['lat']).map(lambda t: math.sqrt(t))


def convert_date_to_numeric(date: str):
    """
    Converts a date to a numerical value of year + (month/12) that can
    be used for fitting a linear model
    """
    # Valid data inputs will allow us to use the following format
    year = int(date[:4])
    month = int(date[4:6])
    return year + (month / 12)


def load_data(filename: str):
    """
    Load house prices dataset and preprocess data.
    Parameters
    ----------
    filename: str
        Path to house prices dataset

    Returns
    -------
    Design matrix and response vector (prices) - either as a single
    DataFrame or a Tuple[DataFrame, Series]
    """

    # First load the raw csv table
    df = pd.read_csv(filename)

    # Removing empty or all zero rows
    df.dropna(inplace=True)
    df = df[df.id != 0]  # valid id's cannot be zero

    # Adding distance from central seattle column
    adding_dist_from_central_seattle(df)

    # One-hot encoding zip codes
    df = pd.concat([df, pd.get_dummies(df['zipcode'])], axis=1)

    # Converting date string to a numeric value
    df['date'] = df['date'].map(lambda t: convert_date_to_numeric(t))

    # Removing unnecessary columns
    df.drop(['id', 'long', 'lat', 'zipcode', 'yr_renovated'], axis=1, inplace=True)

    return df


def plot_correlation_graph(feature_vec: np.ndarray, prices_vec: np.ndarray, feature_name: str, output_folder_name: str):
    """
    Plots scatter graph with input feature values on the x-axis and house price on the y-axis
    and outputs these graphs to the "plots_output_folder"
    """
    if feature_name == 'price':  # we don't want to calculate the correlation for the prices column
        return

    pc = (np.cov(feature_vec, prices_vec)[0][1]) / (feature_vec.std() * prices_vec.std())  # compute the PC value
    x_axis_name = feature_name if str(feature_name)[0] != '9' else f"zip_code {feature_name}"

    fig = px.scatter(x=feature_vec,
                     y=prices_vec,
                     labels={
                         'x': f'{x_axis_name}',
                         'y': 'prices'
                     },
                     title=f"Correlation between {x_axis_name} and price with PC value {round(pc, 6)}")
    fig.write_image(f"{output_folder_name}/{x_axis_name}.png")


def plot_mse_as_func_of_sample_size(train_X: pd.DataFrame, test_X: pd.DataFrame, test_y: pd.Series):
    """
    Function plots the mse as a function of the percentage taken from a given input sample. For each percentage
    point we perform 10 estimations and check their mean
    """
    # The mean loss vector matrix. with the rows:
    # - 0: The % of the sample used
    # - 1: The MSE for the given iteration
    # - 2: The standard deviation of the input for given iteration
    mean_loss_for_each_p = np.empty([3, 91])

    # Setting up test data
    p_test_data = test_X.drop(['price'], axis=1).to_numpy()
    p_test_values = test_y.to_numpy()

    # Performing iterations on increasing sample size
    for p in range(10, 101):
        mse_array = np.empty([1, 10])
        for j in range(10):
            # Set up training data
            p_training_sample = train_X.sample(frac=(p / 100))
            p_sample_data = p_training_sample.drop(['price'], axis=1).to_numpy()
            p_sample_values = p_training_sample['price'].to_numpy()

            # perform fit
            linear_regression_model.fit(p_sample_data, p_sample_values)
            mse_array[0][j] = linear_regression_model.loss(p_test_data, p_test_values)

        # Collecting results
        mean_loss_for_each_p[0][p - 10] = p
        mean_loss_for_each_p[1][p - 10] = np.mean(mse_array)
        mean_loss_for_each_p[2][p - 10] = mse_array.std() * 2

    fig = px.scatter(title="MSE as a function of sample size",
                     x=mean_loss_for_each_p[0],
                     y=mean_loss_for_each_p[1],
                     error_y=mean_loss_for_each_p[2],
                     labels={
                         'x': 'percentage of overall sample used',
                         'y': 'mean MSE over 10 iterations'
                     })
    fig.show()


def feature_evaluation(X: pd.DataFrame, y: pd.Series, output_path: str = ".") -> NoReturn:
    """
    Create scatter plot between each feature and the response.
        - Plot title specifies feature name
        - Plot title specifies Pearson Correlation between feature and response
        - Plot saved under given folder with file name including feature name
    Parameters
    ----------
    X : DataFrame of shape (n_samples, n_features)
        Design matrix of regression problem

    y : array-like of shape (n_samples, )
        Response vector to evaluate against

    output_path: str (default ".")
        Path to folder in which plots are saved
    """

    X.apply(lambda t: plot_correlation_graph(t.to_numpy(), y.to_numpy(), t.name, output_path), axis=0)


if __name__ == '__main__':
    np.random.seed(0)
    # Question 1 - Load and preprocessing of housing prices dataset
    data_frame = load_data('../datasets/house_prices.csv')

    # Question 2 - Feature evaluation with respect to response

    # First create the output folder for the features plots
    if not os.path.exists('plots_output_folder'):
        os.mkdir('plots_output_folder')

    # Performing the features evaluation on the processed data
    feature_evaluation(data_frame, data_frame['price'], 'plots_output_folder')

    # Question 3 - Split samples into training- and testing sets.
    train_X, train_y, test_X, test_y = split_train_test(data_frame, data_frame['price'], 0.75)

    # Question 4 - Fit model over increasing percentages of the overall training data
    # For every percentage p in 10%, 11%, ..., 100%, repeat the following 10 times:
    #   1) Sample p% of the overall training data
    #   2) Fit linear model (including intercept) over sampled set
    #   3) Test fitted model over test set
    #   4) Store average and variance of loss over test set
    # Then plot average loss as function of training size with error ribbon of size (mean-2*std, mean+2*std)

    linear_regression_model = LinearRegression()
    # Plotting the MSE as a function of sample size
    plot_mse_as_func_of_sample_size(train_X, test_X, test_y)
