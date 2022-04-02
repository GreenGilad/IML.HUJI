import IMLearn.learners.regressors.linear_regression
from IMLearn.learners.regressors import PolynomialFitting
from IMLearn.utils import split_train_test

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.io as pio
pio.templates.default = "simple_white"

def plot_mse_of_israeli_model_on_all_countries(data_frame: pd.DataFrame, data_frame_israel_only: pd.DataFrame):
    """
    Plots the MSE values for the israeli polynomial fit for all countries
    """
    # First perform the fist using the chosen rank of 6
    polynomial_fit = PolynomialFitting(6)
    polynomial_fit.fit(data_frame_israel_only['day_of_year'].to_numpy(), data_frame_israel_only['Temp'].to_numpy())

    # Get array of other countries
    unique_countries_array = data_frame['Country'].unique()
    number_of_countries = len(unique_countries_array)

    # The collected MSE values will be stored in this array
    mse_for_each_country = np.empty([1,number_of_countries])

    for i in range(number_of_countries):
        country_data = data_frame[data_frame.Country == unique_countries_array[i]]
        mse_for_each_country[0][i] =\
            polynomial_fit.loss(country_data['day_of_year'].to_numpy(), country_data['Temp'].to_numpy())

    fig = px.bar(title="MSE for Israeli polynomial model by country",
                 x=unique_countries_array,
                 y=mse_for_each_country[0],
                 labels= {
                     'x': 'Countries',
                     'y': 'MSE value for Israeli model'
                 })
    fig.show()

def plot_mse_for_increasing_deg_for_poly_fit(data_frame: pd.DataFrame):
    """
    Plots the mse for polynomial fit using vondermonde matrix with several ranks in the range
    1 to 10
    """
    # First split the input dataframe training data
    train_X, train_y, test_X, test_y = split_train_test(data_frame_israel_only, data_frame['Temp'], 0.75)

    # Setting up test data
    test_data = test_X['day_of_year'].to_numpy()
    test_values = test_y.to_numpy()

    # Setting up training data
    training_data = train_X['day_of_year'].to_numpy()
    training_values = train_y.to_numpy()

    # Collects the mse values and their corresponding ranks.
    # First row of this array is the rank of the vondermonde matrix and the second row
    # is the mse value
    mean_loss_for_each_k = np.empty([2, 10])

    for k in range(1, 11):
        # Perform fit and calculate loss
        polynomial_fit = PolynomialFitting(k)
        polynomial_fit.fit(training_data, training_values)
        loss_with_mse_function = polynomial_fit.loss(test_data, test_values)

        # Collect MSE and rank values
        mean_loss_for_each_k[0][k - 1] = k
        mean_loss_for_each_k[1][k - 1] = round(loss_with_mse_function, 2)
        print(f"The test error for {k} is {round(loss_with_mse_function, 2)}")

    fig = px.bar(mean_loss_for_each_k,
                 title="The MSE for each k in range 10",
                 x=mean_loss_for_each_k[0],
                 y=mean_loss_for_each_k[1],
                 labels={
                     'x': 'Value of k',
                     'y': 'Loss under MSE function'
                 })
    fig.show()


def plot_average_temp_in_israel(data_frame: pd.DataFrame):
    """
    Plots the average temp in Israel by day of the year
    """
    # extract the year column as a string which is later used to set the colour
    years_as_string_arr = data_frame['Year'].map(lambda t: str(t))

    fig = px.scatter(years_as_string_arr,title='Average temp by day of year',
                     x=data_frame['day_of_year'],
                     y= data_frame['Temp'],
                     color=years_as_string_arr,
                     labels={
                         'x': 'Day of year',
                         'y': 'average temp for day'
                     })
    fig.show()

def plot_std_by_month(data_frame: pd.DataFrame):
    """
    PLots standard deviation of daily temp by month in Israel
    """
    # Rearranging data frame to be grouped by month
    grouped_by_month = data_frame.groupby(['Month']).agg({'Temp': ['std']})
    grouped_by_month.columns = ['the_std']
    grouped_by_month.reset_index()

    fig = px.bar(grouped_by_month,
                 title='Standard deviation of average temp by month',
                 labels= {
                     'x' : 'Month',
                     'value': 'standard deviation of daily temp'
                 })
    fig.show()

def plot_average_temp_by_country(data_frame: pd. DataFrame):
    """
    Plots a line graph of average temp by month for each of the countries in the data_frame
    together with the standard deviation of temps for this month
    """
    # First group the data
    grouped_by_month = data_frame.groupby(['Country', 'Month']).agg({'Temp': ['mean','std']})
    grouped_by_month.columns = ['the_mean', 'the_std']
    grouped_by_month = grouped_by_month.reset_index()

    fig = px.line(grouped_by_month,
                  title='Average temp over month with standard deviation by country',
                  x='Month',
                  y='the_mean',
                  error_y='the_std',
                  line_group='Country',
                  color='Country'
                  )
    fig.show()

def load_data(filename: str) -> pd.DataFrame:
    """
    Load city daily temperature dataset and preprocess data.
    Parameters
    ----------
    filename: str
        Path to house prices dataset

    Returns
    -------
    Design matrix and response vector (Temp)
    """
    # Read the raw data
    data_frame = pd.read_csv(filename, parse_dates=[2])
    # Removing invalid temp values from the dataframe
    data_frame = data_frame[data_frame.Temp > -10]
    # Add day of the year column
    data_frame['day_of_year']  = data_frame['Date'].map(lambda t: t.timetuple().tm_yday)

    return data_frame


if __name__ == '__main__':
    np.random.seed(0)
    # Question 1 - Load and preprocessing of city temperature dataset
    data_frame = load_data('../datasets/City_Temperature.csv')
    data_frame_israel_only = data_frame[data_frame.Country == 'Israel']

    # Question 2 - Exploring data for specific country
    plot_average_temp_in_israel(data_frame_israel_only)
    plot_std_by_month(data_frame_israel_only)

    # Question 3 - Exploring differences between countries
    plot_average_temp_by_country(data_frame)

    # Question 4 - Fitting model for different values of `k`
    plot_mse_for_increasing_deg_for_poly_fit(data_frame_israel_only)

    # Question 5 - Evaluating fitted model on different countries
    plot_mse_of_israeli_model_on_all_countries(data_frame[data_frame.Country != 'Israel'],data_frame_israel_only)
