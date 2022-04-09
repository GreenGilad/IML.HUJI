import IMLearn.learners.regressors.linear_regression
from IMLearn.learners.regressors import PolynomialFitting
from IMLearn.utils import split_train_test

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.io as pio

pio.templates.default = "simple_white"


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
    full_data = pd.read_csv(filename, parse_dates=['Date']).dropna().drop_duplicates()
    full_data["DayOfYear"] = full_data['Date'].dt.dayofyear
    full_data = full_data.drop(full_data.index[full_data["Temp"].astype(int) <= -70])
    full_data = full_data.drop(full_data.index[full_data["Day"].astype(int) > 31])
    full_data = full_data.drop(full_data.index[full_data["Day"].astype(int) <= 0])
    full_data = full_data.drop(full_data.index[full_data["Month"].astype(int) > 12])
    full_data = full_data.drop(full_data.index[full_data["Month"].astype(int) <= 0])
    return full_data


if __name__ == '__main__':
    np.random.seed(0)
    # Question 1 - Load and preprocessing of city temperature dataset
    design_matrix = load_data("../datasets/City_Temperature.csv")

    # Question 2 - Exploring data for specific country
    israel_df = design_matrix[design_matrix['Country'] == "Israel"]
    israel_df["Year"] = israel_df["Year"].astype(str)
    px.scatter(israel_df, x="DayOfYear", y="Temp", color="Year", title="Temp as function of day Of year",).show()

    israel_df = israel_df.groupby("Month").agg(Temp_std=('Temp', 'std'))
    israel_df = israel_df.reset_index()
    px.bar(israel_df, x="Month", y="Temp_std").show()

    # # Question 3 - Exploring differences between countries
    Q3_df = design_matrix.groupby(["Month", "Country"]).agg(Temp_std=('Temp', 'std'), Temp_Mean=('Temp', 'mean'))
    Q3_df = Q3_df.reset_index()
    px.line(Q3_df, x="Month", y="Temp_Mean", color='Country', error_y='Temp_std').show()
    # Question 4 - Fitting model for different values of `k`
    Q4_df = design_matrix[design_matrix['Country'] == "Israel"]
    response = Q4_df.pop('Temp')
    q4_train_X, q4_train_y, q4_test_X, q4_test_y = split_train_test(Q4_df, response)
    loss_per_k = []
    for k in range(1, 11):
        polyfit = PolynomialFitting(k)
        polyfit.fit(q4_train_X["DayOfYear"], q4_train_y)
        loss_per_k.append(round(polyfit.loss(q4_test_X["DayOfYear"], q4_test_y), 2))
    print(loss_per_k)
    px.bar(israel_df, x=range(1, 11), y=loss_per_k, labels={'x': 'K', 'y': 'Loss'},
           title='The loss value according to the degree').show()
    # Question 5 - Evaluating fitted model on different countries
    polyfit = PolynomialFitting(5)
    polyfit.fit(Q4_df["DayOfYear"], response)
    all_df = []
    countries = design_matrix['Country'].unique()
    countries = countries[countries != "Israel"]
    for country in countries:
        temp = design_matrix[design_matrix['Country'] == country]
        all_df.append(polyfit.loss(temp['DayOfYear'], temp.pop('Temp')))
    data = pd.DataFrame({"Country": countries, "Test Loss": all_df})
    px.bar(data, x="Country", y="Test Loss", title='Evaluating fitted model on different countries').show()
