import IMLearn.learners.regressors.linear_regression
from IMLearn.learners.regressors import PolynomialFitting
from IMLearn.utils import split_train_test

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.io as pio
import datetime


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
    city_tempdf = pd.read_csv(filename, parse_dates=['Date'])
    shapeofdata = city_tempdf.shape

    city_tempdf['Country-Code'] = [i for i in range(shapeofdata[0])]
    country_list = list(city_tempdf['Country'].unique())
    result = []

    def change_categorical_val(val, category_list):
        if val in category_list:
            result.append(category_list.index(val))

    city_tempdf['Country'].apply(lambda val: change_categorical_val(val, country_list))
    city_tempdf['Country-Code'] = result
    city_tempdf.drop(["City"], axis=1, inplace=True)

    # remove invalid temperature samples
    drop = city_tempdf[city_tempdf['Temp'] < -72.0].index.tolist()
    city_tempdf = city_tempdf.drop(index=drop).reset_index(drop=True)

    # add "dayOfYear" column
    days = []
    city_tempdf['Date'].apply(lambda x: days.append(x.timetuple().tm_yday))
    city_tempdf['dayOfYear'] = days
    city_tempdf.drop(["Date"], axis=1, inplace=True)

    return city_tempdf


"""
NOTE: the figure is not shown if trying to run it regularly, But the printing
at the end of the function is executed. The figure is shown while running in debug
"""
if __name__ == '__main__':
    np.random.seed(0)
    # Question 1 - Load and preprocessing of city temperature dataset
    city_temp_data = load_data("City_Temperature.csv")

    # Question 2 - Exploring data for specific country
    city_temp_data_israel = city_temp_data.loc[city_temp_data["Country-Code"] == 2, :]
    city_temp_data_israel['Year'] = city_temp_data_israel['Year'].apply(str)
    fig = px.scatter(city_temp_data_israel, x="dayOfYear", y="Temp", color="Year")
    fig.update_layout(
        title="Average daily temperatures for each day of the year",
        xaxis_title="Day of the year",
        yaxis_title="Mean temperature")
    fig.show()

    std_months = city_temp_data_israel.groupby('Month').Temp.agg('std')
    uniqe_months = country_list = list(city_temp_data_israel['Month'].unique())
    fig = px.bar(city_temp_data_israel, x=uniqe_months, y=std_months)
    fig.update_layout(
        title="Standard deviation of the daily temperatures per month",
        xaxis_title="Month",
        yaxis_title="Std")
    fig.show()

    # Question 3 - Exploring differences between countries
    month_and_country = city_temp_data.groupby(["Month", "Country"]).Temp.agg(["mean", "std"])
    months = city_temp_data.groupby('Month')
    mean_months = months.Temp.mean()
    std_months = months.Temp.agg('std')
    fig = px.line(month_and_country,
                  x=month_and_country.index.get_level_values("Month"), y="mean", error_y="std",
                  color=month_and_country.index.get_level_values("Country"))
    fig.show()

    # Question 4 - Fitting model for different values of `k`
    temp_data_israel = city_temp_data.loc[city_temp_data["Country-Code"] == 2, :]
    poly_sample = temp_data_israel["dayOfYear"]
    poly_response = temp_data_israel["Temp"]
    train_poly_x, train_poly_y, test_poly_x, test_poly_y = split_train_test(poly_sample.to_frame(), poly_response)
    errors = []
    for k in range(1, 11):
        poly_model = PolynomialFitting(k)
        poly_model.fit(np.array(train_poly_x), np.array(train_poly_y))
        loss = round(poly_model.loss(np.array(test_poly_x), np.array(test_poly_y)), 2)
        errors.append(loss)
        print("Error for {deg} degree model: {loss}".format(deg=k, loss=loss))

    fig = px.bar(x=[i for i in range(1, 11)], y=errors)
    fig.update_layout(
        title="Error according to degree of the polynomial model",
        xaxis_title="degree",
        yaxis_title="mean loss")
    fig.show()

    # Question 5 - Evaluating fitted model on different countries
    poly_model = PolynomialFitting(5)
    poly_model.fit(np.array(poly_sample), np.array(poly_response))
    countries_errors = []
    for country in ["South Africa", "Jordan", "The Netherlands"]:
        data = city_temp_data.loc[city_temp_data["Country"] == country, :]
        sample = data["dayOfYear"]
        response = data["Temp"]
        loss = round(poly_model.loss(np.array(sample), np.array(response)), 2)
        countries_errors.append(loss)

    fig = px.bar(x=["South Africa", "Jordan", "The Netherlands"], y=countries_errors)
    fig.update_layout(
        title="5-degree model's error over each of the tree counties",
        xaxis_title="Country",
        yaxis_title="mean loss")
    fig.show()

    # prints it but don't show the figure. Although it is, if running in debug
    print("finish execution!!")
