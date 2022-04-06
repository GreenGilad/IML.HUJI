import IMLearn.learners.regressors.linear_regression
from IMLearn.learners.regressors import PolynomialFitting
from IMLearn.utils import split_train_test

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.io as pio
from IMLearn.learners.regressors import linear_regression
import plotly.graph_objects as go
pio.templates.default = "simple_white"
from datetime import datetime, date


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
    df = pd.read_csv(filename).dropna().drop_duplicates()
    df = df[df["Day"] <= 31]
    df = df[df["Month"] <= 12]
    df = df[df["Temp"] > -60]


    df['Date'] = pd.to_datetime(df['Date'])
    df['DayOfYear'] = df['Date'].dt.dayofyear
    df = df[pd.DatetimeIndex(df['Date']).day == df['Day']]
    df = df[pd.DatetimeIndex(df['Date']).month == df['Month']]
    df = df[pd.DatetimeIndex(df['Date']).year == df['Year']]

    df = pd.get_dummies(df, prefix='City_', columns=['City'])
    return df


if __name__ == '__main__':
    np.random.seed(0)
    # Question 1 - Load and preprocessing of city temperature dataset
    X = load_data(r"..\datasets\City_Temperature.csv")

    # Question 2 - Exploring data for specific country

    # part 1
    X_ISR = X[X["Country"] == "Israel"]

    X_ISR.insert(0, "YearAsString", X_ISR["Year"].astype(str))
    fig = px.scatter(X_ISR, x="DayOfYear", y='Temp', color="YearAsString",
                     title="temp as function of day of year divided by year").show()
    X_ISR = X_ISR.drop("Year", axis=1)

    # part 2
    X_ISR = X_ISR.groupby('Month')['Temp'].agg('std').reset_index()
    px.bar(X_ISR, x="Month", y="Temp",
           title="temperature std as function of month").show()


    # Question 3 - Exploring differences between countries

    X_by_std = X.groupby(['Country', 'Month'])['Temp'].std().reset_index()
    X_by_mean = X.groupby(['Country', 'Month'])['Temp'].mean().reset_index()

    fig = px.line(X_by_mean, x="Month", y="Temp", color='Country', line_group='Country',
                  error_y=X_by_std['Temp'],
                  title="temp mean as function of month divided by country").show()

    # Question 4 - Fitting model for different values of `k`
    X_4 = X[X["Country"] == "Israel"]
    train_x, train_y, test_x, test_y = split_train_test(X_4.drop("Temp", axis=1), X_4["Temp"])
    p_r_loss = []
    print("Question 4:")
    for k in range(1, 11):
        p_r = PolynomialFitting(k)
        p_r.fit(train_x["DayOfYear"].to_numpy(), train_y.to_numpy())
        loss = round(p_r.loss(test_x["DayOfYear"].to_numpy(), test_y.to_numpy()), 2)
        p_r_loss.append(loss)
        print("for k = " + str(k) + " the loss is: " + str(loss))

    pd_loss = pd.DataFrame(p_r_loss).rename(columns={0: "loss"})
    pd_loss.index = range(1, 11)
    fig = px.bar(pd_loss, x=pd_loss.index, y="loss",
                 title="the loss of PolynomialFitting by k").show()

    # Question 5 - Evaluating fitted model on different countries
    p_r_5 = PolynomialFitting(5)
    p_r_5.fit(train_x["DayOfYear"].to_numpy(), train_y.to_numpy())
    loss5 = []
    for country in ["South Africa", "The Netherlands", "Jordan"]:
        x_test_5 = X[X["Country"] == country]
        y_test_5 = x_test_5["Temp"]
        loss = p_r_5.loss(x_test_5["DayOfYear"].to_numpy(), y_test_5.to_numpy())
        loss5.append(loss)

    loss5 = pd.DataFrame(loss5).rename(columns={0: "loss"})
    loss5["Country"] = ["South Africa", "The Netherlands", "Jordan"]
    px.bar(loss5, x="Country", y="loss",
           title="the loss of PolynomialFitting trained by israel as function of different countries by k").show()
