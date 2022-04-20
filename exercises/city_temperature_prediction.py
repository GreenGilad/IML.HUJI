import IMLearn.learners.regressors.linear_regression
from IMLearn.learners.regressors import PolynomialFitting
from IMLearn.utils import split_train_test

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.io as pio
pio.templates.default = "simple_white"
from IMLearn.utils import split_train_test


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
    df = pd.read_csv(filename,parse_dates=["Date"]).dropna().drop_duplicates()
    df["DayOfYear"]=df["Date"].dt.day_of_year
    df = df[df["Year"].isin(range(2023))]
    df = df[df["Temp"]>-70]
    df = df[df["Temp"] < 70]
    return df


if __name__ == '__main__':
    np.random.seed(0)
    # Question 1 - Load and preprocessing of city temperature dataset
    df=load_data('../datasets/City_Temperature.csv')

    # Question 2 - Exploring data for specific country
    #Part 1
    df_israel=df[df["Country"]=="Israel"]
    fig1=px.scatter(df_israel, title="Average daily temperature in Israel", x="DayOfYear", y="Temp",
                   color=df_israel.Year.astype((str)))
    fig1.show()

    #Part 2
    fig2=px.bar(df_israel.groupby("Month").Temp.agg(["std"]),
                labels={'value': "Standard deviation of the daily temperature"},
                title="Standard deviation of the daily temperatures")
    fig2.show()

    # Question 3 - Exploring differences between countries

    df_group_by_country_and_month=(df.groupby(["Country","Month"],as_index=False)).agg({"Temp":["mean","std"]})
    fig3=px.line(x=df_group_by_country_and_month["Month"], y=df_group_by_country_and_month[("Temp", "mean")],
                 error_y=df_group_by_country_and_month[("Temp","std")],
                 title="The Average Monthly Temperature With Error Bar",
                 color=df_group_by_country_and_month["Country"],labels=dict(x="Month", y="Mean Temp",color="Country"))

    fig3.show()


    # Question 4 - Fitting model for different values of `k`
    train_x, train_y, test_x, test_y=split_train_test(df_israel["DayOfYear"],df_israel["Temp"])
    k_values=range(1,11)
    loss_arr=np.zeros(10)
    for k in k_values:
        poly_fit=PolynomialFitting(k)
        poly_fit.fit(train_x.to_numpy(), train_y.to_numpy())
        loss=round(poly_fit.loss(test_x.to_numpy(),test_y.to_numpy()),2)
        print(f"Test error for k = {k} is {loss}")
        loss_arr[k-1]=loss

    fig4 = px.bar(x=k_values,y=loss_arr,labels=dict(x="Degree", y="Loss Value"),
                  title="Test Error Recorded For Each Value of k")
    fig4.show()



    # Question 5 - Evaluating fitted model on different countries
    poly_fit_israel=PolynomialFitting(5)
    poly_fit_israel.fit(df_israel["DayOfYear"].to_numpy(), df_israel["Temp"].to_numpy())

    df_counties=df.groupby(["Country"])

    countries_arr=[]
    loss_arr = []
    for country_name, df_county in df_counties:
        temp_vec=df_county["Temp"]
        day_of_year_vec=df_county["DayOfYear"]
        loss = poly_fit_israel.loss(day_of_year_vec.to_numpy(), temp_vec.to_numpy())
        countries_arr.append(country_name)
        loss_arr.append(loss)

    fig5 = px.bar(x=countries_arr, y=loss_arr, labels=dict(x="Countries", y="Loss Value"),
                  title="The loss of the other countries for k=5")
    fig5.show()
