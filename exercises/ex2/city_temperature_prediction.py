import pandas
from pandas import DataFrame
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
    return pandas.read_csv(filename, parse_dates=True)


def process_city_temperatures(df: DataFrame) -> pd.DataFrame:
    # remove data with invalid values
    df = df[df['Day'].isin(range(32))]
    df = df[df['Month'].isin(range(13))]
    df = df[df['Year'].isin(range(2023))]
    df["Temp"] = df["Temp"].astype(int)
    df = df[df['Temp'].isin(range(-50, 50))]

    # create DayOfYear coulomn
    date_calculator = lambda row: pd.to_datetime(row.Date).dayofyear
    df['DayOfYear'] = df.apply(date_calculator, axis=1)
    df["Year"] = df["Year"].astype(str)
    df["Temp"] = df["Temp"].astype(int)

    return df


if __name__ == '__main__':
    np.random.seed(0)
    # Question 1 - Load and preprocessing of city temperature dataset
    df = load_data("datasets/City_Temperature.csv")
    df = process_city_temperatures(df)

    # Question 2 - Exploring data for specific country
    israel_df = df[df["Country"] == "Israel"]
    fig = px.scatter(israel_df, x="DayOfYear", y="Temp", color="Year")
    fig.show()

    deviation_by_months_in_IL = israel_df.groupby(['Month']).Temp.agg(['std'])
    fig = px.bar(deviation_by_months_in_IL, y='std', title="Deviations value as a function of month in Israel")
    fig.show()

    # Question 3 - Exploring differences between countries
    deviation_and_mean_by_months_and_countries = df.groupby(['Country', 'Month']).Temp.agg(['std', 'mean']).reset_index()
    fig = px.line(deviation_and_mean_by_months_and_countries, x='Month', y='mean', error_y='std', color='Country',
                  title="Mean Temperature as a function of month in any country",
                  labels={'mean': 'Mean monthly temperature'})
    fig.show()

    # Question 4 - Fitting model for different values of `k`
    train_X, train_y, test_X, test_y = split_train_test(df["DayOfYear"], df["Temp"])
    train_X, train_y, test_X, test_y = train_X.to_numpy(), train_y.to_numpy(), test_X.to_numpy(), test_y.to_numpy()
    results = []
    for k in range(1, 11):
        model = PolynomialFitting(k)
        model.fit(train_X, train_y)
        loss = np.round(model.loss(test_X, test_y), decimals=2)
        results.append(loss)
        print(f'Lost value is {loss}, for k {k}')
    fig = px.bar(y=results, x=range(1, 11), title="Loss Value as a function of degree k")
    fig.show()


    # Question 5 - Evaluating fitted model on different countries' data
    # setup model - 5 is the closest degree
    model = PolynomialFitting(5)
    model.fit(israel_df["DayOfYear"], israel_df["Temp"])

    # iterate on countries data
    countries = ["South Africa", "Israel", "The Netherlands", "Jordan"]
    results = []
    for country in countries:
        country_data = df[df["Country"] == country]
        results.append(model.loss(country_data["DayOfYear"], country_data["Temp"]))
    fig = px.bar(x=countries, y=results,  title="Loss Values for each country's data using israel data model")
    fig.show()
