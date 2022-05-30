import pandas as pd
import plotly.express as px
import plotly.io as pio

from IMLearn.learners.regressors.polynomial_fitting import *
from IMLearn.utils.utils import split_train_test

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
    df = pd.read_csv(filename).dropna().drop_duplicates()
    df = df[df["Temp"] > -50]
    df['Date'] = pd.to_datetime(df['Date'])
    df = df[df['Month'] == df['Date'].dt.month]
    df = df[df['Day'] == df['Date'].dt.day]
    df = df[df['Year'] == df['Date'].dt.year]
    df['DayOfYear'] = df['Date'].dt.dayofyear
    return df


if __name__ == '__main__':
    np.random.seed(0)
    # Question 1 - Load and preprocessing of city temperature dataset
    X = load_data("../datasets/City_Temperature.csv")

    # Question 2 - Exploring data for specific country
    X_IL = X[X['Country'] == "Israel"]

    X_IL["Year"] = X_IL["Year"].astype(str)
    fig = px.scatter(X_IL, x="DayOfYear", y="Temp", color="Year",
                     title="Temperatures in Israel as function of "
                           "day of the year")
    fig.show()

    X_IL_month = X_IL.groupby('Month')["Temp"].std()
    X_IL_month = X_IL.reset_index()
    fig = px.bar(X_IL_month, x="Month", y="Temp",
                 title="Temperatures std in Israel by month")
    fig.update_traces(marker_color='rgb(158,202,225)',
                      marker_line_color='rgb(8,48,107)',
                      marker_line_width=1.5, opacity=0.6)
    fig.show()

    # Question 3 - Exploring differences between countries
    X_std = X.groupby(['Month', 'Country'])["Temp"].std().reset_index()
    X_mean = X.groupby(['Month', 'Country'])["Temp"].mean().reset_index()
    fig = px.line(X_mean, x="Month", y="Temp", line_group="Country",
                  color='Country',
                  title="Temperatures means and std for all countries "
                        "by month", error_y=X_std["Temp"])
    fig.show()

    # Question 4 - Fitting model for different values of `k`
    y_true = X_IL["Temp"]
    X_IL = X_IL.drop("Temp", axis=1)
    x_train, y_train, x_test, y_test = split_train_test(X_IL, y_true)
    loss = []
    for k in range(1, 11):
        polreg = PolynomialFitting(k)
        polreg.fit(x_train["DayOfYear"], y_train.to_numpy())
        loss.append(polreg.loss(x_test["DayOfYear"], y_test.to_numpy()))

    loss = pd.DataFrame(loss).rename(columns={0: "loss"})
    loss.index = range(1, len(loss) + 1)
    fig = px.bar(loss, x=loss.index, y="loss",
                 title="Loss as function of k",
                 labels={"index": "k"})
    fig.update_traces(marker_color='rgb(158,202,225)',
                      marker_line_color='rgb(8,48,107)',
                      marker_line_width=1.5, opacity=0.6)
    fig.show()

    # Question 5 - Evaluating fitted model on different countries
    best_model = PolynomialFitting(5)
    best_model.fit(X_IL["DayOfYear"].to_numpy(), y_true.to_numpy())
    errors = []
    countries = ["South Africa", "The Netherlands", "Jordan"]
    for c in countries:
        new_x = X[X['Country'] == c]
        errors.append(best_model.loss(new_x['DayOfYear'].to_numpy(), new_x['Temp'].to_numpy()))

    errors = pd.DataFrame(errors).rename(columns = {0:"error"})
    errors["countries"] = countries
    fig = px.bar(errors, x="countries", y="error",
                 title="error for each country")
    fig.update_traces(marker_color='rgb(158,202,225)',
                      marker_line_color='rgb(8,48,107)',
                      marker_line_width=1.5, opacity=0.6)
    fig.show()
