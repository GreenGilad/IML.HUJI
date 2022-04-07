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

    df = pd.read_csv(filename, parse_dates={"DayOfYear": [2]})
    df = df[(-30 <= df["Temp"]) & (df["Temp"] <= 58)]
    df = df[(1 <= df["Day"]) & (df["Day"] <= 31)]
    df = df[(1 <= df["Month"]) & (df["Month"] <= 12)]
    df = df[(1900 <= df["Year"]) & (df["Year"] <= 2022)]
    df["DayOfYear"] = df["DayOfYear"].apply(lambda x: x.dayofyear)
    return df


if __name__ == '__main__':
    np.random.seed(0)
    plot_dir = "ex2_plots"
    data_filepath = "/Users/natandavids/IML/IML.HUJI/datasets/City_Temperature.csv"


    # Question 1 - Load and preprocessing of city temperature dataset
    data = load_data(data_filepath)

    # Question 2 - Exploring data for specific country
    Israel_temperature_plot_filename = "/Israel_Daily_Temp.png"

    Israel_data = data[data["Country"] == "Israel"]
    Israel_data["Year"] = Israel_data["Year"].astype(str)
    temperature_graph = px.scatter(Israel_data, x="DayOfYear", y="Temp", color="Year", title="Daily Temperature in Israel")
    temperature_graph.show()
    temperature_graph.write_image(plot_dir + Israel_temperature_plot_filename)

    month_std_plot_filename = "/Monthly_STD.png"

    monthly = Israel_data.groupby("Month")
    monthly = monthly["Temp"].std()
    monthly_std_graph = px.bar(monthly, title="Standard Deviation by Month")
    monthly_std_graph.show()
    monthly_std_graph.write_image(plot_dir + month_std_plot_filename)

    # Question 3 - Exploring differences between countries
    month_mean_plot_filename = "/Monthly_AVG.png"

    grouped = data[["Country", "Month", "Temp"]].groupby(["Country", "Month"]).agg([np.mean, np.std])
    grouped = grouped.reset_index()
    months = grouped[("Month",)]
    mean = grouped[("Temp", "mean")]
    std = grouped[("Temp", "std")]
    country = grouped[("Country",)]
    avg_monthly_temp = px.line(x=months, y=mean, error_y=std, color=country, title="Mean Temperature by Month")
    avg_monthly_temp.show()
    avg_monthly_temp.write_image(plot_dir + month_mean_plot_filename)

    # Question 4 - Fitting model for different values of `k`
    lossplot_filename = "/Loss_by_Polynomial_Degree.png"
    max_deg = 10
    round_to = 2

    loss = np.empty(max_deg)
    Israel_X, Israel_y, Israel_test_X, Israel_test_y = split_train_test(Israel_data["DayOfYear"].to_frame(), Israel_data["Temp"])
    Israel_X, Israel_y, Israel_test_X, Israel_test_y = Israel_X.to_numpy().squeeze(), Israel_y.to_numpy(), Israel_test_X.to_numpy().squeeze(), Israel_test_y.to_numpy()
    for k in range(1, max_deg+1):
        model = PolynomialFitting(k)
        model.fit(Israel_X, Israel_y)
        loss[k-1] = round(model.loss(Israel_test_X, Israel_test_y), round_to)
        print(loss[k-1])

    loss_graph = px.bar(x=np.arange(1,11), y=loss, title="Loss of Polynomial Model by Degree", labels={'x': "degree k", 'y': "loss"})
    loss_graph.show()
    loss_graph.write_image(plot_dir + lossplot_filename)

    # Question 5 - Evaluating fitted model on different countries
    loss_by_country_filename = "/Loss_By_Country.png"
    deg = np.argmin(loss) + 1
    Israel_model = PolynomialFitting(deg)
    Israel_model.fit(Israel_data["DayOfYear"].to_numpy(), Israel_data["Temp"].to_numpy())
    countries = data["Country"].unique()
    loss = np.empty_like(countries)
    for i, country in enumerate(countries):
        subframe = data[data["Country"]==country]
        loss[i] = Israel_model.loss(subframe["DayOfYear"].to_numpy(), subframe["Temp"].to_numpy())

    loss_by_country = px.bar(x=countries, y=loss, title="Israel Fitted Model Loss by Country")
    loss_by_country.show()
    loss_by_country.write_image(plot_dir + loss_by_country_filename)
