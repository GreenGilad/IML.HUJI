import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.io as pio

import IMLearn.learners.regressors.linear_regression as lnrg
import IMLearn.utils.utils as utl

pio.templates.default = "simple_white"


def process_temp_values(df):
    df = df.drop(df[df.Temp < -10].index)
    dayOfYear = []
    for date in df['Date']:
        dayOfYear.append(date.dayofyear)

    df["DayOfYear"] = dayOfYear
    col_list = list(df)
    col_list[len(col_list) - 2], col_list[len(col_list) - 1] = col_list[len(col_list) - 1], col_list[len(col_list) - 2]
    df["Temp"], df["DayOfYear"] = df["DayOfYear"], df["Temp"]
    df.columns = col_list
    return df



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
    df = pd.read_csv(filename, parse_dates=['Date'], dayfirst=True)
    df = df.dropna()
    df = process_temp_values(df)
    df.drop(columns=['Date'], inplace=True)
    return df


if __name__ == '__main__':
    np.random.seed(0)
    # Question 1 - Load and preprocessing of city temperature dataset
    df = load_data('../datasets/City_Temperature.csv')

    # Question 2 - Exploring data for specific country

    israel_arr = df.loc[df['Country'] == "Israel"]
    x_arr = israel_arr['DayOfYear']
    y_arr = israel_arr['Temp']
    z_arr = israel_arr['Year']


    plt.scatter(x=x_arr, y=y_arr, c=z_arr, s=4)
    plt.xlabel("Day Of Year")
    plt.ylabel("Temperature")
    plt.title("Temperature as a function of Day Of Year")
    plt.show()

    std_of_month_temp = israel_arr.groupby(['Month','DayOfYear']).agg({'Temp':'std'}).groupby('Month').mean().to_numpy().reshape((1, 12))

    plt.bar(range(1, 13), std_of_month_temp[0])
    plt.xlabel("Month")
    plt.ylabel("Average Temperature")
    plt.title("Average Temperature as a function of Month")
    plt.show()
    # Question 3 - Exploring differences between countries
    country_names = set(df['Country'])

    for country in country_names:
        current_country = df.loc[df['Country'] == country]
        current_country_std = current_country.groupby(["Month", "DayOfYear"]).agg({'Temp': "std"}).groupby("Month").mean().to_numpy().reshape((1, 12))[0]
        current_country_mean = current_country.groupby(["Month", "DayOfYear"]).agg({'Temp': "mean"}).groupby("Month").mean().to_numpy().reshape((1, 12))[0]
        plt.plot(range(1, 13), current_country_mean, label=country)
        plt.fill_between(range(1, 13), current_country_mean-current_country_std, current_country_mean+current_country_std, alpha=0.5)
    plt.legend(loc="best")
    plt.xlabel("Month")
    plt.ylabel("Average Temperature")
    plt.title("Average Temperature as a function of Month")
    plt.show()

    # Question 4 - Fitting model for different values of `k`
    israel_temp = pd.array(israel_arr.pop("Temp"))
    israel_day_of_year = pd.DataFrame({"DayOfYear": israel_arr.pop("DayOfYear")})
    x_train, y_train, x_test, y_test = utl.split_train_test(israel_day_of_year, israel_temp, 0.75)

    x_vander_train = []
    y_vander_train = []
    x_vander_test = []
    y_vander_test = []

    x_train = x_train.to_numpy()
    y_train = y_train.to_numpy()
    x_test = x_test.to_numpy()
    y_test = y_test.to_numpy()

    loss = []

    for k in range(1, 11):
        for i, row in enumerate(x_train):
            x_vander_train.append(np.vander(row, k+1)[0])
            y_vander_train.append(y_train[i])

        for i, row in enumerate(x_test):
            x_vander_test.append(np.vander(row, k+1)[0])
            y_vander_test.append(y_test[i])

        model = lnrg.LinearRegression(False)
        model.fit(x_vander_train, y_vander_train)
        loss.append(model.loss(x_vander_test, y_vander_test))

        x_vander_train = []
        y_vander_train = []
        x_vander_test = []
        y_vander_test = []

    plt.bar(np.arange(1, 11), loss)
    plt.xlabel("Degree of polynomial")
    plt.ylabel("Loss")
    plt.title("Loss as a function of K")
    plt.show()

    # creating model
    k = np.argmin(loss) + 1

    for i, row in enumerate(x_train):
        x_vander_train.append(np.vander(row, k+1)[0])
        y_vander_train.append(np.ones(k)[0] * y_train[i])

    for i, row in enumerate(x_test):
        x_vander_test.append(np.vander(row, k+1)[0])
        y_vander_test.append(np.ones(k)[0] * y_test[i])

    model = lnrg.LinearRegression(False)
    model.fit(x_vander_train, y_vander_train)

    print("test error recorded for each value of k:", loss)

    # Question 5 - Evaluating fitted model on different countries
    loss = []
    x_val = []
    y_val = []
    countries = df['Country'].unique()
    for country in countries:
        curr_arr = df.loc[df['Country'] == country]
        for row in curr_arr["DayOfYear"].to_numpy():
            x_val.append(np.vander(np.array([row]), k+1)[0])
        for i, row in enumerate(curr_arr['Temp'].to_numpy()):
            y_val.append(row)

        loss.append(model.loss(x_val, y_val))
        x_val = []
        y_val = []

    plt.bar(countries, loss)
    plt.xlabel("Country")
    plt.ylabel("Loss")
    plt.title("Loss as a function of K")
    plt.show()
    print("Loss per state ", loss)