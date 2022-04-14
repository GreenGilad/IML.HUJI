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
    data = pd.read_csv(filename, parse_dates=['Date'])
    # clean NA's
    data.dropna(inplace=True)
    unreal_temp_ind = data[data['Temp'] == -72.77777777777777].index
    data.drop(unreal_temp_ind, inplace=True)
    data['Year'] = data.loc[:, 'Year'].astype(str)
    data['DayOfYear'] = data.apply(lambda row: row.Date.timetuple().tm_yday, axis=1)
    # data.to_csv(r'C:\Users\elish\OneDrive\Desktop\test_city_temp.csv')
    return data


if __name__ == '__main__':
    np.random.seed(0)
    # Question 1 - Load and preprocessing of city temperature dataset
    data_loaded = load_data(r"C:\Users\elish\OneDrive\Documents\GitHub\IML\IML.HUJI\datasets\City_Temperature.csv")

    # Question 2 - Exploring data for specific country
    only_israel = data_loaded[data_loaded['Country'] == 'Israel']
    fig2_1 = px.scatter(x=only_israel['DayOfYear'], y=only_israel['Temp'], color=only_israel['Year'],
                        labels={'x': 'Day Of Year', 'y': 'Temperature'},
                        title='Temp as a function of DayOfYear')
    fig2_1.write_html('fig2_1.html', auto_open=True)

    sd_func = lambda x: np.var(x) ** 0.5
    israel_temp_sd_by_month = only_israel.groupby('Month').Temp.agg(sd_func)
    fig2_2 = px.bar(israel_temp_sd_by_month, labels={'value': 'sd of daily temp', 'Month': 'Month'},
                    title='The sd of daily temperature by months')
    fig2_2.write_html('fig2_2.html', auto_open=True)

    # Question 3 - Exploring differences between countries
    avrg_func = lambda x: np.mean(x)
    data_by_country_and_month = data_loaded.groupby(['Country', 'Month']).Temp.agg([sd_func, avrg_func]).reset_index()
    fig3 = px.line(data_by_country_and_month, x='Month', y='<lambda_1>', color='Country', error_y='<lambda_0>',
                   labels={'<lambda_1>': 'Mean temperature'})
    fig3.write_html('fig3.html', auto_open=True)

    # Question 4 - Fitting model for different values of `k`
    random_data_il = only_israel.iloc[np.random.permutation(len(only_israel))]
    only_day_il = random_data_il['DayOfYear']
    only_temp_il = random_data_il['Temp']
    train_x, train_y, test_X, test_y = split_train_test(only_day_il, only_temp_il, train_proportion=0.75)
    loss_by_increasing_k = []
    for k in range(1, 11):
        model = PolynomialFitting(k=k)
        model.fit(train_x, train_y)
        loss = model.loss(test_X, test_y)
        loss_by_increasing_k.append(round(loss, 2))
        print('k = ' + str(k) + ': MSE = ' + str(round(loss, 2)))
    mse_k_df = pd.DataFrame({'k': range(1, 11), 'MSE': loss_by_increasing_k})
    fig4 = px.bar(x=mse_k_df['k'], y=mse_k_df['MSE'], labels={'x': 'degrees of the polynom', 'y': 'MSE'},
                  title='MSE by the degrees of the polynom')
    fig4.write_html('fig4.html', auto_open=True)

    # Question 5 - Evaluating fitted model on different countries
    model = PolynomialFitting(k=loss_by_increasing_k.index(min(loss_by_increasing_k)) + 1)
    model.fit(only_day_il, only_temp_il)
    countries = list(set(data_loaded['Country']) - {'Israel'})
    errors_by_country = list()
    for c in countries:
        only_c = data_loaded[data_loaded['Country'] == c]
        only_day = only_c['DayOfYear']
        only_temp = only_c['Temp']
        errors_by_country.append(model.loss(only_day, only_temp))
    fig5 = px.bar(x=countries, y=errors_by_country, labels={'x': 'Countries', 'y': 'MSE'},
                  title='MSE of prediction by fitted model on Israel')
    fig5.write_html('fig5.html', auto_open=True)
