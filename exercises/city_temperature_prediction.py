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
    dff = pd.read_csv(filename, parse_dates=['Date'])
    # no nans, days & months & years are in range,

    # remove 99F (-72.77777777777777 C) - missing data
    indx = dff[dff['Temp'] == -72.77777777777777].index
    dff.drop(indx, inplace=True)
    # add dayof year column.
    dff['DayOfYear'] = dff.apply(lambda row: row.Date.timetuple().tm_yday, axis=1)
    return dff


if __name__ == '__main__':
    np.random.seed(0)
    # Question 1 - Load and preprocessing of city temperature dataset
    df = load_data(r'C:\Users\Rafi Levy\Documents\GitHub\IML.HUJI\datasets\City_Temperature.csv')
    df['Year'] = df.loc[:, 'Year'].astype(str)
    # Question 2 - Exploring data for specific country
    il_data = df[df['Country'] == 'Israel']

    fig2_1 = px.scatter(il_data, x='DayOfYear', y='Temp', color='Year', title='Temp as a function of DayOfYear')
    fig2_1.write_html('israel_temp_year.html', auto_open=True)

    sd_func = lambda v: np.sqrt(np.var(list(v)))
    month_df = il_data.groupby('Month').agg({'Temp': sd_func})

    fig2_2 = px.bar(month_df, height=300, labels={'x': 'Month', 'value': 'sd of daily Temp'},
                    title='bar plot of sd of Temps as a function of month')
    fig2_2.write_html('bar_plot_months.html', auto_open=True)

    # Question 3 - Exploring differences between countries
    avg_func = lambda v: np.mean(v)
    month_country_df = df.groupby(['Country', 'Month']).Temp.agg([sd_func, avg_func]).rename(
        columns={'<lambda_0>': 'sd', '<lambda_1>': 'mean'}).reset_index()

    fig3 = px.line(month_country_df, x='Month', y='mean', color='Country', title='mean temp as a function month',
                     error_y='sd')
    fig3.write_html('plot_months_country.html', auto_open=True)

    # Question 4 - Fitting model for different values of `k`
    temp_label = il_data['Temp']
    temp_feature = il_data['DayOfYear']
    train_x, train_y, test_x, test_y = split_train_test(temp_feature, temp_label, 0.75)
    res = [[], []]
    for i in range(1, 10):
        pol = PolynomialFitting(i)
        pol.fit(train_x.to_numpy(), train_y.to_numpy())
        res[0].append(i)
        res[1].append(round(pol.loss(test_x.to_numpy(), test_y.to_numpy()), 2))
    k_mse_df = pd.DataFrame({'k': res[0], 'mse': res[1]})
    print(k_mse_df)

    fig4_1 = px.bar(k_mse_df, x='k', y='mse', height=300, labels={'x': 'k', 'value': 'mse'},
                    title='bar plot of mse as a function of k')
    fig4_1.write_html('k_mse.html', auto_open=True)

    # Question 5 - Evaluating fitted model on different countries
    pol = PolynomialFitting(4)
    pol.fit(il_data['DayOfYear'].to_numpy(), il_data['Temp'].to_numpy())
    indx = df[df['Country'] == 'Israel'].index
    no_il = df.drop(indx, inplace=False)

    res = [[], []]
    for i in no_il.Country.unique():
        indx = no_il[df['Country'] != i].index
        country_df = no_il.drop(indx, inplace=False)
        res[0].append(i)
        res[1].append(pol.loss(country_df['DayOfYear'].to_numpy(), country_df['Temp'].to_numpy()))

    res = pd.DataFrame({'Country': res[0], 'mse': res[1]})
    fig5 = px.bar(res, x='Country', y='mse', height=300, labels={'x': 'k', 'value': 'mse'},
                    title='bar plot of mse as a function of the country')
    fig5.write_html('country_mse.html', auto_open=True)
