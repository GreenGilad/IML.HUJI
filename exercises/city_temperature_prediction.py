import IMLearn.learners.regressors.linear_regression
from IMLearn.learners.regressors import PolynomialFitting
from IMLearn.utils import split_train_test

import numpy as np
import pandas as pd
import datetime as dt
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
    dateparser = lambda x: int(dt.datetime.strftime(pd.Timestamp(x).to_pydatetime(), '%j')) / 365

    df = pd.read_csv(filename, parse_dates=['Date'])

    for country in np.unique(df.Country):
        city = df.City[df.Country == country].values[0]
        df[f'{country}_{city}'] = np.where(df.Country == country, 1, 0)

    df = df[df.Temp > -15]
    df['DayOfYear'] = df.Date.apply(dateparser)
    df.drop(columns=['City', 'Date'], inplace=True)
    return df


if __name__ == '__main__':
    np.random.seed(0)
    # Question 1 - Load and preprocessing of city temperature dataset
    df = load_data('../datasets/City_Temperature.csv')

    # Question 2 - Exploring data for specific country
    israel_df = df[df.Country == 'Israel']
    fig_a = px.scatter(israel_df, x='DayOfYear', y='Temp', color='Year')
    fig_a.update_layout(title_text='Temp in Israel by Day of Year over Several Years',
                        title_x=0.5)
    fig_a.show()

    fig_b = px.bar(x=pd.unique(israel_df.Month),
                   y=israel_df.groupby(israel_df.Month).Temp.apply(np.std),
                   labels = dict(x='Month',
                                 y='Standard Deviation of Temp'))
    fig_b.update_layout(title_text='STD of Temp by Month',
                         title_x=0.5)
    fig_b.show()

    # Question 3 - Exploring differences between countries
    q3_df = pd.DataFrame({
        'Std': df.groupby(['Country', 'Month']).Temp.apply(np.std),
        'Mean': df.groupby(['Country', 'Month']).Temp.apply(np.mean),
        'Country': [pair[0] for pair in df.groupby(['Country', 'Month']).indices.keys()],
        'Month': [pair[1] for pair in df.groupby(['Country', 'Month']).indices.keys()],
    })

    fig_3 = px.line(q3_df, x='Month', y='Mean', color='Country', error_y='Std')
    fig_3.update_layout(title_text='Mean Temp by Month and Country',
                        title_x=0.5)
    fig_3.show()


    # Question 4 - Fitting model for different values of `k`
    israel_temp = df[df.Country == 'Israel']
    israel_y = israel_temp.Temp.to_numpy()
    israel_x = israel_temp.DayOfYear.to_numpy()
    #
    train_x, train_y, test_x, test_y = split_train_test(israel_x, israel_y)
    #
    error = []
    for k in range(1, 11):
        poly_model = PolynomialFitting(k)
        poly_model.fit(train_x.to_numpy(), train_y.to_numpy())
        err = round(poly_model.loss(test_x.to_numpy(), test_y.to_numpy()), 2)
        print(f'k={k}, error={err}')
        error.append(err)

    fig_4 = px.bar(x=range(1, 11), y=error, labels=dict(x='k', y='Loss'))
    fig_4.update_layout(title_text='Loss of Polynomial Regression by k',
                        title_x=0.5)
    fig_4.show()



    # Question 5 - Evaluating fitted model on different countries
    poly_model = PolynomialFitting(5)
    poly_model.fit(israel_x, israel_y)

    error_5 = {}

    for country, group in df.groupby('Country'):
        if country == 'Israel':
            continue
        error_5[country] = poly_model.loss(np.array(group.DayOfYear), np.array(group.Temp))

    res_5 = pd.DataFrame({'Country': error_5.keys(),
                          'Loss': error_5.values(),})
    fig_5 = px.bar(res_5,x='Country', y='Loss', color='Country')
    fig_5.update_layout(title_text='Loss on all countries when fitted to Israel with k=6',
                        title_x=0.5)
    fig_5.show()
