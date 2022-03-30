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
    dateparser = lambda x: int(dt.datetime.strftime(pd.Timestamp(x).to_pydatetime(), '%j'))

    df = pd.read_csv(filename, parse_dates=['Date'])

    for country in np.unique(df.Country):
        city = df.City[df.Country == country].values[0]
        df[f'{country}_{city}'] = np.where(df.Country == country, 1, 0)

    df = df[df.Temp > -15]
    df['DayOfYear'] = df.Date.apply(dateparser)
    # df.drop(columns=['Country', 'City', 'Date'], inplace=True)
    df.drop(columns=['City', 'Date'], inplace=True)
    return df


if __name__ == '__main__':
    np.random.seed(0)
    # Question 1 - Load and preprocessing of city temperature dataset
    df = load_data('../datasets/City_Temperature.csv')

    # Question 2 - Exploring data for specific country
    # israel_df = df[df.Country == 'Israel']
    # fig_a = px.scatter(israel_df, x='DayOfYear', y='Temp', color='Year')
    # fig_a.update_layout(title_text='Temp in Israel by Day of Year over Several Years',
    #                     title_x=0.5)
    # fig_a.show()
    #
    # fig_b = px.bar(x=pd.unique(israel_df.Month),
    #                y=israel_df.groupby(israel_df.Month).Temp.apply(np.std),
    #                labels = dict(x='Month',
    #                              y='Standard Deviation of Temp'))
    # fig_b.update_layout(title_text='STD of Temp by Month',
    #                      title_x=0.5)
    # fig_b.show()

    # Question 3 - Exploring differences between countries
    q3_df = pd.DataFrame({
        'Std': df.groupby(['Country', 'Month']).Temp.apply(np.std),
        'Country': [pair[0] for pair in df.groupby(['Country', 'Month']).indices.keys()],
        'Month': [pair[1] for pair in df.groupby(['Country', 'Month']).indices.keys()],
    })

    q3_fig = px.bar(q3_df, x='')
    # std_full = groups.Temp.apply(np.std)
    # print(pd.unique(groups[['Country', 'Month']]))
    # std_full = df.groupby(['Country', 'Month']).Temp.apply(np.std)
    print('a')

    # Question 4 - Fitting model for different values of `k`
    raise NotImplementedError()

    # Question 5 - Evaluating fitted model on different countries
    raise NotImplementedError()
