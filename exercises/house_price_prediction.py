from os import path

from IMLearn.utils import split_train_test
from IMLearn.learners.regressors import LinearRegression

from typing import NoReturn
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio

pio.templates.default = "simple_white"


DATE_REGEX = r'(20[0-2][0-9])((0[1-9])|(1[0-2]))([0][1-9]|[1-2][0-9]|3[0-1])T[0]{6}'
normalize_coef = {}
MAX_RANDOM_SEED = 1000

def load_data(filename: str):
    """
    Load house prices dataset and preprocess data.
    Parameters
    ----------
    filename: str
        Path to house prices dataset

    Returns
    -------
    Design matrix and response vector (prices) - either as a single
    DataFrame or a Tuple[DataFrame, Series]
    """
    # read data:
    df = pd.read_csv(filename)

    # clean data:
    # price:
    df = df[df.price > 0]

    # date:
    df = df[df.date.str.match(DATE_REGEX)==True]
    df.date = pd.to_datetime(df.date).\
        apply(lambda x: ((x.year - 2010) * 365 + x.month * 30 + x.day) / 2500)

    # zip:
    df.zipcode = df.zipcode - 98000
    for z in sorted(list(np.unique(df.zipcode))):
        df[f'zip_{int(z)}'] = (df.zipcode == z).astype(int)

    # rooms and area:
    df = df[(df.bathrooms > 0) & (df.bedrooms > 0) &
            (df.bedrooms < 15) & (df.sqft_lot15 > 0) &
            (df.sqft_basement > 0)]

    # lat:
    df.lat = df.lat - 47

    # long:
    df.long = -df.long - 121

    # year renovated
    df.yr_renovated = np.where(df.yr_renovated > 0, df.yr_renovated - 1930, 0)

    # remove un used values:
    df.drop(columns=['id', 'zipcode'], inplace=True)

    # split:
    return df[[feature for feature in list(df.columns) if feature != 'price']],\
           df.price


def feature_evaluation(X: pd.DataFrame, y: pd.Series, output_path: str = ".") -> NoReturn:
    """
    Create scatter plot between each feature and the response.
        - Plot title specifies feature name
        - Plot title specifies Pearson Correlation between feature and response
        - Plot saved under given folder with file name including feature name
    Parameters
    ----------
    X : DataFrame of shape (n_samples, n_features)
        Design matrix of regression problem

    y : array-like of shape (n_samples, )
        Response vector to evaluate against

    output_path: str (default ".")
        Path to folder in which plots are saved
    """

    df = pd.concat([X, y], axis=1)

    for feature in X.columns[:20]:
        if feature[0:3] == 'zip':
            continue
        fig = px.scatter(df[[feature, y.name]], x=feature, y=y.name)
        pearson = np.cov(df[feature], y)[0][1] / (np.std(df[feature])*np.std(y))

        title = f'{feature} - Pearson: {pearson:.4f}'

        fig.update_layout(
            title_text=title,
            title_x=0.5)

        fig.show()
        pio.write_image(fig, path.join(output_path, f"pearson correlation of {feature}.png"))


if __name__ == '__main__':
    np.random.seed(0)
    # Question 1 - Load and preprocessing of housing prices dataset
    x, y = load_data('../datasets/house_prices.csv')

    # Question 2 - Feature evaluation with respect to response
    feature_evaluation(x, y, '../plots/ex2')

    # Question 3 - Split samples into training- and testing sets.
    x_train, y_train, x_test, y_test = split_train_test(x, y, 0.75)

    # Question 4 - Fit model over increasing percentages of the overall training data
    # For every percentage p in 10%, 11%, ..., 100%, repeat the following 10 times:
    #   1) Sample p% of the overall training data
    #   2) Fit linear model (including intercept) over sampled set
    #   3) Test fitted model over test set
    #   4) Store average and variance of loss over test set
    # Then plot average loss as function of training size with error ribbon of size (mean-2*std, mean+2*std)
    df_train = pd.concat([x_train, y_train], axis=1)
    lin_reg_model = LinearRegression()
    loss_df = pd.DataFrame(columns=['x', 'loss', '2_std_up', '2_std_down'])
    loss_df.x = np.linspace(0.1, 1, 91).round(2)

    for i, frac in enumerate(loss_df.x):
        print(f'frac: {frac}')
        # std_arr = []
        loss_arr = []
        for _ in range(10):
            # generate train sample:
            sample = df_train.sample(frac=frac, random_state=np.random.randint(MAX_RANDOM_SEED))

            # train model:
            lin_reg_model.fit(sample.drop('price', axis=1), sample.price)

            # predict:
            loss = lin_reg_model.loss(np.array(x_test), np.array(y_test).reshape(-1, 1))
            loss_arr.append(loss)

        # save results:
        loss = np.mean(loss_arr)
        std = np.std(loss_arr)
        loss_df.at[i, 'loss'] = loss
        loss_df.at[i, '2_std_up'] = loss + (2 * std)
        loss_df.at[i, '2_std_down'] = loss - (2 * std)


    fig = px.line(loss_df, x='x', y=['loss', '2_std_up', '2_std_down'])
    fig.update_layout(
        title_text='Loss and Std over training size',
        title_x=0.5)
    fig.show()



