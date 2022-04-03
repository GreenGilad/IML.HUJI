from IMLearn.utils import split_train_test
from IMLearn.learners.regressors import LinearRegression

from typing import NoReturn
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio
pio.templates.default = "simple_white"


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
    df = pd.read_csv(filename, index_col='id')
    # get rid of "date" feature
    # get rid of "lat" and "long" features - location will be encoded via one-hot zipcode entries
    # price feature to separate series
    # add total sqft feature (sum of sqft columns)
    # add ratio of rooms to total sqft (bedrooms+bathrooms / sqft_total)
    df.drop(['date', 'lat', 'long'], inplace=True, axis=1)
    zip_codes = pd.get_dummies(df.pop('zipcode'))
    df = pd.concat([df, zip_codes], axis=1)

    df['total_sqft'] = df.loc[:, 'sqft_living'] + df.loc[:, 'sqft_lot'] + df.loc[:, 'sqft_above']+ df.loc[:, 'sqft_basement']
    df['room_size'] = df['total_sqft'] / (df.loc[:, 'bedrooms'] + df.loc[:, 'bathrooms'])

    y = df.pop('price')

    return df, y


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
    raise NotImplementedError()


if __name__ == '__main__':
    np.random.seed(0)
    # Question 1 - Load and preprocessing of housing prices dataset
    raise NotImplementedError()

    # Question 2 - Feature evaluation with respect to response
    raise NotImplementedError()

    # Question 3 - Split samples into training- and testing sets.
    raise NotImplementedError()

    # Question 4 - Fit model over increasing percentages of the overall training data
    # For every percentage p in 10%, 11%, ..., 100%, repeat the following 10 times:
    #   1) Sample p% of the overall training data
    #   2) Fit linear model (including intercept) over sampled set
    #   3) Test fitted model over test set
    #   4) Store average and variance of loss over test set
    # Then plot average loss as function of training size with error ribbon of size (mean-2*std, mean+2*std)
    raise NotImplementedError()
