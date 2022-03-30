from datetime import date

import dateutil.utils
import pandas

from IMLearn.utils import split_train_test
from IMLearn.learners.regressors import LinearRegression

from typing import NoReturn
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio

pio.templates.default = "simple_white"
POSITIVE_OR_ZERO_COLS = ["yr_renovated", "floors", "sqft_basement","bathrooms"]
POSITIVE_COLS = ["sqft_living", "price", "sqft_living15", "sqft_above", "yr_built", "sqft_lot", "sqft_lot15"]
REDUNDANT_COLS = ["lat", "long"]
DATE_TIME_FORMAT = "%Y%m%dT%H%M%S%f"
MAX_ROOMS = 15
MAX_LOT_SQRT = 1250000
MAX_LOT_14_SQRT = 500000

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
    return pandas.read_csv(filename)


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


def process_houses_data_frame(df: pandas.DataFrame):
    # remove missing and duplicate data rows
    df = df.dropna().drop_duplicates()

    # remove data with invalid values (except MAY_BE_NEGATIVE_COLS)

    df = df[df["sqft_lot15"] < MAX_LOT_14_SQRT]
    df = df[df["bedrooms"] < MAX_ROOMS]
    df = df[df["sqft_lot"] < MAX_LOT_SQRT]

    for col in POSITIVE_COLS:
        df = df[df[col] > 0]
    for col in POSITIVE_OR_ZERO_COLS:
        df = df[df[col] >= 0]

    df = df[df['view'].isin(range(5)) &
            df['grade'].isin(range(1, 15)) &
            df['waterfront'].isin([0, 1]) &
            df['condition'].isin(range(1,  6))]

    # remove redundant cols
    for col in REDUNDANT_COLS:
        df = df.drop(col, 1)

    # merge yr_renovated and yr_built to one col
    df['today_year'] = int(date.today().year)
    df["yr_renovated"] = df["yr_renovated"].astype(int)
    df["yr_built"] = df["yr_built"].astype(int)

    df['yr_built'] = df.apply(lambda x:
                              x['yr_built'] if x['yr_built'] >= x["yr_renovated"] - 5 else x["yr_renovated"],
                              axis=1)
    df['yr_built'] = df.apply(lambda x:
                              x['today_year'] if x['today_year'] < x['yr_built'] else x["yr_built"],
                              axis=1)

    df = df.drop("yr_renovated", 1)
    df = df.drop("today_year", 1)


    # relative date for today in months
    # change discrete zipcodes to linear, set resolution to 10 units intervals (98000, 98010..98200)


    today = date.today()
    #df["date"] = dt["date"].apply(lambda x: (today - datetime.strptime(x, DATE_TIME_FORMAT).days) // 30)
    df['date'] = df['date'].apply(
        lambda x: (today - date(int(x[0:4]), int(x[4:6]), int(x[6:8]))).days // 30)

    # change first col to intercept of ones
    df['id'] = df['id'].apply(lambda x: 1)

    # form zipcode as dummies features
    df["zipcode"] = df["zipcode"].astype(int)
    df = pd.get_dummies(df, prefix='zipcode_', columns=['zipcode'])

    return df

if __name__ == '__main__':
    np.random.seed(0)
    # Question 1 - Load and preprocessing of housing prices dataset
    df = load_data("datasets/house_prices.csv")
    df = process_houses_data_frame(df)
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
