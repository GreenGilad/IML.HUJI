from datetime import date, datetime

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
REDUNDANT_COLS = ["id", "lat", "long"]
DATE_TIME_FORMAT = "%Y%m%dT%H%M%S%f"


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

    # relative date for today in months
    today = dateutil.utils.today()
    df["date"] = dt["date"].apply(lambda x: (today - datetime.strptime(x, DATE_TIME_FORMAT)) // 30)

    # change discrete zipcodes to linear, set resolution to 10 units intervals (98000, 98010..98200)
    zip = pd.get_dummies()

    # change first column to intercept of ones
    dt['id'] = dt['id'].apply(lambda x: 1)
    pd.cut()

    # cs solution
    df = df.dropna().drop_duplicates()





    df["recently_renovated"] = np.where(df["yr_renovated"] >= np.percentile(df.yr_renovated.unique(), 70), 1, 0)
    df = df.drop("yr_renovated", 1)

    df["decade_built"] = (df["yr_built"] / 10).astype(int)
    df = df.drop("yr_built", 1)

    df["zipcode"] = df["zipcode"].astype(int)
    df = pd.get_dummies(df, prefix='zipcode_', columns=['zipcode'])
    df = pd.get_dummies(df, prefix='decade_built_', columns=['decade_built'])

    # Removal of outliers (Notice that there exists methods for better defining outliers
    # but for this course this suffices
    df = df[df["bedrooms"] < 20]
    df = df[df["sqft_lot"] < 1250000]
    df = df[df["sqft_lot15"] < 500000]

    df.insert(0, 'intercept', 1, True)
    return df.drop("price", 1), df.price


if __name__ == '__main__':
    np.random.seed(0)
    # Question 1 - Load and preprocessing of housing prices dataset
    dt = load_data("datasets/house_prices.csv")
    process_houses_data_frame(dt)
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
