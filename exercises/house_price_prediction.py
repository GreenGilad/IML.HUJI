from IMLearn.utils import split_train_test
from IMLearn.learners.regressors import LinearRegression

from typing import NoReturn
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio

pio.templates.default = "simple_white"


def get_only_positive(data_frame):
    # These values can't be negative:
    for col in \
            ["bathrooms",
             "floors",
             "sqft_basement",
             "yr_renovated"]:
        data_frame = data_frame[data_frame[col] >= 0]

    # These values can't be negative or zero:
    for col in \
            ["price",
             "sqft_living",
             "sqft_lot",
             "sqft_above",
             "yr_built",
             "sqft_living15",
             "sqft_lot15"]:
        data_frame = data_frame[data_frame[col] > 0]

    return data_frame


def filter_by_range(data_frame):
    data_frame = \
        data_frame[data_frame["waterfront"].isin([0, 1]) &
                   data_frame["view"].isin(range(5)) &
                   data_frame["condition"].isin(range(1, 6)) &
                   data_frame["grade"].isin(range(1, 15))]

    data_frame = data_frame[data_frame["bedrooms"] < 20]
    data_frame = data_frame[data_frame["sqft_lot"] < 1250000]
    data_frame = data_frame[data_frame["sqft_lot15"] < 500000]

    return data_frame


def make_indicators(data_frame):
    data_frame['last_20_renovated'] = np.where((2022 - data_frame['yr_renovated'] <= 20), 1, 0)
    data_frame["decade_built"] = (data_frame["yr_built"] / 10).astype(int)

    data_frame = data_frame.drop("yr_renovated", 1)
    data_frame = data_frame.drop("yr_built", 1)

    # Making zipcode an indicator
    data_frame["zipcode"] = data_frame["zipcode"].astype(int)
    data_frame = \
        pd.get_dummies(data_frame,
                       prefix='zipcode', columns=['zipcode'])

    # Making decade_built an indicator
    data_frame = \
        pd.get_dummies(
            data_frame,
            prefix='decade_built', columns=['decade_built'])

    return data_frame


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
    data_frame = pd.read_csv(filename).dropna().drop_duplicates()

    # id and date are not valuable for the price calculation
    # the lat and long coordinates are in correlation with the zipcode
    data_frame.drop(columns=['id', 'date', 'lat', 'long'], inplace=True)

    data_frame = get_only_positive(data_frame)
    data_frame = filter_by_range(data_frame)
    data_frame = make_indicators(data_frame)

    data_frame.insert(0, 'intercept', 1, True)

    price_vector = data_frame.pop('price')
    return data_frame, price_vector


def calculate_pearson(feature1, feature2):
    return (np.cov(feature1, feature2) / (np.std(feature1) * np.std(feature2)))[0][1]


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
    features = [
        'bedrooms',
        'bathrooms',
        'sqft_living',
        'sqft_lot',
        'floors',
        'view',
        'condition',
        'grade',
        'sqft_above',
        'sqft_basement',
        'yr_built',
        'yr_renovated',
        'sqft_living15',
        'sqft_lot15']

    for i in range(len(features)):
        feature = X[features[i]]
        fig = go.Figure()
        fig.update_layout(
            title=
            f"Price as a function of the feature: {features[i]}, {round(calculate_pearson(feature, y), 2)}",
            xaxis_title=f"{features[i]}",
            yaxis_title='Price'
        )

        fig.add_scatter(
            x=feature,
            y=y,
            mode="markers"
        )

        fig.write_image(f'{output_path}/{features[i]}.png')


if __name__ == '__main__':
    np.random.seed(0)
    # Question 1 - Load and preprocessing of housing prices dataset
    features, prices = load_data("../datasets/house_prices.csv")

    # Question 2 - Feature evaluation with respect to response
    feature_evaluation(features, prices, './ex2_plots')

    # Question 3 - Split samples into training- and testing sets.
    # raise NotImplementedError()

    # Question 4 - Fit model over increasing percentages of the overall training data
    # For every percentage p in 10%, 11%, ..., 100%, repeat the following 10 times:
    #   1) Sample p% of the overall training data
    #   2) Fit linear model (including intercept) over sampled set
    #   3) Test fitted model over test set
    #   4) Store average and variance of loss over test set
    # Then plot average loss as function of training size with error ribbon of size (mean-2*std, mean+2*std)
    # raise NotImplementedError()
