from datetime import date, datetime
import pandas
from typing import NoReturn, Tuple
import numpy as np
import pandas as pd
import plotly.io as pio
import plotly.express as px
import plotly.graph_objects as go

from IMLearn.learners.regressors import LinearRegression
from IMLearn.utils import split_train_test

pio.templates.default = "simple_white"
POSITIVE_OR_ZERO_COLS = ["yr_renovated", "floors", "sqft_basement", "bathrooms"]
POSITIVE_COLS = ["sqft_living", "price", "sqft_living15", "sqft_above", "yr_built", "sqft_lot", "sqft_lot15"]
REDUNDANT_COLS = ["lat", "long"]
DATE_TIME_FORMAT = "%Y%m%dT%H%M%S%f"
MAX_ROOMS = 15
MAX_LOT_SQRT = 1250000
MAX_LOT_14_SQRT = 500000
REPEAT_FACTOR = 3
RESOLUTION = 0.01
RENOVATED_FACTOR = 0.25


def load_data(filename: str) -> pd.DataFrame:
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
    features = X.columns
    features = [feat for feat in set(features) if 'zipcode' not in feat and 'id' != feat]
    deviation_y = np.std(y)
    best_beneficial_value, best_beneficial_feature = None, None
    worst_beneficial_value, worst_beneficial_feature = None, None

    for feature in features:
        if feature == 'id':
            continue
        covariance = np.cov(X[feature], y)[0, 1]
        deviation_x = np.std(X[feature])
        value = covariance / (deviation_x * deviation_y)
        if not best_beneficial_value or best_beneficial_value < value:
            best_beneficial_value = value
            best_beneficial_feature = feature
        if not worst_beneficial_value or worst_beneficial_value > value:
            worst_beneficial_value = value
            worst_beneficial_feature = feature
        fig = px.scatter(pd.DataFrame({'x': X[feature], 'y': y}), x="x", y="y", trendline="ols",
                         labels={"x": feature + " values", "y": "House price"},
                         title="Pearson Correlation between " + feature + " and price is " + f'{value:.3f}')
        fig.write_image(feature + "_correlation.png")

    print(f"Worst feature is {worst_beneficial_feature}, value is {worst_beneficial_value}\n"
          f"Best  feature is {best_beneficial_feature}, value is {best_beneficial_value}")


def process_houses_data_frame(df: pandas.DataFrame) \
        -> Tuple[pd.DataFrame, pd.Series]:
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
            df['condition'].isin(range(1, 6))]

    # remove redundant cols
    for col in REDUNDANT_COLS:
        df = df.drop(col, 1)

    # merge yr_renovated and yr_built to one col
    df['today_year'] = int(date.today().year)
    df["yr_renovated"] = df["yr_renovated"].astype(int)
    df["yr_built"] = df["yr_built"].astype(int)

    df['yr_built'] = df.apply(lambda x:
                              x['yr_built'] if x['yr_built'] >= x["yr_built"] + RENOVATED_FACTOR * (x["yr_renovated"] - x['yr_built'])
                              else x["yr_built"] + RENOVATED_FACTOR * (x["yr_renovated"]-x['yr_built']),
                              axis=1)

    df = df.drop("yr_renovated", 1)
    df = df.drop("today_year", 1)

    # relative date for today in months
    today = date.today()
    df['date'] = df['date'].apply(
        lambda x: (today - date(int(x[0:4]), int(x[4:6]), int(x[6:8]))).days // 30)

    # change first col to intercept of ones
    df['id'] = df['id'].apply(lambda x: 1)

    # form zipcode as dummies features
    df["zipcode"] = df["zipcode"].astype(int)
    df = pd.get_dummies(df, prefix='zipcode.', columns=['zipcode'])

    # Separate data X and prices Y
    prices = df['price']
    df = df.drop("price", 1)
    return df, prices


if __name__ == '__main__':
    np.random.seed(0)
    # Question 1 - Load and preprocessing of housing prices dataset
    df = load_data("datasets/house_prices.csv")
    X, y = process_houses_data_frame(df)
    # Question 2 - Feature evaluation with respect to response
    feature_evaluation(X, y)
    # Question 3 - Split samples into training- and testing sets.
    train_X, train_Y, test_X, test_Y = split_train_test(X, y)
    # Question 4 - Fit model over increasing percentages of the overall training data
    # For every percentage p in 10%, 11%, ..., 100%, repeat the following 10 times:
    #   1) Sample p% of the overall training data
    #   2) Fit linear model (including intercept) over sampled set
    #   3) Test fitted model over test set
    #   4) Store average and variance of loss over test set
    # Then plot average loss as function of training size with error ribbon of size (mean-2*std, mean+2*std)
    percentages = np.arange(0.1, 1.005, RESOLUTION)
    model = LinearRegression()
    means, deviations = [], []
    test_X, test_Y = test_X.to_numpy(), test_Y.to_numpy()
    for p in percentages:
        losses = []
        for _ in range(REPEAT_FACTOR):
            train_X_cur = train_X.sample(frac=p)
            train_Y_cur = train_Y.reindex_like(train_X_cur)
            model.fit(train_X_cur.to_numpy(), train_Y_cur.to_numpy())
            losses.append(model.loss(test_X, test_Y))
        means.append(np.mean(np.array(losses)))
        deviations.append(np.std(np.array(losses)))
    means = np.array(means)
    deviations = np.array(deviations)
    fig = go.Figure([go.Scatter(x=percentages * 100, y=means, name="Mean Prediction", mode="markers+lines"),
                     go.Scatter(x=percentages * 100, y=means + 2 * deviations, name="Upper confidence Prediction", mode="markers+lines"),
                     go.Scatter(x=percentages * 100, y=means - 2 * deviations, name="Lower confidence bound", mode="markers+lines")],
                    layout=go.Layout(title="Prediction of Loss values as a function of training data size",
                                     xaxis=dict(title="Percentage of full training data"),
                                     yaxis=dict(title="Loss value")))
    fig.show()
