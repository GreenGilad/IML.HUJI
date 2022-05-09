from typing import NoReturn

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio

from IMLearn.learners.regressors import linear_regression
from IMLearn.utils.utils import split_train_test

pio.templates.default = "simple_white"


def load_data(filename: str) -> (pd.DataFrame, pd.Series):
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
    full_data = pd.read_csv(filename).dropna().drop_duplicates()

    features = full_data[
        ["zipcode", "bedrooms", "bathrooms", "sqft_living", "sqft_lot",
         "floors", "waterfront", "view", "condition", "price", "grade",
         "sqft_above", "sqft_basement", "yr_built", "yr_renovated",
         "sqft_living15", "sqft_lot15"]]

    for feature in ["bedrooms", "bathrooms", "waterfront", "view", "grade",
                    "sqft_above", "sqft_basement", "yr_renovated"]:
        features = features[features[feature] >= 0]

    for feature in ["price", "zipcode", "sqft_living", "sqft_lot", "floors",
                    "condition", "yr_built", "sqft_living15", "sqft_lot15"]:
        features = features[features[feature] > 0]

    dummy_zipcode = pd.get_dummies(features["zipcode"])
    features = pd.concat([features, dummy_zipcode], axis=1)
    features = features.drop("zipcode", axis=1)

    features["years_renovated_ago"] = features[
        ["yr_built", "yr_renovated"]].max(axis=1).apply(lambda x: 2022 - x)
    features.dropna(inplace=True)
    features["years_renovated_ago"] = features["years_renovated_ago"].astype(
        int)

    labels = features["price"]
    features = features.drop("price", axis=1)

    return features, labels


def feature_evaluation(X: pd.DataFrame, y: pd.Series,
                       output_path: str = ".") -> NoReturn:
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

    def get_corr(x: pd.Series, y: pd.Series) -> float:
        cov = np.cov(x, y)[0,1]
        x_std = x.std()
        y_std = y.std()
        if x_std == 0 or y_std == 0:
            return 0
        return cov / (x_std * y_std)

    for feature in X:
        if type(feature) == float:
            continue
        fig = go.Figure(
            [go.Scatter(x=X[feature], y=y, name=feature,
                        showlegend=True, mode='markers',
                        marker=dict(color="red", opacity=.7))],
            layout=go.Layout(title=(
                f"Corr between {feature} and price values. pearson corr = {get_corr(X[feature], y)}"),
                xaxis={"title": f"x - {feature} value"},
                yaxis={"title": "y - Response = price"},
                height=400))
        # fig.show()
        fig.write_image(f"{output_path}/{feature}.jpeg")


if __name__ == '__main__':
    np.random.seed(0)
    # Question 1 - Load and preprocessing of housing prices dataset
    X, y_true = load_data(r"..\datasets\house_prices.csv")

    # Question 2 - Feature evaluation with respect to response
    feature_evaluation(X, y_true,
                       r"C:\Users\User\Documents\CSE_2\IML\ex2\feature_eval")

    # Question 3 - Split samples into training- and testing sets.
    x_train, y_train, x_test, y_test = split_train_test(X, y_true)

    # Question 4 - Fit model over increasing percentages of the overall training data
    # For every percentage p in 10%, 11%, ..., 100%, repeat the following 10 times:
    #   1) Sample p% of the overall training data
    #   2) Fit linear model (including intercept) over sampled set
    #   3) Test fitted model over test set
    #   4) Store average and variance of loss over test set
    # Then plot average loss as function of training size with error ribbon of size (mean-2*std, mean+2*std)
    linreg = linear_regression.LinearRegression()
    mean, std = [], []
    for i in range(91):
        percent = (10 + i) / 100
        preds = []
        for j in range(10):
            new_train_X = x_train.sample(frac=percent, random_state=j)
            new_train_y = y_train.sample(frac=percent, random_state=j)
            linreg.fit(new_train_X.to_numpy(), new_train_y.to_numpy())
            preds.append(linreg.loss(x_test.to_numpy(), y_test.to_numpy()))
        preds = np.asarray(preds)
        mean.append(np.mean(preds))
        std.append(np.std(preds))

    mean = np.asarray(mean)
    std = np.asarray(std)
    x = np.linspace(10, 100, 91)
    fig = go.Figure(
        (go.Scatter(x=x, y=mean, mode="markers+lines",
                    name="Mean Prediction", line=dict(dash="dash"),
                    marker=dict(color="green", opacity=.7), showlegend=True
                    ),
         go.Scatter(x=x, y=mean - 2 * std, fill=None,
                    mode="lines", line=dict(color="lightgrey"),
                    showlegend=False),
         go.Scatter(x=x, y=mean + 2 * std, fill='tonexty',
                    mode="lines", line=dict(color="lightgrey"),
                    showlegend=False),))
    # title = "Loss and std as function of percentages of training set"
    fig.update_layout(title=dict({'text': "Loss and std as function of "
                                          "percentages of training set"}),
                      xaxis={"title": f"x - percentage of data"},
                      yaxis={"title": "y - mean value"}
                      ).show()
