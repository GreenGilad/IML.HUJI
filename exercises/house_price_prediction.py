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
    full_data = pd.read_csv(filename).dropna().drop_duplicates()
    full_data = full_data.drop(full_data.index[full_data["price"] < 0])
    full_data["date"] = full_data["date"].str[:4]
    full_data = full_data.drop(full_data.index[full_data["date"] == '0'])
    full_data = full_data.drop(full_data.index[full_data["bedrooms"] <= 0])
    full_data = full_data.drop(full_data.index[full_data["bathrooms"] <= 0])
    full_data = full_data.drop(full_data.index[full_data["sqft_living"] <= 0])
    full_data = full_data.drop(full_data.index[full_data["sqft_lot"] <= 0])
    full_data = full_data.drop(full_data.index[full_data["floors"] <= 0])
    full_data = full_data.drop(full_data.index[full_data["condition"] <= 0])
    full_data = full_data.drop(full_data.index[full_data["grade"] <= 0])
    full_data = full_data.drop(full_data.index[full_data["sqft_above"] < 0])
    labels = full_data["price"]
    full_data = full_data.drop(columns=["lat", "long", "id", "price", "zipcode"])
    full_data = pd.get_dummies(full_data)
    return full_data, labels


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
    for col_name, col_val in X.iteritems():
        corr = X[col_name].cov(y) / (np.std(col_val) * np.std(y))
        fig = go.Figure([go.Scatter(x=col_val, y=y, mode='markers')],
                        layout=go.Layout(title=f"Correlation between the {str(col_name)} and Y is: {corr}",
                                         xaxis_title=f"{str(col_name)}",
                                         yaxis_title="Price"))
        fig.write_image(output_path + fr"\Q2\{str(col_name)}_graphs.png")


if __name__ == '__main__':
    np.random.seed(0)
    # Question 1 - Load and preprocessing of housing prices dataset
    features, response = load_data('../datasets/house_prices.csv')

    # Question 2 - Feature evaluation with respect to response
    feature_evaluation(features, response)

    # Question 3 - Split samples into training- and testing sets.
    q3_train_X, q3_train_y, q3_test_X, q3_test_y = split_train_test(features, response)

    # Question 4 - Fit model over increasing percentages of the overall training data
    # For every percentage p in 10%, 11%, ..., 100%, repeat the following 10 times:
    #   1) Sample p% of the overall training data
    #   2) Fit linear model (including intercept) over sampled set
    #   3) Test fitted model over test set
    #   4) Store average and variance of loss over test set
    # Then plot average loss as function of training size with error ribbon of size (mean-2*std, mean+2*std)
    mean_loss, std_loss = [], []
    x=np.arange(10, 101, 1).astype(int)
    for p in x:
        temp_pred = []
        for _ in range(10):
            train_X, train_y, temp1, temp2 = split_train_test(q3_train_X, q3_train_y, train_proportion=p/100)
            model = LinearRegression()
            model.fit(train_X, train_y)
            temp_pred.append(model.loss(q3_test_X, q3_test_y))
        mean_loss.append(np.mean(temp_pred))
        std_loss.append(np.std(temp_pred))
    mean_loss, std_loss = np.array(mean_loss), np.array(std_loss)
    go.Figure([go.Scatter(x=x, y=mean_loss, mode="markers+lines", name="Mean Loss",
                          line=dict(dash="dash"), marker=dict(color="green", opacity=.7)),
               go.Scatter(x=x, y=mean_loss - 2 * std_loss, fill='tonexty', mode="lines",
                          line=dict(color="lightgrey"), showlegend=False),
               go.Scatter(x=x, y=mean_loss + 2 * std_loss, fill='tonexty', mode="lines",
                          line=dict(color="lightgrey"), showlegend=False)],
              layout=go.Layout(title=f"Average Loss as function of training size",
                               xaxis_title=f"Percent",
                               yaxis_title="Average Loss")).show()
