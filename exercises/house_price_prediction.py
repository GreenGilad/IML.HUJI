import go as go

from IMLearn.utils import split_train_test
from IMLearn.learners.regressors import LinearRegression

from typing import NoReturn
from IMLearn.utils.utils import split_train_test
from IMLearn.learners.regressors import linear_regression
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
    df = pd.read_csv(filename).dropna().drop_duplicates()
    # remove unnecessary columns
    for label in ["date", "long", "lat", "id"]:
        df = df.drop(label, 1)

    # remove the lines that don't fit the condition
    for label in ["price", "sqft_living", "sqft_lot", "sqft_above", "yr_built", "sqft_living15", "sqft_lot15"]:
        df = df[df[label] > 0]

    for label in ["bathrooms", "floors", "sqft_basement", "yr_renovated"]:
        df = df[df[label] >= 0]

    df = df[df["waterfront"].isin([0, 1]) & df["view"].isin(range(5)) &
            df["condition"].isin(range(1, 6)) & df["grade"].isin(range(1, 15))]

    df["decade_built"] = (df["yr_built"] / 10).astype(int)
    df.drop("yr_built", 1)
    df["zipcode"] = (df["zipcode"] / 10).astype(int)

    df["yr_renovated"] = df["yr_renovated"].astype(int)
    df["recently_renovated"] = np.where(df["yr_renovated"] >= 2010, 1, 0)
    df = df.drop("yr_renovated", 1)

    df = pd.get_dummies(df, prefix='zipcode_', columns=['zipcode'])
    df = pd.get_dummies(df, prefix='decade_built_', columns=['decade_built'])

    # delete rare inputs
    df = df[df["bedrooms"] < 20]
    df = df[df["sqft_lot"] < 1250000]
    df = df[df["sqft_lot15"] < 500000]
    df.insert(0, 'intercept', 1, True)
    return df.drop("price", 1), df.price


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
    # y_np = y.to_numpy()
    for label, content in X.items():
        x = X[label]
        cov_x_y = np.cov(x, y)[0, 1]
        std_y = y.std()
        std_x = x.std()

        person_correlation = 0
        if (x.std() != 0) and (y.std() != 0):
            person_correlation = cov_x_y / (std_x * std_y)

        go.Figure([go.Scatter(x=x, y=y, mode='markers',
                              line=dict(width=4, color="rgb(204,68,83)"))],
                  layout=go.Layout(title="person_correlation is : " + str(person_correlation),
                                   xaxis_title=label,
                                   yaxis_title="price",
                                   height=500)).show()
            #write_image(output_path + str(label) + ".png")


if __name__ == '__main__':
    np.random.seed(0)
    # Question 1 - Load and preprocessing of housing prices dataset
    X, y_true = load_data(r"..\datasets\house_prices.csv")

    # Question 2 - Feature evaluation with respect to response
    feature_evaluation(X, y_true)

    # Question 3 - Split samples into training- and testing sets.
    train_x, train_y, test_x, test_y = split_train_test(X, y_true)

    # Question 4 - Fit model over increasing percentages of the overall training data
    # For every percentage p in 10%, 11%, ..., 100%, repeat the following 10 times:
    #   1) Sample p% of the overall training data
    #   2) Fit linear model (including intercept) over sampled set
    #   3) Test fitted model over test set
    #   4) Store average and variance of loss over test set
    # Then plot average loss as function of training size with error ribbon of size (mean-2*std, mean+2*std)

    mean_loss = []
    std_loss = []
    for p in range(10, 101):
        pred_loss = []
        for i in range(1, 10):
            x_p = train_x.sample(frac=p / 100)
            y_p = train_y.reindex_like(x_p)
            l_r = LinearRegression(False)
            l_r.fit(x_p.to_numpy(), y_p.to_numpy())
            pred_loss.append(l_r.loss(test_x.to_numpy(), test_y.to_numpy()))

        pred_loss1 = np.asarray(pred_loss)
        mean_loss.append(np.mean(pred_loss1))
        std_loss.append(np.std(pred_loss1))

    x = np.linspace(10, 100, 91)
    mean_loss1 = np.asarray(mean_loss)
    std_loss1 = np.asarray(std_loss)

    fig = go.Figure(
        (go.Scatter(x=x, y=mean_loss1, mode="markers+lines", name="mean loss predict", line=dict(dash="dash"),
                    marker=dict(color="green", opacity=.7),
                    ),
         go.Scatter(x=x, y=mean_loss1 - (2 * std_loss1), fill=None, mode="lines", line=dict(color="lightgrey"),
                    showlegend=False),
         go.Scatter(x=x, y=mean_loss1 + 2 * std_loss1, fill='tonexty', mode="lines", line=dict(color="lightgrey"),
                    showlegend=False),))
    title = "Fit a linear regression model over increasing percentages of the training set and " \
            "measure the loss over the test set"
    fig.update_layout(title=dict({'text': title})).show()
