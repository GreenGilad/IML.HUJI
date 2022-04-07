from IMLearn.utils import split_train_test
from IMLearn.learners.regressors import LinearRegression

from typing import NoReturn
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio
pio.templates.default = "simple_white"
import os


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

    df.fillna(0, inplace=True)
    df[df <= 0] = 0.1

    zip_codes = pd.get_dummies(df.pop('zipcode'))
    df = pd.concat([df, zip_codes], axis=1)


    df['total_sqft'] = df['sqft_living'] + df['sqft_lot'] + df['sqft_above'] + df['sqft_basement']
    df['room_size'] = df['total_sqft'] / (df['bedrooms'] + df['bathrooms'])
    df['floors'] = df['floors'].astype(float)

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
    sigmaX = X.std(axis=0)
    sigmay = y.std()
    pearson = X.apply(lambda col: y.cov(col)) / (sigmaX*sigmay)
    for feature in pearson.index:
        plot = px.scatter(x=X[feature],
                          y=y,
                          labels={"x": str(feature), "y": "Price (USD)"},
                          title=f"<b>Pearson Correlation by Feature</b><br>{feature} correlation={pearson[feature]}")
        plot.show()
        plot.write_image(output_path + f"/PearsonPlot_{feature}.png")


if __name__ == '__main__':
    np.random.seed(0)
    data_filepath = "/Users/natandavids/IML/IML.HUJI/datasets/house_prices.csv"
    plot_dir = "ex2_plots"
    # Question 1 - Load and preprocessing of housing prices dataset
    X, y = load_data(data_filepath)

    # Question 2 - Feature evaluation with respect to response

    if not os.path.exists(plot_dir):
        os.mkdir(plot_dir)
    feature_evaluation(X, y, plot_dir)

    # Question 3 - Split samples into training- and testing sets.

    train_X, train_y, test_X, test_y = split_train_test(X, y)

    # Question 4 - Fit model over increasing percentages of the overall training data
    # For every percentage p in 10%, 11%, ..., 100%, repeat the following 10 times:
    #   1) Sample p% of the overall training data
    #   2) Fit linear model (including intercept) over sampled set
    #   3) Test fitted model over test set
    #   4) Store average and variance of loss over test set
    # Then plot average loss as function of training size with error ribbon of size (mean-2*std, mean+2*std)

    percent_range = 10, 101
    n_trials = 10
    confidence_weight = 2
    plot_filename = "/Average_Loss.png"


    avgloss = np.zeros(percent_range[1])
    std = np.zeros(percent_range[1])

    train_X['response'] = train_y

    for p in range(*percent_range):
        loss = np.empty(n_trials)
        for i in range(n_trials):
            data = train_X.sample(frac=p/100)
            response = data['response']
            LR = LinearRegression()
            LR._fit(data.drop('response', axis=1).to_numpy(), response.to_numpy())
            loss[i] = LR.loss(test_X.to_numpy(), test_y.to_numpy())
        avgloss[p], std[p] = loss.mean(), loss.std()

    x = np.arange(*percent_range)
    avgloss = avgloss[percent_range[0]:]
    std = std[percent_range[0]:]
    graph = px.scatter(x=x, y=avgloss, title="Average Loss By Percentage of Data Sampled", labels={'x': "percentage", 'y': "Avg Loss"})
    graph.add_scatter(x=x, y=avgloss - confidence_weight * std)
    graph.add_scatter(x=x, y=avgloss + confidence_weight * std)
    graph.show()
    graph.write_image(plot_dir + plot_filename)

