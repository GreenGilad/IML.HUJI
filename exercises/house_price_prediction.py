from IMLearn.utils import split_train_test
from IMLearn.learners.regressors import LinearRegression

from typing import NoReturn
import numpy as np
import pandas as pd
import plotly.io as pio
import matplotlib.pyplot as plt
from IMLearn import utils

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
    df = pd.read_csv(filename)
    df = df.dropna()

    df = df.drop(df[(df.id <= 0) | (df.price < 0) | (df.bedrooms <= 0) |
                    (df.sqft_living < 1000) | (df.sqft_lot < 1000) |
                    (df.sqft_living15 < 1000) | (df.sqft_lot15 < 1000)].index)
    df.pop('date')
    df.pop('id')
    df.pop("lat")
    df.pop("long")

    response = df.pop('price')
    return df, response


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
    for feature in X.columns:
        varx = np.var(X[feature].array)
        pearson_correlation = np.cov(X[feature].array, y
                                     ) / (np.power(varx * np.var(y), 0.5))

        plt.scatter(X[feature], y)
        plt.xlim(min(X[feature]), max(X[feature]))
        plt.xlabel(str(feature))
        plt.ylabel('Price')
        plt.title(label="Price as a function of " + str(feature) + "\n" +
                        " with Pearson correlation of: "
                        + str(pearson_correlation[0][1]))
        plt.savefig(output_path + "/" + str(feature), dpi=100,
                    bbox_inches='tight')
        plt.close()


if __name__ == '__main__':
    np.random.seed(0)
    # Question 1 - Load and preprocessing of housing prices dataset
    data, response = load_data("../datasets/house_prices.csv")

    # Question 2 - Feature evaluation with respect to response
    feature_evaluation(data, response)

    # Question 3 - Split samples into training- and testing sets.
    train_X, train_y, test_X, test_y = utils.split_train_test(data, response)

    # Question 4 - Fit model over increasing percentages of the overall training data
    # For every percentage p in 10%, 11%, ..., 100%, repeat the following 10 times:
    #   1) Sample p% of the overall training data
    #   2) Fit linear model (including intercept) over sampled set
    #   3) Test fitted model over test set
    #   4) Store average and variance of loss over test set
    # Then plot average loss as function of training size with error ribbon of size (mean-2*std, mean+2*std)
    plt.cla()
    lin = LinearRegression(False)
    vals = 0
    mean_loss, standard_deviations = [], []
    for p in range(10, 100):
        var_arr = []
        for rep in range(0, 10):
            tr_X, tr_y, tst_X, tst_y = utils.split_train_test(train_X, train_y,
                                                              p / 100)
            lin.fit(tr_X.to_numpy(), tr_y.to_numpy())
            tmp = lin.loss(test_X.to_numpy(), test_y.to_numpy())
            var_arr.append(tmp)
        mean_loss.append(np.mean(var_arr))
        standard_deviations.append(np.std(var_arr))
    standard_deviations = np.multiply(standard_deviations, 2)

    plt.plot(np.arange(10, 100), mean_loss)
    plt.fill_between(np.arange(10, 100),
                     np.subtract(mean_loss, standard_deviations),
                     mean_loss + standard_deviations, alpha=0.2)
    plt.title("Mean loss as function of p%")
    plt.xlabel('Percentage (p%)')
    plt.ylabel('Mean loss')
    plt.show()
