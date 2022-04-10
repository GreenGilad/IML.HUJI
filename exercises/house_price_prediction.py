from typing import NoReturn

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.io as pio

import IMLearn.metrics.loss_functions as lsfnc


import IMLearn.learners.regressors.linear_regression as lreg

from IMLearn import utils as utl

pio.templates.default = "simple_white"


def filter_data_set__values(df):
    """
    removes all non positive values for some relevant features
    such as price, sqft, etc...
    :param df: panda df
    :return: processed df
    """
    df.drop(df[(df.bathrooms <= 0) | (df.price <= 0)].index | (df.bedrooms <= 0))
    df.drop(df[(df.sqft_living <= 1000) | (df.sqft_lot <= 1500) | (df.floors <= 0)].index)

    zipcode_array = set()
    for i in df["zipcode"].unique():
        zipcode_array.add(i)

    for zipcode in zipcode_array:
        df[zipcode] = np.zeros(np.shape(df)[0])

    for ind in df.index:
        zipcode = df["zipcode"][ind]
        df.at[ind, zipcode] = 1




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
    filter_data_set__values(df)

    df.pop("date")
    df.pop("id")
    df.pop("zipcode")
    df.pop("lat")
    df.pop("long")


    price = df.pop('price')
    return df, price



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

    features = X.columns[0:15]
    for f in features:
        arr1 = X[f].to_numpy()
        arr2 = y.to_numpy()
        covMat = np.cov(arr1, arr2)
        stdMul = np.std(X[f]) * np.std(y)
        pearson_correlation = covMat[1][0] / stdMul
        title = "Price as a function of " + str(f) + "pearson Correlation is " + str(pearson_correlation)

        plt.scatter(X[f], y)
        plt.xlabel(str(f))
        plt.ylabel('Price')
        plt.title(label=title)
        plt.savefig(output_path + "/" + str(f)+".png", dpi=100, bbox_inches='tight')
        plt.clf()
        plt.cla()
        plt.close()


if __name__ == '__main__':
    np.random.seed(0)
    # Question 1 - Load and preprocessing of housing prices dataset
    df, price = load_data('../datasets/house_prices.csv')

    # Question 2 - Feature evaluation with respect to response
    feature_evaluation(df, price)
    # Question 3 - Split samples into training- and testing sets.
    x_train, y_train, x_test, y_test = utl.split_train_test(df, price, 0.75)

    # Question 4 - Fit model over increasing percentages of the overall training data
    # For every percentage p in 10%, 11%, ..., 100%, repeat the following 10 times:
    #   1) Sample p% of the overall training data
    #   2) Fit linear model (including intercept) over sampled set
    #   3) Test fitted model over test set
    #   4) Store average and variance of loss over test set
    # Then plot average loss as function of training size with error ribbon of size (mean-2*std, mean+2*std)
    number_of_samples = x_train.shape[0]

    mse_lst = []
    std_lst = []

    numOfiterations = 10
    for percentage in range(10, 100):
        curr_mse_lst = []
        evaluate_val = 0
        for fit in range(0, numOfiterations):
            x_percent_train = x_train.sample(frac=percentage/100)
            y_percent_train = y_train[x_percent_train.index]

            lin_reg = lreg.LinearRegression(True)
            lin_reg.fit(x_percent_train.values, y_percent_train)

            curr_error = lin_reg.loss(x_test.values, y_test.values)
            curr_mse_lst.append(curr_error)
            evaluate_val += curr_error

        mse_lst.append(evaluate_val/numOfiterations)
        std_lst.append(np.std(curr_mse_lst))

    std_lst = np.multiply(std_lst, 2)

    plt.plot(range(10, 100), mse_lst)
    plt.fill_between(range(10, 100), mse_lst + std_lst, np.subtract(mse_lst, std_lst), alpha=0.5)
    plt.fill_between(range(10, 100), np.subtract(mse_lst, std_lst), mse_lst + std_lst, color='gray', alpha=0.5)
    plt.title("MSE values as function of p% - mean loss")
    plt.xlabel('Percentage (p%)')
    plt.ylabel('MSE')
    plt.show()

