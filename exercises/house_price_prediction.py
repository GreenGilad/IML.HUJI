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
    # read csv
    dff = pd.read_csv(filename)
    # remove nans
    dff.dropna(inplace=True)

    # remove negative prices
    indx = dff[dff['price'] < 0].index
    dff.drop(indx, inplace=True)

    # remove negative sqft_living
    indx = dff[dff['sqft_living'] < 0].index
    dff.drop(indx, inplace=True)

    # remove negative sqft_lot
    indx = dff[dff['sqft_lot'] < 0].index
    dff.drop(indx, inplace=True)

    # remove negative sqft_living15
    indx = dff[dff['sqft_living15'] < 0].index
    dff.drop(indx, inplace=True)

    # remove negative sqft_lot15
    indx = dff[dff['sqft_lot15'] < 0].index
    dff.drop(indx, inplace=True)

    # remove wrong zipccode
    ind = dff[dff['zipcode'] == 0].index
    dff.drop(ind, inplace=True)

    # make dummies for zipcode
    arr = dff['zipcode']
    arr = arr.astype(int)
    dummy_df = pd.get_dummies(arr)
    dff = pd.concat([dummy_df, dff], axis=1)

    # make dummies for dates by year
    lst = list(dff['date'])
    n_lst = [int(str(x)[:4]) for x in lst]
    dff['date'] = n_lst
    arr = dff['date']
    dummy_df = pd.get_dummies(arr)
    dff = pd.concat([dummy_df, dff], axis=1)

    # make new col of house age when sold
    dff['age_at_sale'] = dff['date'] - dff['yr_built']

    # add years from last renovation
    renovation_indx = dff[dff['yr_renovated'] == 0].index
    dff['yrs_since_renovation'] = dff['date'] - dff['yr_renovated']
    dff.loc[renovation_indx, 'yrs_since_renovation'] = dff['date'] - dff['yr_built']

    # size with respect to neighbours
    dff['living_space_ratio'] = dff.apply(lambda row: row.sqft_living / row.sqft_living15, axis=1)
    dff['lot_space_ratio'] = dff.apply(lambda row: row.sqft_lot / row.sqft_lot15, axis=1)

    # separte label and features
    label = dff['price']
    dff.drop(columns=['price', 'lat', 'long', 'zipcode', 'date', 'yr_renovated', 'id', 2014], inplace=True)
    res = (dff, label)
    return res


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
    # create plots & pearson
    sd_y = np.sqrt(np.var(y))
    prsn_matrix = []
    for i in X.columns:
        sd_x = np.sqrt(np.var(X[i]))
        covv = np.cov(X[i], y)[0][1]
        pearson = covv / (sd_x * sd_y)
        prsn_matrix.append([i, pearson])
        fig = px.scatter(x=X[i], y=y, title='price & ' + str(i) + '. pearson corr: ' + str(round(pearson,4)), labels={'x': str(i), 'y': 'price'})
        fig.write_html(output_path + r'\plot_' + str(i) + '.html')
    prsn_matrix = pd.DataFrame(prsn_matrix, columns=['feature', 'pearson coef'])
    prsn_matrix.to_excel('pearson.xlsx')


if __name__ == '__main__':
    np.random.seed(0)
    # Question 1 - Load and preprocessing of housing prices dataset
    X, y = load_data(r'C:\Users\Rafi Levy\Documents\GitHub\IML.HUJI\datasets\house_prices.csv')

    # Question 2 - Feature evaluation with respect to response
    feature_evaluation(X, y, 'scatter_plots')

    # Question 3 - Split samples into training- and testing sets.
    train_x, train_y, test_x, test_y = split_train_test(X, y, 0.75)

    # Question 4 - Fit model over increasing percentages of the overall training data
    res = [[], [], []]
    length = len(train_x)
    for p in range(10, 101, 1):
        percentage_p = int(length * (p / 100))
        loss_lst = []
        for i in range(10):
            X_sample = train_x.sample(percentage_p)
            y_sample = y[X_sample.index]
            lin = LinearRegression()
            lin.fit(X_sample.to_numpy(), y_sample.to_numpy())
            loss_lst.append(lin.loss(test_x.to_numpy(), test_y.to_numpy()))
        res[0].append(round(p / 100, 2))
        res[1].append(np.mean(np.array(loss_lst)))
        res[2].append(np.sqrt(np.var(np.array(loss_lst))))
    res = np.array(res)

    figg = go.Figure([go.Scatter(x=res[0], y=res[1] - 2 * res[2], fill=None, mode="lines", line=dict(color="lightgrey"),
                                 showlegend=False),
                      go.Scatter(x=res[0], y=res[1] + 2 * res[2], fill='tonexty', mode="lines",
                                 line=dict(color="lightgrey"), showlegend=False),
                      go.Scatter(x=res[0], y=res[1], mode="markers+lines", marker=dict(color="black", size=1),
                                 showlegend=False)],
                     layout=go.Layout(
                         title=r"mean and sd of prediction error as function of p % of test data",
                         height=500))
    figg.write_html('plot.html', auto_open=True)

    # For every percentage p in 10%, 11%, ..., 100%, repeat the following 10 times:
    #   1) Sample p% of the overall training data
    #   2) Fit linear model (including intercept) over sampled set
    #   3) Test fitted model over test set
    #   4) Store average and variance of loss over test set
    # Then plot average loss as function of training size with error ribbon of size (mean-2*std, mean+2*std)
