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
    #
    data = pd.read_csv(filename)
    # clean NA's
    data.dropna(inplace=True)

    # clean negative prices
    negative_price_ind = data[data['price'] < 0].index
    data.drop(negative_price_ind, inplace=True)

    # clean zero bedrooms
    zero_bedrooms_ind = data[data['bedrooms'] <= 0].index
    data.drop(zero_bedrooms_ind, inplace=True)

    # clean zero bathrooms
    zero_bathrooms_ind = data[data['bathrooms'] <= 0].index
    data.drop(zero_bathrooms_ind, inplace=True)

    # create dummies for year of sale
    date_by_year_only = [int(str(date)[:4]) for date in data['date']]
    data['date'] = date_by_year_only
    year_dummies = pd.get_dummies(data['date'])
    data = pd.concat([data, year_dummies], axis=1)

    # creat dummies for zip
    zip_dummies = pd.get_dummies(data['zipcode'])
    data = pd.concat([data, zip_dummies], axis=1)

    # creat new feature: 'age_at_sale'
    data['age_at_sale'] = data['date'] - data['yr_built']

    # creat new feature: 'years_from_renovation'
    data['years_from_renovation'] = data['date'] - data['yr_renovated']
    renovated_house_ind = data[data['yr_renovated'] == 0].index
    data.loc[renovated_house_ind, 'years_from_renovation'] = data['date'] - data['yr_built']

    # creat new feature: 'ratio_sqft_living'
    data['ratio_sqft_living'] = data['sqft_living'] / data['sqft_living15']

    # creat new feature: 'ratio_sqft_lot'
    data['ratio_sqft_lot'] = data['sqft_lot'] / data['sqft_lot15']
    # print(data)
    price = data['price']
    # drop all irrelevant features
    data.drop(columns=['id', 'date', 'yr_renovated', 'yr_built', 'zipcode', 'lat', 'long', 'price'], inplace=True) #2014,98001,'sqft_lot15','sqft_living15',
    #data.to_csv(r'C:\Users\elish\OneDrive\Desktop\test.csv')

    return (data, price)


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
    y_sd = (np.var(y)) ** 0.5
    for i in X.columns:
        # print(X[i])
        cov_ = np.cov(X[i], y)[0][1]
        feature_sd = (np.var(X[i])) ** 0.5
        # print(feature_sd)
        pearson = cov_ / (feature_sd * y_sd)
        print(str(i), pearson)
        fig = px.scatter(x=X[i], y=y, title='Pearson Correlation between ' + str(i) + ' and price. Pearson = ' + str(pearson),
                         labels={'y': 'price', 'x': str(i)})
        fig.write_html(output_path+ r'\pearson_plot_' + str(i) +'.html', auto_open=False)


if __name__ == '__main__':
    np.random.seed(0)
    # Question 1 - Load and preprocessing of housing prices dataset
    data_loaded = load_data(r"C:\Users\elish\OneDrive\Documents\GitHub\IML\IML.HUJI\datasets\house_prices.csv")
    # Question 2 - Feature evaluation with respect to response
    feature_evaluation(data_loaded[0], data_loaded[1],
                       r'C:\Users\elish\OneDrive\Documents\GitHub\IML\IML.HUJI\exercises\ex_2_plots')
    # Question 3 - Split samples into training- and testing sets.
    train_x, train_y, test_X, test_y = split_train_test(data_loaded[0], data_loaded[1], train_proportion=0.75)
    # Question 4 - Fit model over increasing percentages of the overall training data
    # For every percentage p in 10%, 11%, ..., 100%, repeat the following 10 times:
    std_of_mse_by_increasing_percentage = []
    mean_mse_by_increasing_percentage = []
    all_p = []
    for p in range(10, 101, 1):
        temp_n_of_train_set = int(p / 100 * len(train_x))
        all_p.append(p / 100)
        temp_mse_by_p = []
        for i in range(10):
            current_model = LinearRegression()
            #   1) Sample p% of the overall training data
            temp_features_train_set = pd.DataFrame.sample(self=train_x, n=temp_n_of_train_set)
            temp_y_train_set = train_y[temp_features_train_set.index]
            #   2) Fit linear model (including intercept) over sampled set
            current_model.fit(temp_features_train_set.to_numpy(), temp_y_train_set.to_numpy())
            #   3) Test fitted model over test set
            temp_mse_by_p.append(current_model.loss(test_X.to_numpy(), test_y.to_numpy()))
        #   4) Store average and variance of loss over test set
        mean_mse_by_increasing_percentage.append(np.mean(np.array(temp_mse_by_p)))
        std_of_mse_by_increasing_percentage.append(np.var(np.array(temp_mse_by_p)) ** 0.5)
    # Then plot average loss as function of training size with error ribbon of size (mean-2*std, mean+2*std)
    fig = go.Figure([go.Scatter(x=np.array(all_p), y=np.array(mean_mse_by_increasing_percentage) + 2 * np.array(
        std_of_mse_by_increasing_percentage), fill=None, mode="lines", line=dict(color="lightgrey"),
                                showlegend=False),
                     go.Scatter(x=np.array(all_p), y=np.array(mean_mse_by_increasing_percentage) - 2 * np.array(
                         std_of_mse_by_increasing_percentage), fill='tonexty', mode="lines",
                                line=dict(color="lightgrey"),
                                showlegend=False),
                     go.Scatter(x=np.array(all_p), y=np.array(mean_mse_by_increasing_percentage), mode="markers+lines",
                                marker=dict(color="black", size=1), showlegend=False)],
                    layout=go.Layout(
                        title=r"mse as function of the percentage of the data used for training the model",
                        height=500))
    fig.write_html(r'model_mse.html', auto_open=True)