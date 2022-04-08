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
    df = pd.read_csv(filename)
    # remove infecting features
    df.drop(["id", "long", "date", "lat"], axis=1, inplace=True)

    # remove invalid rows:
    df.drop(df[df['price'] <= 0].index, inplace=True)
    df.drop(df[df['sqft_living'] <= 0].index, inplace=True)
    df.drop(df[df['sqft_above'] <= 0].index, inplace=True)
    df.drop(df[df['yr_built'] <= 0].index, inplace=True)
    df.drop(df[df['sqft_living15'] <= 0].index, inplace=True)
    df.drop(df[df['sqft_lot'] <= 0].index, inplace=True)
    df.drop(df[df['sqft_lot15'] <= 0].index, inplace=True)

    # remove houses without rooms or too many rooms
    df.drop(df[df['bathrooms'] < 0].index, inplace=True)
    df.drop(df[df['bedrooms'] < 0].index, inplace=True)
    df.drop(df[df['bedrooms'] >= 15].index, inplace=True)

    # remove house that seems to be to much big
    df.drop(df[df['sqft_lot'] > 982998].index, inplace=True)

    # add feature: has been Recently_renovated
    conditions1 = [
        (df['yr_built'] > 2005) | (df['yr_renovated'] > 2005),
        (df['yr_built'] <= 2005) & (df['yr_renovated'] <= 2005)]
    df['Recently_renovated_or_built'] = np.select(conditions1, [1, 0])
    df.drop(["yr_built", "yr_renovated"], axis=1, inplace=True)

    # add feature of maximum people in the house
    mean_sqft_above = np.ceil(df['sqft_above'].mean())
    mean_bedrooms = df['bedrooms'].mean()
    conditions2 = [
        df['sqft_above'] > mean_sqft_above,
        df['sqft_above'] <= mean_sqft_above]
    values2 = [df['bedrooms'] +
               np.ceil((1 / (mean_sqft_above // mean_bedrooms))
                       * (df['sqft_above'] - mean_sqft_above)),
               df['bedrooms']]
    df['number_of_people'] = np.select(conditions2, values2)

    # convert categorical "zipcode" to numerical
    df["zipcode"] = df["zipcode"].astype(int)
    df = pd.get_dummies(df, prefix='zipcode', columns=['zipcode'])

    # replace nan values with the mean of this feature
    for column in df:
        mean = df[column].mean()
        df[column].replace(np.nan, mean, inplace=True)

    # add intercept column
    df.insert(0, "intercept", 1, allow_duplicates=True)
    return df.drop(["price"], axis=1), df['price']


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
    # remove irrelevant columns
    X = X.drop("intercept", 1)
    X = X.loc[:, ~(X.columns.str.contains('^zipcode'))]

    for feature in X:
        cov_x_y = np.cov(X[feature], y)
        cor_x_y = cov_x_y[0][1] / np.sqrt(cov_x_y[0][0] * cov_x_y[1][1])
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=X[feature],
            y=y,
            mode="markers"
        ))

        title = "Feature name: {feature}\n Pearson correlation: {cor}"
        fig.update_layout(
            title=title.format(feature=feature, cor=cor_x_y),
            xaxis_title="feature values",
            yaxis_title="response values"
        )
        fig.write_image(output_path + feature + ".png")


"""
NOTE: the figure is not shown if trying to run it regularly, But the printing
at the end of the function is executed. The figure is shown while running in debug
"""
if __name__ == '__main__':
    np.random.seed(0)
    # Question 1 - Load and preprocessing of housing prices dataset
    X, y = load_data('house_prices.csv')

    # Question 2 - Feature evaluation with respect to response
    feature_evaluation(X, y)

    # Question 3 - Split samples into training- and testing sets.
    train_x, train_y, test_x, test_y = split_train_test(X, y)

    # Question 4 - Fit model over increasing percentages of the overall training data
    # For every percentage p in 10%, 11%, ..., 100%, repeat the following 10 times:
    #   1) Sample p% of the overall training data
    #   2) Fit linear model (including intercept) over sampled set
    #   3) Test fitted model over test set
    #   4) Store average and variance of loss over test set
    # Then plot average loss as function of training size with error ribbon of size (mean-2*std, mean+2*std)
    lr = LinearRegression(True)
    train_x.insert(0, "response", np.array(train_y), allow_duplicates=True)
    mean_losses = []
    std_losses = []
    for p in range(10, 101):
        inner_loss = []
        for i in range(10):
            sample = train_x.sample(frac=p/float(100))
            lr.fit(sample.drop(["response"], 1), sample.response)
            inner_loss.append(lr.loss(np.array(test_x), np.array(test_y)))
        mean_losses.append(np.mean(np.array(inner_loss)))
        std_losses.append(np.std(np.array(inner_loss)))
    percents = np.arange(10, 101)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=percents, y=mean_losses, mode="markers+lines", name="Mean loss",
                             line=dict(dash="dash"), marker=dict(color="green", opacity=.7)))
    fig.add_trace(go.Scatter(x=percents, y=np.array(mean_losses) - np.array(std_losses) * 2, fill=None, mode="lines",
                             line=dict(color="lightgrey"), showlegend=False))
    fig.add_trace(go.Scatter(x=percents, y=np.array(mean_losses) + np.array(std_losses) * 2, fill='tonexty',
                    mode="lines", line=dict(color="lightgrey"), showlegend=False))
    fig.update_layout(
        title="Mean loss as a function of training set size",
        xaxis_title="Percents of training dataset",
        yaxis_title="Mean loss values"
    )
    fig.show()

    # prints it but don't show the figure. Although it is, if running in debug
    print("finish execution")



