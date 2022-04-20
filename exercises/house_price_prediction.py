from IMLearn.utils import split_train_test
from IMLearn.learners.regressors import LinearRegression

from typing import NoReturn
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio
pio.templates.default = "simple_white"
from IMLearn.utils import split_train_test
import IMLearn.learners.regressors.linear_regression as lin_reg


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
    df = pd.read_csv (filename).dropna().drop_duplicates()

    columns_to_drop=["id","long","lat"]
    for i in columns_to_drop:
        df = df.drop(i ,axis = 1)

    #drop invalid data
    arr_should_be_positive = ["price","sqft_lot","sqft_living","floors","sqft_living15","yr_built","sqft_lot15"]
    arr_should_be_non_negative = ["bathrooms","bedrooms","sqft_above","sqft_basement","yr_renovated"]

    for i in arr_should_be_positive:
        df=df[df[i]>0]

    for i in arr_should_be_non_negative:
        df=df[df[i]>=0]

    #check ranges
    df = df[df["waterfront"].isin([0,1])]
    df = df[df["view"].isin(range(5))]
    df = df[df["condition"].isin(range(1,6))]
    df = df[df["grade"].isin(range(1,14))]

    #create new column-check if the house recent renovated
    df["year_sold"]=pd.to_datetime(df["date"]).dt.year
    df["yr_renovated"]=df["yr_renovated"].astype(int)
    df["recent_renovated"]=np.where(df["year_sold"]-df["yr_renovated"] <=20,1,0)
    df = df.drop("date", axis=1)
    df = df.drop("yr_renovated", axis=1)

    df=pd.get_dummies(df, prefix="zip_code", columns=["zipcode"])

    prices_vec = df["price"]
    df = df.drop("price",axis=1)

    return df,prices_vec


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
    for f in X:
        p_corr = np.cov(X[f],y)[1,0] / (np.std(X[f]) * np.std(y))
        fig = px.scatter(x=X[f], y=y)
        fig.update_layout(title=f"Correlation between {f} and response. "f"Pearson Correlation: {p_corr}",
                          xaxis_title=f"{f} Values", yaxis_title="Price Values")
        fig.write_image(f"{output_path}/{f}.png")


if __name__ == '__main__':
    np.random.seed(0)
    # Question 1 - Load and preprocessing of housing prices dataset
    X,y = load_data('../datasets/house_prices.csv')

    # Question 2 - Feature evaluation with respect to response
    feature_evaluation(X, y, "../Q2_figs")

    # Question 3 - Split samples into training- and testing sets.
    train_x,train_y,test_x,test_y=split_train_test(X,y)

    # Question 4 - Fit model over increasing percentages of the overall training data
    # For every percentage p in 10%, 11%, ..., 100%, repeat the following 10 times:
    #   1) Sample p% of the overall training data
    #   2) Fit linear model (including intercept) over sampled set
    #   3) Test fitted model over test set
    #   4) Store average and variance of loss over test set
    # Then plot average loss as function of training size with error ribbon of size (mean-2*std, mean+2*std)
    percentages=np.arange(10,101,1)
    mean_loss_values=np.zeros(91)
    std_loss_values = np.zeros(91)
    for p in percentages:
        sum_mse=0
        p_loss_values=np.zeros(10)
        for i in range(0,10):
            train_x_p,train_y_p,test_x_p,test_y_p=split_train_test(train_x,train_y,(p/100))

            train_x_arr=train_x_p.to_numpy()
            train_y_arr = train_y_p.to_numpy()
            lin_reg=LinearRegression()
            lin_reg.fit(train_x_arr,train_y_arr)

            loss=lin_reg.loss(test_x.to_numpy(),test_y.to_numpy())
            p_loss_values[i]=loss

        mean_loss_values[p-10]=np.mean(p_loss_values)
        std_loss_values[p-10]=np.std(p_loss_values)

    fig=go.Figure()
    fig.update_layout(xaxis_title="p%", yaxis_title="Mean Loss")

    fig.add_scatter(name="Mean Loss of p%", x=percentages, y=mean_loss_values,
                    mode="markers+lines",marker=dict(color="green"))

    fig.add_scatter(name="Confidence Interval (+2)", x=percentages, y=mean_loss_values + 2 * std_loss_values,
                    mode="lines", marker=dict(color="grey"),fill="tonexty")

    fig.add_scatter(name="Confidence Interval (-2)", x=percentages, y=mean_loss_values - 2 * std_loss_values,
                    mode="lines",marker=dict(color="grey"),fill="tonexty")

    fig.show()
