from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn import datasets
from IMLearn.metrics import mean_square_error
from IMLearn.utils import split_train_test
from IMLearn.model_selection import cross_validate
from IMLearn.learners.regressors import PolynomialFitting, LinearRegression, RidgeRegression
from sklearn.linear_model import Lasso

from utils import *
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def select_polynomial_degree(n_samples: int = 100, noise: float = 5):
    """
    Simulate data from a polynomial model and use cross-validation to select the best fitting degree

    Parameters
    ----------
    n_samples: int, default=100
        Number of samples to generate

    noise: float, default = 5
        Noise level to simulate in responses
    """
    # Question 1 - Generate dataset for model f(x)=(x+3)(x+2)(x+1)(x-1)(x-2) + eps for eps Gaussian noise
    # and split into training- and testing portions

    X = np.linspace(-1.2, 2, n_samples)
    eps = np.random.normal(0, noise, n_samples)
    given_polynomial = lambda x: (x+3)*(x+2)*(x+1)*(x-1)*(x-2)
    y_without_noise = given_polynomial(X)
    y = y_without_noise + eps
    train_X, train_y, test_X, test_y = split_train_test(pd.DataFrame(X), pd.Series(y), 2/3)
    train_x_graph = np.array(train_X).flatten().tolist()
    train_y_graph = np.array(train_y).flatten().tolist()
    test_x_graph = np.array(test_X).flatten().tolist()
    test_y_graph = np.array(test_y).flatten().tolist()

    if noise !=0 :
        go.Figure([go.Scatter(x=train_x_graph, y=train_y_graph, mode='markers',  name="$train$"),
                   go.Scatter(x=test_x_graph, y=test_y_graph, mode='markers', name="$test$"),
                   go.Scatter(x=X, y=y_without_noise, mode='markers', name="$data_without_noise$")],
                  layout=go.Layout(title="Data layout as a function of training and test" + " --noise: "
                                         + str(noise) + " n_samples: " + str(n_samples) + "--",
                                   xaxis_title="$m\\text{ label}$",
                                   yaxis_title="sample",
                                   height=600)).show()
    else:
        go.Figure([go.Scatter(x=train_x_graph, y=train_y_graph, mode='markers',  name="$train$"),
                   go.Scatter(x=test_x_graph, y=test_y_graph, mode='markers', name="$test$")],
                  layout=go.Layout(title="Data layout as a function of training and test" + " --noise: "
                                         + str(noise) + " n_samples: " + str(n_samples) + "--",
                                   xaxis_title="$m\\text{ label}$",
                                   yaxis_title="sample",
                                   height=600)).show()
    # Question 2 - Perform CV for polynomial fitting with degrees 0,1,...,10
    average_training = []
    validation_errors = []
    for k in range(11):
        estimator = PolynomialFitting(k)
        train_loss, val_loss = cross_validate(estimator, train_X.to_numpy(), train_y.to_numpy(), mean_square_error)
        average_training.append(train_loss)
        validation_errors.append(val_loss)
    go.Figure([go.Scatter(x=list(range(11)), y=average_training, mode='markers + lines', name="$average_training$"),
               go.Scatter(x=list(range(11)), y=validation_errors, mode='markers + lines', name="$validation_errors$")],
              layout=go.Layout(title="Train loss and validation loss as function of k" + " --noise: "
                                     + str(noise) + " n_samples: " + str(n_samples) + "--",
                               xaxis_title="$\\text{k : polynomial degree}$",
                               yaxis_title="$\\text{losses}$",
                               height=700)).show()

    # Question 3 - Using best value of k, fit a k-degree polynomial model and report test error
    best_k = int(np.argmin(validation_errors))
    estimator = PolynomialFitting(best_k)
    estimator.fit(np.array(train_X).flatten(), np.array(train_y).flatten())
    test_error = estimator.loss(np.array(test_X).flatten(), np.array(test_y).flatten())
    print("--------- noise:", noise, ", n_samples:", n_samples, "---------")
    print("chosen k:", best_k)
    print("test error:", round(test_error, 2))



def select_regularization_parameter(n_samples: int = 50, n_evaluations: int = 500):
    """
    Using sklearn's diabetes dataset use cross-validation to select the best fitting regularization parameter
    values for Ridge and Lasso regressions

    Parameters
    ----------
    n_samples: int, default=50
        Number of samples to generate

    n_evaluations: int, default = 500
        Number of regularization parameter values to evaluate for each of the algorithms
    """
    # Question 6 - Load diabetes dataset and split into training and testing portions
    X, y = datasets.load_diabetes(return_X_y=True)
    X, y = np.array(X), np.array(y)
    indexes = np.random.choice(X.shape[0], n_samples)
    train_X = X[indexes]
    train_y = y[indexes]
    test_X = np.delete(X, indexes, axis=0)
    test_y = np.delete(y, indexes, axis=0)

    # Question 7 - Perform CV for different values of the regularization parameter for Ridge and Lasso regressions
    ridge_average_training = []
    ridge_validation_errors = []
    lasso_average_training = []
    lasso_validation_errors = []
    scale = np.linspace(0, 3, n_evaluations)
    for i in scale:
        ridge = RidgeRegression(i)
        ridge_train_loss, ridge_val_loss = \
            cross_validate(ridge, train_X, train_y, mean_square_error)
        ridge_average_training.append(ridge_train_loss)
        ridge_validation_errors.append(ridge_val_loss)

        lasso = Lasso(i)
        lasso_train_loss, lasso_val_loss = \
            cross_validate(lasso, train_X, train_y, mean_square_error)
        lasso_average_training.append(lasso_train_loss)
        lasso_validation_errors.append(lasso_val_loss)

    go.Figure([go.Scatter(x=scale, y=lasso_average_training, mode='markers + lines', name="$lasso_average_training$"),
               go.Scatter(x=scale, y=lasso_validation_errors, mode='markers + lines', name="$lasso_validation_errors$"),
               go.Scatter(x=scale, y=ridge_average_training, mode='markers + lines', name="$ridge_average_training$"),
               go.Scatter(x=scale, y=ridge_validation_errors, mode='markers + lines', name="$ridge_validation_errors$")
               ],
              layout=go.Layout(title="Train loss and validation loss in Lasso & Ridge as function of lambda",
                               xaxis_title="$\\text{lamda}$",
                               yaxis_title="$\\text{losses}$",
                               height=700)).show()

    go.Figure([go.Scatter(x=scale, y=ridge_average_training, mode='markers + lines', name="$ridge_average_training$"),
               go.Scatter(x=scale, y=ridge_validation_errors, mode='markers + lines', name="$ridge_validation_errors$")],
              layout=go.Layout(title="Train loss and validation loss in Rigde as function of lambda",
                               xaxis_title="$\\text{lamda}$",
                               yaxis_title="$\\text{losses}$",
                               height=700)).show()


    # Question 8 - Compare best Ridge model, best Lasso model and Least Squares model
    ridge_best_lamda = scale[int(np.argmin(ridge_validation_errors))]
    ridge = RidgeRegression(ridge_best_lamda)
    ridge.fit(train_X, train_y)
    ridge_test_error = ridge.loss(test_X, test_y)
    print("------------ ridge -----------------")
    print("chosen lamda:", ridge_best_lamda)
    print("test error:", ridge_test_error)

    lasso_best_lamda = scale[int(np.argmin(lasso_validation_errors))]
    lasso = Lasso(lasso_best_lamda)
    lasso.fit(train_X, train_y)
    lasso_test_error = mean_square_error(y, lasso.predict(X))
    print("------------ lasso -----------------")
    print("chosen lamda:", lasso_best_lamda)
    print("test error:", lasso_test_error)

    linear_regression = LinearRegression()
    linear_regression.fit(train_X, train_y)
    linear_regression_error = mean_square_error(y, linear_regression.predict(X))
    print("------------ Linear Regression -----------------")
    print("test error:", linear_regression_error)

if __name__ == '__main__':
    np.random.seed(0)
    select_polynomial_degree()
    select_polynomial_degree(100, 0)
    select_polynomial_degree(1500, 10)
    select_regularization_parameter()

