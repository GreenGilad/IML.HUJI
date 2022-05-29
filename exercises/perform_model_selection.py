from __future__ import annotations

import numpy as np
from sklearn import datasets
from IMLearn.metrics import mean_square_error
from IMLearn.utils import split_train_test
from IMLearn.model_selection import cross_validate
from IMLearn.learners.regressors import PolynomialFitting, LinearRegression, RidgeRegression
from sklearn.linear_model import Lasso
from utils import *
import plotly.graph_objects as go


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
    data_fun = lambda x: (x + 3) * (x + 2) * (x + 1) * (x - 1) * (x - 2)
    noise_vec = np.random.normal(0, noise, n_samples)
    X = np.random.uniform(-1.2, 2, n_samples)
    y = data_fun(X)
    y_noise = y + noise_vec

    train_x, train_y, test_x, test_y = split_train_test(X, y_noise, 2 / 3)

    plot1 = go.Figure(layout=dict(title=dict(text='samples: ' + str(n_samples) + ' noise: ' + str(noise))))
    plot1.add_trace(go.Scatter(x=X, y=y, mode='markers', name='true_values'))
    plot1.add_trace(go.Scatter(x=train_x, y=train_y, mode='markers', name='train'))
    plot1.add_trace(go.Scatter(x=test_x, y=test_y, mode='markers', name='test'))

    plot1.write_html(str(noise) + '_noise_Q1.html', auto_open=False)

    # Question 2 - Perform CV for polynomial fitting with degrees 0,1,...,10
    loss_func = lambda y_true, y_pred: mean_square_error(y_true, y_pred)
    train_score_lst = []
    test_score_lst = []
    for i in range(11):
        my_poly = PolynomialFitting(i)
        train_s, test_s = cross_validate(my_poly, train_x, train_y, loss_func, 5)
        train_score_lst.append(train_s)
        test_score_lst.append(test_s)
    plot2 = go.Figure(layout=dict(title=dict(text='samples: ' + str(n_samples) + ' noise: ' + str(noise))))
    plot2.add_trace(go.Scatter(x=list(range(11)), y=train_score_lst, mode='lines+markers', name='avrg train_score'))
    plot2.add_trace(go.Scatter(x=list(range(11)), y=test_score_lst, mode='lines+markers', name='avrg test score'))
    plot2.write_html('str(noise)' + '_noise_Q2.html', auto_open=False)

    # Question 3 - Using best value of k, fit a k-degree polynomial model and report test error
    k_star = int(np.argmin(test_score_lst))
    my_pol = PolynomialFitting(k_star).fit(train_x, train_y).loss(test_x, test_y)
    print('noise:', noise, 'k:', k_star, 'error:', round(my_pol, 2))
    return


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
    train_x, train_y, test_x, test_y = split_train_test(X, y, n_samples / len(y))

    # Question 7 - Perform CV for different values of the regularization parameter for Ridge and Lasso regressions
    r_train_score_lst, r_test_score_lst, l_train_score_lst, l_test_score_lst = [], [], [], []
    loss_func = lambda y_true, y_pred: mean_square_error(y_true, y_pred)
    lam_range = np.linspace(0.000001, 0.5, n_evaluations)
    for i in lam_range:
        r_train_loss, r_test_loss = cross_validate(RidgeRegression(lam=i), train_x, train_y, loss_func)
        l_train_loss, l_test_loss = cross_validate(Lasso(alpha=i, max_iter=10000), train_x, train_y, loss_func)
        r_train_score_lst.append(r_train_loss)
        r_test_score_lst.append(r_test_loss)
        l_train_score_lst.append(l_train_loss)
        l_test_score_lst.append(l_test_loss)

    plot7 = go.Figure(layout=dict(title=dict(text='samples: ' + str(n_samples))))
    plot7.add_trace(go.Scatter(x=lam_range, y=r_train_score_lst, mode='lines',
                               name='ridge avrg train_score'))
    plot7.add_trace(go.Scatter(x=lam_range, y=r_test_score_lst, mode='lines',
                               name='ridge avrg test score'))
    plot7.add_trace(go.Scatter(x=lam_range, y=l_train_score_lst, mode='lines',
                               name='lasso avrg train_score'))
    plot7.add_trace(go.Scatter(x=lam_range, y=l_test_score_lst, mode='lines',
                               name='lasso avrg test score'))
    plot7.write_html('Q7.html', auto_open=True)

    # Question 8 - Compare best Ridge model, best Lasso model and Least Squares model
    rid_lam = np.argmin(r_test_score_lst)
    lasso_lam = np.argmin(l_test_score_lst)
    print('ridge lambda:', round(lam_range[rid_lam], 2), '\nlasso lambda:', round(lam_range[lasso_lam], 2))
    print('cv ridge loss:', round(r_test_score_lst[rid_lam], 2), '\ncv lasso loss:',
          round(l_test_score_lst[lasso_lam], 2))
    my_ridge_loss = RidgeRegression(lam_range[rid_lam]).fit(train_x, train_y).loss(test_x, test_y)
    my_lasso = Lasso(lam_range[lasso_lam], max_iter=10000).fit(train_x, train_y).predict(test_x)
    my_lasso_loss = loss_func(test_y, my_lasso)
    my_reg_loss = LinearRegression().fit(train_x, train_y).loss(test_x, test_y)
    print('ridge loss:', round(my_ridge_loss, 2), '\nlasso loss:', round(my_lasso_loss, 2), '\nreg loss:',
          round(my_reg_loss, 2))


if __name__ == '__main__':
    np.random.seed(0)
    select_polynomial_degree(n_samples=100, noise=5)
    select_polynomial_degree(n_samples=100, noise=0)
    select_polynomial_degree(n_samples=1500, noise=10)
    select_regularization_parameter()
