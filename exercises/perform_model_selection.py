from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn import datasets
from IMLearn.metrics import mean_square_error
from IMLearn.utils import split_train_test
from IMLearn.model_selection import cross_validate
from IMLearn.learners.regressors import PolynomialFitting, LinearRegression, RidgeRegression, LassoRegression
from sklearn.linear_model import Lasso

from utils import *
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px


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
    sample_range = (-1.2, 2)
    noise_mean = 0
    split_train_ratio = 2/3
    def f(X):
        return (X+3)*(X+2)*(X+1)*(X-1)*(X-2)

    X = np.random.uniform(low=sample_range[0], high=sample_range[1], size=n_samples)
    eps = np.random.normal(noise_mean, noise, n_samples)
    y = f(X) + eps

    df_X = pd.DataFrame(data=X)
    df_y = pd.Series(data=y)

    train_X, train_y, test_X, test_y = split_train_test(df_X, df_y, split_train_ratio)

    true_train = pd.Series(data=f(train_X.squeeze().to_numpy()))
    true_test = pd.Series(data=f(test_X.squeeze().to_numpy()))

    samples = pd.concat((train_X.squeeze(), train_X.squeeze(), test_X.squeeze(), test_X.squeeze()))
    labels = pd.concat((train_y, true_train, test_y.squeeze(), true_test))
    names = ['training']*train_y.shape[0] + ['training_without_noise']*true_train.shape[0] + ['test']*test_y.shape[0] + ['test_without_noise']*true_test.shape[0]

    sample_plot_title = f'polynomial_samples_n={n_samples}_std={noise}'

    sample_plot = px.scatter(x=samples, y=labels, color=names, title=sample_plot_title)
    sample_plot.show()
    sample_plot.write_image(f'ex5_plots/{sample_plot_title}.png')

    # Question 2 - Perform CV for polynomial fitting with degrees 0,1,...,10
    degree_range = 10

    training_loss = np.empty(degree_range + 1)
    validation_loss = np.empty(degree_range + 1)
    for k in range(degree_range + 1):
        model = PolynomialFitting(k)
        mean_loss, _ = cross_validate(model, train_X.squeeze().to_numpy(), train_y.to_numpy(), mean_square_error)
        training_loss[k] = mean_loss
        validation_loss[k] = model.loss(test_X.squeeze().to_numpy(), test_y.to_numpy())

    loss = np.concatenate((training_loss, validation_loss))
    degree = np.concatenate((np.arange(degree_range + 1), np.arange(degree_range + 1)))
    label = np.array(['training']*(degree_range+1) + ['validation']*(degree_range+1))

    loss_plot_title = f'polynomial_fitted_loss_n={n_samples}_std={noise}'
    loss_plot = px.line(x=degree, y=loss, color=label, title=loss_plot_title)
    loss_plot.show()
    loss_plot.write_image(f'ex5_plots/{loss_plot_title}.png')

    # Question 3 - Using best value of k, fit a k-degree polynomial model and report test error
    k_star = np.argmin(validation_loss)
    print("Degree of polynomial of lowest MSE: ", k_star)
    print("-------------------------with loss:", validation_loss[k_star])

    optimal_model = PolynomialFitting(k_star)
    test_error = optimal_model.fit(train_X.squeeze().to_numpy(), train_y.to_numpy()).loss(test_X.squeeze().to_numpy(), test_y.to_numpy())
    prediction_graph = sample_plot.add_scatter(x=test_X.squeeze().to_numpy(),
                                               y=optimal_model.predict(test_X.squeeze().to_numpy()),
                                               name=f'predicted deg={k_star}',
                                               mode='markers')
    prediction_graph.update_layout(title_text=f'polynomial samples with predicted degree={k_star} n={n_samples} std={noise}')
    prediction_graph.show()

    print("Test error of optimal model: ", round(test_error, 2))

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
    data_dict = datasets.load_diabetes(as_frame=True)
    data, target = data_dict['data'], data_dict['target']

    split_train_ratio = 1 - (n_samples / target.size)

    train_X, train_y, test_X, test_y = split_train_test(data, target, split_train_ratio)

    # Question 7 - Perform CV for different values of the regularization parameter for Ridge and Lasso regressions
    eval_range = (1, n_evaluations)
    ridge_stretch = 0.2
    lasso_stretch = 5

    ridge_training_loss = np.empty(eval_range[1]-eval_range[0])
    ridge_validation_loss = np.empty(eval_range[1]-eval_range[0])

    lasso_training_loss = np.empty(eval_range[1]-eval_range[0])
    lasso_validation_loss = np.empty(eval_range[1]-eval_range[0])

    for lam in range(*eval_range):
        lambda_val = lam/eval_range[1]
        ridge_model = RidgeRegression(ridge_stretch*lambda_val)
        ridge_mean_loss, _ = cross_validate(ridge_model, train_X.to_numpy(), train_y.to_numpy(),
                                            mean_square_error)
        ridge_training_loss[lam-eval_range[0]] = ridge_mean_loss
        ridge_validation_loss[lam-eval_range[0]] = ridge_model.loss(test_X.to_numpy(), test_y.to_numpy())

        lasso_model = LassoRegression(lasso_stretch*lambda_val, optimizer=None)
        lasso_mean_loss, _ = cross_validate(lasso_model, train_X.to_numpy(), train_y.to_numpy(),
                                            mean_square_error)
        lasso_training_loss[lam-eval_range[0]] = lasso_mean_loss
        lasso_validation_loss[lam-eval_range[0]] = lasso_model.loss(test_X.to_numpy(), test_y.to_numpy())

    ridge_title = f"Ridge Regression Loss for lambda in range {[round(ridge_stretch * (x/eval_range[1]), 4) for x in eval_range]} n_samples={n_samples} n_evals={n_evaluations}"

    ridge_plot = px.line(x=np.linspace(*[ridge_stretch * (x/eval_range[1]) for x in eval_range], num=eval_range[1]-eval_range[0]),
                         y=[ridge_validation_loss, ridge_training_loss], title=ridge_title)
    for trace, id in zip(ridge_plot["data"], ("validation", "training")):
        trace["name"] = f"{id} loss"
    ridge_plot.show()
    ridge_plot.write_image(f"ex5_plots/{ridge_title}.png")

    lasso_title = f"Lasso Regression Loss for lambda in range {[round(lasso_stretch * (x/eval_range[1]), 4) for x in eval_range]} n_samples={n_samples} n_evals={n_evaluations}"
    lasso_plot = px.line(x=np.linspace(*[lasso_stretch * (x/eval_range[1]) for x in eval_range], num=eval_range[1]-eval_range[0]),
                         y=[lasso_validation_loss, lasso_training_loss], title=lasso_title)
    for trace, id in zip(lasso_plot["data"], ("validation", "training")):
        trace["name"] = f"{id} loss"
    lasso_plot.show()
    lasso_plot.write_image(f"ex5_plots/{lasso_title}.png")

    # Question 8 - Compare best Ridge model, best Lasso model and Least Squares model
    ridge_lambdah = ridge_stretch * ((np.argmin(ridge_validation_loss) + eval_range[0]) / eval_range[1])
    lasso_lambdah = lasso_stretch * ((np.argmin(lasso_validation_loss) + eval_range[0]) / eval_range[1])

    models = RidgeRegression(ridge_lambdah), LassoRegression(lasso_lambdah, optimizer=None), LinearRegression()
    for name, model in zip([f"ridge regression lambda={ridge_lambdah}", f"lasso regression lambda={lasso_lambdah}", "least squares"], models):
        model.fit(train_X.to_numpy(), train_y.to_numpy())
        loss = model.loss(test_X.to_numpy(), test_y.to_numpy())
        print()
        print(f"Validation Error for {name}")
        print(loss)


if __name__ == '__main__':
    np.random.seed(0)
    for params in [(100, 5), (100, 0), (1500, 10)]:
        print('~'*20,'\n',f'Selecting optimal polynomial degree with sample size and noise level: {params}', '\n', '~'*20)
        select_polynomial_degree(*params)
    select_regularization_parameter()
