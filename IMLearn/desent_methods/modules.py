import numpy as np
from ..base import BaseModule


class L2(BaseModule):
    """
    Class representing the L2 module

    Represents the function: f(w)=||w||^2_2
    """
    def __init__(self, weights: np.ndarray = None):
        """
        Initialize a module instance

        Parameters:
        ----------
        weights: np.ndarray, default=None
            Initial value of weights
        """
        super().__init__(weights)

    def compute_output(self, **kwargs) -> np.ndarray:
        """
        Compute the output value of the L2 function at point self.weights

        Parameters
        ----------
        kwargs:
            No additional arguments are expected

        Returns
        -------
        output: ndarray of shape (1,)
            Value of function at point self.weights
        """
        return np.sum(self.weights ** 2)

    def compute_jacobian(self, **kwargs) -> np.ndarray:
        """
        Compute L2 derivative with respect to self.weights at point self.weights

        Parameters
        ----------
        kwargs:
            No additional arguments are expected

        Returns
        -------
        output: ndarray of shape (n_in,)
            L2 derivative with respect to self.weights at point self.weights
        """
        return 2 * self.weights


class L1(BaseModule):
    def __init__(self, weights: np.ndarray = None):
        """
        Initialize a module instance

        Parameters:
        ----------
        weights: np.ndarray, default=None
            Initial value of weights
        """
        super().__init__(weights)

    def compute_output(self, **kwargs) -> np.ndarray:
        """
        Compute the output value of the L1 function at point self.weights

        Parameters
        ----------
        kwargs:
            No additional arguments are expected

        Returns
        -------
        output: ndarray of shape (1,)
            Value of function at point self.weights
        """
        return np.sum(np.absolute(self.weights))

    def compute_jacobian(self, **kwargs) -> np.ndarray:
        """
        Compute L1 derivative with respect to self.weights at point self.weights

        Parameters
        ----------
        kwargs:
            No additional arguments are expected

        Returns
        -------
        output: ndarray of shape (n_in,)
            L1 derivative with respect to self.weights at point self.weights
        """
        return np.sign(self.weights)


class LogisticModule(BaseModule):
    """
    Class representing the logistic regression objective function

    Represents the function: f(w) = - (1/m) sum_i^m[y*<x_i,w> - log(1+exp(<x_i,w>))]
    """
    def __init__(self, weights: np.ndarray = None):
        """
        Initialize a logistic regression module instance

        Parameters:
        ----------
        weights: np.ndarray, default=None
            Initial value of weights
        """
        super().__init__(weights)

    def compute_output(self, X: np.ndarray, y: np.ndarray, **kwargs) -> np.ndarray:
        """
        Compute the output value of the logistic regression objective function at point self.weights

        Parameters
        ----------
        X: ndarray of shape (n_samples, n_features)
            Design matrix to use when computing objective

        y: ndarray of shape(n_samples,)
            Binary labels of samples to use when computing objective

        Returns
        -------
        output: ndarray of shape (1,)
            Value of function at point self.weights
        """
        # f(w) = - (1/m) sum_i^m[y*<x_i,w> - log(1+exp(<x_i,w>))
        log = -np.log(1 + np.exp(X @ self.weights))
        all_func = np.sum(y * (X @ self.weights) - log)
        return (-1 / X.shape[0]) * all_func

    def compute_jacobian(self, X: np.ndarray, y: np.ndarray, **kwargs) -> np.ndarray:
        """
        Compute the gradient of the logistic regression objective function at point self.weights

        Parameters
        ----------
        X: ndarray of shape (n_samples, n_features)
            Design matrix to use when computing objective

        y: ndarray of shape(n_samples,)
            Binary labels of samples to use when computing objective

        Returns
        -------
        output: ndarray of shape (n_features,)
            Derivative of function with respect to self.weights at point self.weights
        """
        exp = np.exp(self.weights @ X.T) / (1 + np.exp(self.weights @ X.T))
        return -(1 / X.shape[0]) * (y @ X - X.T @ exp)


class RegularizedModule(BaseModule):
    """
    Class representing a general regularized objective function of the format:
                                    f(w) = F(w) + lambda*R(w)
    for F(w) being some fidelity function, R(w) some regularization function and lambda
    the regularization parameter
    """
    def __init__(self,
                 fidelity_module: BaseModule,
                 regularization_module: BaseModule,
                 lam: float = 1.,
                 weights: np.ndarray = None,
                 include_intercept: bool = True):
        """
        Initialize a regularized objective module instance

        Parameters:
        -----------
        fidelity_module: BaseModule
            Module to be used as a fidelity term

        regularization_module: BaseModule
            Module to be used as a regularization term

        lam: float, default=1
            Value of regularization parameter

        weights: np.ndarray, default=None
            Initial value of weights

        include_intercept: bool default=True
            Should fidelity term (and not regularization term) include an intercept or not
        """
        super().__init__()
        self.fidelity_module_, self.regularization_module_, self.lam_ = fidelity_module, regularization_module, lam
        self.include_intercept_ = include_intercept

        if weights is not None:
            # self.weights(weights)
            self.weights_ = weights
            if self.include_intercept_:
                self.regularization_module_.weights = weights[1:]
            else:
                self.regularization_module_.weights = weights
            self.fidelity_module_.weights = weights

    def compute_output(self, **kwargs) -> np.ndarray:
        """
        Compute the output value of the regularized objective function at point self.weights

        Parameters
        ----------
        kwargs:
            No additional arguments are expected

        Returns
        -------
        output: ndarray of shape (1,)
            Value of function at point self.weights
        """
        # f(w) = F(w) + lambda *R(w)
        return self.fidelity_module_.compute_output(**kwargs) + \
               self.lam_ * self.regularization_module_.compute_output(**kwargs)

    def compute_jacobian(self, **kwargs) -> np.ndarray:
        """
        Compute module derivative with respect to self.weights at point self.weights

        Parameters
        ----------
        kwargs:
            No additional arguments are expected

        Returns
        -------
        output: ndarray of shape (n_in,)
            Derivative with respect to self.weights at point self.weights
        """
        jacob_func = self.fidelity_module_.compute_jacobian(**kwargs)
        jacob_regu = self.regularization_module_.compute_jacobian(**kwargs)

        if self.include_intercept_:
            jacob_regu = np.insert(jacob_regu, 0, 0, axis=0)

        return jacob_func + self.lam_ * jacob_regu

    @property
    def weights(self):
        """
        Wrapper property to retrieve module parameter

        Returns
        -------
        weights: ndarray of shape (n_in, n_out)
        """
        return self.weights_

    @weights.setter
    def weights(self, weights: np.ndarray) -> None:
        """
        Setter function for module parameters

        In case self.include_intercept_ is set to True, weights[0] is regarded as the intercept
        and is not passed to the regularization module

        Parameters
        ----------
        weights: ndarray of shape (n_in, n_out)
            Weights to set for module
        """
        self.weights_ = weights
        if self.include_intercept_:
            self.regularization_module_.weights = weights[1:]
        else:
            self.regularization_module_.weights = weights
        self.fidelity_module_.weights = weights
