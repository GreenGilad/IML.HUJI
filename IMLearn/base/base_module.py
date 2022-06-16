from __future__ import annotations
from abc import ABC
import numpy as np


class BaseModule(ABC):
    """
    Base class representing a function to be optimized in a descent method algorithm

    Attributes
    ----------
    weights_ : ndarray of shape (n_in, n_out)
        Parameters of function with respect to which the function is optimized.
    """

    def __init__(self, weights: np.ndarray = None) -> BaseModule:
        """
        Initialize a module instance

        Parameters:
        ----------
        weights: np.ndarray, default None
            Initial value of weights
        """
        self.weights_ = weights

    def compute_output(self, **kwargs) -> np.ndarray:
        """
        Compute the output value of the function

        Parameters
        ----------
        kwargs: Additional arguments to be passed and used by derived objects

        Returns
        -------
        output: ndarray of shape (n_out,)
            Value of function at `input`

        Examples
        --------
        For f:R^d->R defined by f(x) = <w,x> then: n_in=d, n_out=1 and thus output shape is (1,)
        """
        raise NotImplementedError()

    def compute_jacobian(self, **kwargs) -> np.ndarray:
        """
        Compute the derivative of the function with respect to each of its parameters

        Parameters
        ----------
        kwargs: Additional arguments to be passed and used by derived objects

        Returns
        -------
        output: ndarray of shape (n_out, n_in)
            Derivative of function with respect to its parameters at `input`

        Examples
        --------
        For f:R^d->R defined by f(x) = <w,x> then: n_in=d, n_out=1 and thus output shape is (1,d)

        """
        raise NotImplementedError()

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

        Parameters
        ----------
        weights: ndarray array of shape (n_in, n_out)
        """
        self.weights_ = weights

    @property
    def shape(self):
        """
        Specify the dimensions of the function

        Returns
        -------
        shape: Tuple[int]
            Specifying the dimensions of the functions parameters. If ``self.weights`` is None returns `(0,)`
        """
        return self.weights.shape if self.weights is not None else (0,)


