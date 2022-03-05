from IMLearn.base import BaseModule, BaseLearningRate
from .learning_rate import FixedLR
import numpy as np

OUTPUT_VECTOR_TYPE = ["last", "best", "average"]


class GradientDescent:
    """
    Gradient Descent algorithm

    Gradient descent algorithm for minimizing convex functions
    """
    def __init__(self, learning_rate: BaseLearningRate = FixedLR(1e-3), tol: float = 1e-4, max_iter: int = 1000,
                 out_type: str = "last", batch_size=None):
        raise NotImplementedError()

    def fit(self, f: BaseModule, X, y):
        raise NotImplementedError()
