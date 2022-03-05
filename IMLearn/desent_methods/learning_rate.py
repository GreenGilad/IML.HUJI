import numpy as np
from IMLearn.base import BaseModule, BaseLearningRate


class FixedLR(BaseLearningRate):
    def __init__(self, base_lr: float):
        super().__init__()
        self.base_lr = base_lr

    def lr_step(self, **lr_kwargs) -> float:
        return self.base_lr


class AdaptiveLR(BaseLearningRate):
    def __init__(self, alpha: float, beta: float):
        super().__init__()
        self.alpha = alpha
        self.beta = beta

    def lr_step(self, f: BaseModule, x: np.ndarray, dx: np.ndarray, **lr_kwargs):
        raise NotImplementedError()


class ExponentialLR(FixedLR):
    def __init__(self, base_lr: float, decay_rate: float):
        super().__init__(base_lr)
        self.decay_rate = decay_rate

    def lr_step(self, iter, **lr_kwargs) -> float:
        raise NotImplementedError()
