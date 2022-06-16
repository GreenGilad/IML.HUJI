import numpy as np

from IMLearn.base import BaseModule, BaseLR


class FixedLR(BaseLR):
    """
    Class representing a fixed learning rate
    """
    def __init__(self, base_lr: float):
        """
        Instantiate a fixed learning-rate object

        Parameters:
        -----------
         base_lr: float
            Learning rate value to be returned at each call
        """
        super().__init__()
        self.base_lr = base_lr

    def lr_step(self, **lr_kwargs) -> float:
        """
        Specify learning rate at call

        Returns:
        --------
        eta: float
            Fixed learning rate specified when initializing instance

        Note:
        -----
        No arguments are expected
        """
        raise NotImplementedError()


class ExponentialLR(FixedLR):
    """
    Class representing an exponentially decaying learning rate
    """
    def __init__(self, base_lr: float, decay_rate: float):
        """
        Instantiate an exponentially decaying learning-rate object, i.e. eta_t = eta*gamma^t

        Parameters:
        ----------
        base_lr: float
            Learning to be returned at t=0 (i.e eta)

        decay_rate: float
            Decay rate of learning-rate (i.e. gamma)
        """
        super().__init__(base_lr)
        self.decay_rate = decay_rate

    def lr_step(self, t: int, **lr_kwargs) -> float:
        """
        Specify learning rate at call `t`

        Parameters:
        -----------
        t: int
            Step time for which to calculate learning rate

        Returns:
        --------
        eta_t: float
            Exponential decay according to eta_t = eta*gamma^t
        """
        raise NotImplementedError()
