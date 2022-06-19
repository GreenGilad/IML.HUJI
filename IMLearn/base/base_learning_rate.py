from abc import ABC


class BaseLR(ABC):
    """
    Base class of learning rates (step size) strategies to be used in different descent methods
    """

    def lr_step(self, **lr_kwargs) -> float:
        """
        Return a step size, depending on the strategy and given input arguments

        Parameters
        ----------
        lr_kwargs:
            Arbitrary keyword arguments needed for inheriting learning rate strategies

        Returns
        -------
        learning_rate: float
            Learning rate to be used in descent method
        """
        raise NotImplementedError()
