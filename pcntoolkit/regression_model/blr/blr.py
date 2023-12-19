import numpy as np
from .blr_conf import BLRConf


class BLR:

    def __init__(self, conf: BLRConf):
        self._conf: BLRConf = conf

    def example_function_using_example_parameter(self, my_int: int):
        """
        This is an example function that uses the example parameter.
        """
        return my_int + self._conf.example_parameter

    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        Fits the model.
        """
        # some fitting logic
        # ...
        raise NotImplementedError(
            f"Fit method not implemented for {self.__class__.__name__}")

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predicts on new data.
        """
        # some prediction logic
        # ...
        raise NotImplementedError(
            f"Predict method not implemented for {self.__class__.__name__}")

    def fit_predict(self, X: np.ndarray, y: np.ndarray, X_test) -> np.ndarray:
        """
        Fits and predicts the model.
        """
        # some fit_predict logic
        # ...
        raise NotImplementedError(
            f"Fit-predict method not implemented for {self.__class__.__name__}")

    @property
    def conf(self) -> BLRConf:
        return self._conf
