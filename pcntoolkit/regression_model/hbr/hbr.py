import numpy as np

from .hbr_conf import HBRConf


class HBR:

    def __init__(self, conf: HBRConf):
        self._conf:HBRConf = conf

    def fit(self, X:np.ndarray, y:np.ndarray):
        """
        Fits the model.
        """
        # some fitting logic
        # ...
        raise NotImplementedError(
            f"Fit method not implemented for {self.__class__.__name__}")
    
    def predict(self, X:np.ndarray) -> np.ndarray:
        """
        Predicts on new data.
        """
        # some prediction logic
        # ...
        raise NotImplementedError(
            f"Predict method not implemented for {self.__class__.__name__}")
    
    def fit_predict(self, X:np.ndarray, y:np.ndarray, X_test) -> np.ndarray:
        """
        Fits and predicts the model.
        """
        # some fit_predict logic
        # ...
        raise NotImplementedError(
            f"Fit-predict method not implemented for {self.__class__.__name__}")