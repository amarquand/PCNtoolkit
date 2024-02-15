import numpy as np

from pcntoolkit.regression_model.gpr.gpr_conf import GPRConf
from pcntoolkit.regression_model.regression_model import RegressionModel


class GPR(RegressionModel):

    def __init__(
        self, name: str, reg_conf: GPRConf, is_fitted=False, is_from_dict=False
    ):
        """
        Initializes the model.
        Any mutable parameters should be initialized here.
        Any immutable parameters should be initialized in the configuration.
        """
        super().__init__(name, reg_conf, is_fitted, is_from_dict)

    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        Fits the model.
        """
        # some fitting logic
        # ...
        raise NotImplementedError(
            f"Fit method not implemented for {self.__class__.__name__}"
        )

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predicts on new data.
        """
        # some prediction logic
        # ...
        raise NotImplementedError(
            f"Predict method not implemented for {self.__class__.__name__}"
        )

    def fit_predict(self, X: np.ndarray, y: np.ndarray, X_test) -> np.ndarray:
        """
        Fits and predicts the model.
        """
        # some fit_predict logic
        # ...
        raise NotImplementedError(
            f"Fit-predict method not implemented for {self.__class__.__name__}"
        )

    @classmethod
    def from_dict(cls, dict):
        """
        Creates a configuration from a dictionary.
        """
        name = dict["name"]
        conf = GPRConf.from_dict(dict["reg_conf"])
        is_fitted = dict["is_fitted"]
        is_from_dict = True
        self = cls(name, conf, is_fitted, is_from_dict)
        return self

    @classmethod
    def from_args(cls, name, args):
        """
        Creates a configuration from command line arguments
        """
        conf = GPRConf.from_args(args)
        is_fitted = args.get("is_fitted", False)
        is_from_dict = True
        self = cls(name, conf, is_fitted, is_from_dict)
        return self
