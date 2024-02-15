import numpy as np

from pcntoolkit.regression_model.regression_model import RegressionModel

from .blr_conf import BLRConf


class BLR(RegressionModel):

    def __init__(
        self, name: str, reg_conf: BLRConf, is_fitted=False, is_from_dict=False
    ):
        """
        Initializes the model.
        Any mutable parameters should be initialized here.
        Any immutable parameters should be initialized in the configuration.
        """
        super().__init__(name, reg_conf, is_fitted, is_from_dict)

    def example_function_using_example_parameter(self, my_int: int):
        """
        This is an example function that uses the example parameter.
        """
        return my_int + self._conf.example_parameter

    @classmethod
    def from_dict(cls, dict):
        """
        Creates a configuration from a dictionary.
        """
        name = dict["name"]
        conf = BLRConf.from_dict(dict["reg_conf"])
        is_fitted = dict["is_fitted"]
        is_from_dict = True
        self = cls(name, conf, is_fitted, is_from_dict)
        return self

    @classmethod
    def from_args(cls, name, args):
        """
        Creates a configuration from command line arguments
        """
        conf = BLRConf.from_args(args)
        is_fitted = args.get("is_fitted", False)
        is_from_dict = True
        self = cls(name, conf, is_fitted, is_from_dict)
        return self
