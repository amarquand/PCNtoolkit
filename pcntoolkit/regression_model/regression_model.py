from abc import ABC, abstractmethod

from pcntoolkit.regression_model.reg_conf import RegConf


class RegressionModel(ABC):
    def __init__(self, name, reg_conf: RegConf, is_fitted=False, is_from_dict=False):
        self._name = name
        self._reg_conf: RegConf = reg_conf
        self.is_fitted = is_fitted
        self._is_from_dict = is_from_dict

    @property
    def name(self) -> str:
        return self._name

    @property
    def reg_conf(self) -> RegConf:
        return self._reg_conf

    @property
    def is_from_dict(self) -> bool:
        return self._is_from_dict

    def to_dict(self) -> dict:
        """
        Converts the regression model to a dictionary.
        """
        my_dict = {}
        my_dict["name"] = self.name
        my_dict["reg_conf"] = self.reg_conf.to_dict()
        my_dict["is_fitted"] = self.is_fitted
        my_dict["is_from_dict"] = self.is_from_dict
        return my_dict

    @classmethod
    @abstractmethod
    def from_dict(cls, dict, path=None):
        """
        Loads a regression model from a dictionary.
        Takes an optional path argument to load large model components from.
        """
        pass

    @classmethod
    @abstractmethod
    def from_args(cls, name, args):
        pass
