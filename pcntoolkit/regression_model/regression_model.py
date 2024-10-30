from __future__ import annotations

from abc import ABC, abstractmethod

from pcntoolkit.regression_model.reg_conf import RegConf


class RegressionModel(ABC):
    def __init__(
        self,
        name: str,
        reg_conf: RegConf,
        is_fitted: bool = False,
        is_from_dict: bool = False,
    ):
        self._name: str = name
        self._reg_conf: RegConf = reg_conf
        self.is_fitted: bool = is_fitted
        self._is_from_dict: bool = is_from_dict

    @property
    def name(self) -> str:
        return self._name

    @property
    def reg_conf(self) -> RegConf:
        return self._reg_conf

    @property
    def is_from_dict(self) -> bool:
        return self._is_from_dict

    def to_dict(self, path: str | None = None) -> dict:
        my_dict: dict[str, str | dict | bool] = {}
        my_dict["name"] = self.name
        my_dict["reg_conf"] = self.reg_conf.to_dict(path)
        my_dict["is_fitted"] = self.is_fitted
        my_dict["is_from_dict"] = self.is_from_dict
        return my_dict

    @classmethod
    @abstractmethod
    def from_dict(cls, dct: dict, path: str) -> RegressionModel:
        pass

    @classmethod
    @abstractmethod
    def from_args(cls, name: str, args: dict) -> RegressionModel:
        pass

    @property
    def has_random_effect(self) -> bool:
        return self.reg_conf.has_random_effect
