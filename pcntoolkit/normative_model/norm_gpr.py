import warnings

import numpy as np
import xarray as xr

from pcntoolkit.dataio.norm_data import NormData
from pcntoolkit.normative_model.norm_base import NormBase
from pcntoolkit.normative_model.norm_conf import NormConf
from pcntoolkit.regression_model.gpr.gpr import GPR
from pcntoolkit.regression_model.gpr.gpr_conf import GPRConf

#


class NormGPR(NormBase):
    def __init__(self, norm_conf: NormConf, reg_conf: GPRConf):
        super().__init__(norm_conf)
        self._reg_conf: GPRConf = reg_conf
        self.model_type = GPR
        self.model: GPR = None

    @classmethod
    def from_args(cls, args):
        """
        Creates a configuration from command line arguments.
        """
        norm_conf = NormConf.from_args(args)
        hbrconf = GPRConf.from_args(args)
        self = cls(norm_conf, hbrconf)
        return self

    @staticmethod
    def reg_conf_from_args(dict):
        return GPRConf.from_args(dict)

    def models_to_dict(self, dict):
        raise NotImplementedError(
            f"Models to dict method not implemented for {self.__class__.__name__}"
        )

    def dict_to_models(self, dict):
        raise NotImplementedError(
            f"Dict to models method not implemented for {self.__class__.__name__}"
        )

    def _fit(self, data: NormData):
        raise NotImplementedError(
            f"Fit method not implemented for {self.__class__.__name__}"
        )

    def _predict(self, data: NormData) -> NormData:
        raise NotImplementedError(
            f"Predict method not implemented for {self.__class__.__name__}"
        )

    def _fit_predict(self, data: NormData) -> NormData:
        raise NotImplementedError(
            f"Fit-predict method not implemented for {self.__class__.__name__}"
        )

    def _transfer(self, data: NormData) -> NormBase:
        raise NotImplementedError(
            f"Transfer method not implemented for {self.__class__.__name__}"
        )

    def _extend(self, data: NormData):
        raise NotImplementedError(
            f"Extend method not implemented for {self.__class__.__name__}"
        )

    def _tune(self, data: NormData):
        raise NotImplementedError(
            f"Tune method not implemented for {self.__class__.__name__}"
        )

    def _merge(self, other: NormBase):
        raise NotImplementedError(
            f"Merge method not implemented for {self.__class__.__name__}"
        )

    def _quantiles(self, data: NormData, quantiles: list[float]) -> xr.DataArray:
        raise NotImplementedError(
            f"Quantiles method not implemented for {self.__class__.__name__}"
        )

    def _zscores(self, data: NormData) -> xr.DataArray:
        raise NotImplementedError(
            f"Zscores method not implemented for {self.__class__.__name__}"
        )

    def n_params(self) -> xr.DataArray:
        raise NotImplementedError(
            f"n_params method not implemented for {self.__class__.__name__}"
        )
