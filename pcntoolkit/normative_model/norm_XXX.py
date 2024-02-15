import warnings

import numpy as np
import xarray as xr

from pcntoolkit.dataio.norm_data import NormData
from pcntoolkit.normative_model.norm_base import NormBase
from pcntoolkit.normative_model.norm_conf import NormConf
from pcntoolkit.regression_model.gpr.gpr import GPR
from pcntoolkit.regression_model.gpr.gpr_conf import GPRConf


class NormGPR(NormBase):
    def __init__(self, norm_conf: NormConf, reg_conf: GPRConf):
        super().__init__(norm_conf)
        self.reg_conf: GPRConf = reg_conf
        self.regression_model_type = GPR
        self.current_regression_model: GPR = None

    @classmethod
    def from_args(cls, args):
        """
        Creates a configuration from command line arguments.
        """
        norm_conf = NormConf.from_args(args)
        hbrconf = GPRConf.from_args(args)
        self = cls(norm_conf, hbrconf)
        return self

    def _fit(self, data: NormData):
        """
        Fit self.model on data.
        Will be called for each model in self.regression_models from the super class.
        Data contains only the response variable for the current model.
        """
        raise NotImplementedError(
            f"Fit method not implemented for {self.__class__.__name__}"
        )

    def _predict(self, data: NormData) -> NormData:
        """
        Make predictions on data using self.model.
        Will be called for each model in self.regression_models from the super class.
        Data contains only the response variable for the current model.
        """
        raise NotImplementedError(
            f"Predict method not implemented for {self.__class__.__name__}"
        )

    def _fit_predict(self, fit_data: NormData, predict_data: NormData) -> NormData:
        """
        Fit and predict on data using self.model.
        Will be called for each model in self.regression_models from the super class.
        Data contains only the response variable for the current model.
        """
        raise NotImplementedError(
            f"Fit-predict method not implemented for {self.__class__.__name__}"
        )

    def _transfer(self, data: NormData) -> NormBase:
        """
        Transfer the model to a new dataset.
        Will be called for each model in self.regression_models from the super class.
        Data contains only the response variable for the current model.
        """
        raise NotImplementedError(
            f"Transfer method not implemented for {self.__class__.__name__}"
        )

    def _extend(self, data: NormData):
        """
        Extend the model to a new dataset.
        Will be called for each model in self.regression_models from the super class.
        Data contains only the response variable for the current model.
        """
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
        """
        Compute quantiles for the model at the given data points.
        Will be called for each model in self.regression_models from the super class.
        Data contains only the response variable for the current model.
        The return type should be a DataArray with dimensions:
        - quantile_zscores
        - datapoints

        ```
        quantiles = np.zeros((len(zscores), data.X.shape[0])))
        for i, zscore in enumerate(zscores):
            quantiles[i, :] = *compute quantiles for zscore*

        return xr.DataArray(
            quantiles,
            dims=["quantile_zscores", "datapoints"],
            coords={"quantile_zscores": zscores},
        )```
        """

        raise NotImplementedError(
            f"Quantiles method not implemented for {self.__class__.__name__}"
        )

    def _zscores(self, data: NormData) -> xr.DataArray:
        """
        Compute zscores for the model at the given data points.
        Will be called for each model in self.regression_models from the super class.
        Data contains only the response variable for the current model.
        The return type should be a DataArray with dimensions:
        - datapoints

        ```
        zscores = *compute zscores for data*
        return xr.DataArray(
            zscores,
            dims=["datapoints"],
        )```
        """
        raise NotImplementedError(
            f"Zscores method not implemented for {self.__class__.__name__}"
        )

    def n_params(self) -> xr.DataArray:
        """
        compute the number of parameters for the model.
        """
        raise NotImplementedError(
            f"n_params method not implemented for {self.__class__.__name__}"
        )
