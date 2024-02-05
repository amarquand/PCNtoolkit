import warnings

import numpy as np
import xarray as xr

from pcntoolkit.dataio.norm_data import NormData
from pcntoolkit.normative_model.norm_base import NormBase
from pcntoolkit.normative_model.norm_conf import NormConf
from pcntoolkit.regression_model.blr.blr import BLR
from pcntoolkit.regression_model.blr.blr_conf import BLRConf


class NormBLR(NormBase):
    def __init__(self, norm_conf: NormConf, reg_conf: BLRConf):
        super().__init__(norm_conf)
        self._reg_conf: BLRConf = reg_conf
        self.model_type = BLR
        self.model: BLR = None

    @classmethod
    def from_args(cls, args):
        """
        Creates a configuration from command line arguments.
        """
        norm_conf = NormConf.from_args(args)
        hbrconf = BLRConf.from_args(args)
        self = cls(norm_conf, hbrconf)
        return self

    @staticmethod
    def reg_conf_from_args(dict):
        return BLRConf.from_args(dict)

    def models_to_dict(self, path=None):
        """
        Converts the models to a dictionary with serializable components.
        This dictionary is used to save the model to disk as json.
        Must contain all the information needed to recreate the model.
        Takes an optional path argument to save any large model components to disk.
        """
        raise NotImplementedError(
            f"Models to dict method not implemented for {self.__class__.__name__}"
        )

    def dict_to_models(self, dict):
        """
        Converts the dictionary to self.models.
        """
        raise NotImplementedError(
            f"Dict to models method not implemented for {self.__class__.__name__}"
        )

    def _fit(self, data: NormData):
        """
        Fit self.model on data.
        Will be called for each model in self.models from the super class.
        Data contains only the response variable for the current model.
        """
        raise NotImplementedError(
            f"Fit method not implemented for {self.__class__.__name__}"
        )

    def _predict(self, data: NormData) -> NormData:
        """
        Make predictions on data using self.model.
        Will be called for each model in self.models from the super class.
        Data contains only the response variable for the current model.
        """
        raise NotImplementedError(
            f"Predict method not implemented for {self.__class__.__name__}"
        )

    def _fit_predict(self, fit_data: NormData, predict_data: NormData) -> NormData:
        """
        Fit and predict on data using self.model.
        Will be called for each model in self.models from the super class.
        Data contains only the response variable for the current model.
        """
        raise NotImplementedError(
            f"Fit-predict method not implemented for {self.__class__.__name__}"
        )

    def _transfer(self, data: NormData) -> NormBase:
        """
        Transfer the model to a new dataset.
        Will be called for each model in self.models from the super class.
        Data contains only the response variable for the current model.
        """
        raise NotImplementedError(
            f"Transfer method not implemented for {self.__class__.__name__}"
        )

    def _extend(self, data: NormData):
        """
        Extend the model to a new dataset.
        Will be called for each model in self.models from the super class.
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
        Will be called for each model in self.models from the super class.
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
        Will be called for each model in self.models from the super class.
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
