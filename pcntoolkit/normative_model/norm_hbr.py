"""
norm_hbr.py

This module provides the implementation of the NormHBR class, which is a specialized
normative model using Hierarchical Bayesian Regression (HBR). It extends the NormBase
class and provides methods for fitting, predicting, and transferring models, as well as
computing centiles and z-scores.

Classes
-------
NormHBR : NormBase
    A class for creating and managing a normative model using Hierarchical Bayesian Regression.

Dependencies
------------
- numpy
- xarray
- pcntoolkit.dataio.norm_data.NormData
- pcntoolkit.normative_model.norm_base.NormBase
- pcntoolkit.normative_model.norm_conf.NormConf
- pcntoolkit.regression_model.hbr.hbr_data
- pcntoolkit.regression_model.hbr.hbr.HBR
- pcntoolkit.regression_model.hbr.hbr_conf.HBRConf

Usage
-----
To use this module, create an instance of the NormHBR class with the appropriate
configuration objects, and then call its methods to fit, predict, or transfer models.

Example
-------
>>> from pcntoolkit.normative_model.norm_conf import NormConf
>>> from pcntoolkit.regression_model.hbr.hbr_conf import HBRConf
>>> from pcntoolkit.dataio.norm_data import NormData
>>> norm_conf = NormConf(...)
>>> hbr_conf = HBRConf(...)
>>> model = NormHBR(norm_conf, hbr_conf)
>>> data = NormData(...)
>>> model._fit(data)
>>> predictions = model._predict(data)

Notes
-----
- The NormHBR class assumes that the data provided is compatible with the HBR model.
- The class provides several methods that are not implemented and will raise
  NotImplementedError if called. These methods are placeholders for future extensions.

"""

from __future__ import annotations

from typing import Any

import numpy as np
import xarray as xr

from pcntoolkit.dataio.norm_data import NormData
from pcntoolkit.normative_model.norm_base import NormBase
from pcntoolkit.normative_model.norm_conf import NormConf
from pcntoolkit.regression_model.hbr import hbr_data
from pcntoolkit.regression_model.hbr.hbr import HBR
from pcntoolkit.regression_model.hbr.hbr_conf import HBRConf


class NormHBR(NormBase):
    """
    A class for creating and managing a normative model using Hierarchical Bayesian Regression (HBR).

    This class extends the NormBase class and provides methods for fitting, predicting, and transferring
    models, as well as computing centiles and z-scores. It is designed to work with data compatible with
    the HBR model.

    Parameters
    ----------
    norm_conf : NormConf
        Configuration object for the normative model.
    reg_conf : Optional[HBRConf], optional
        Configuration object for the regression model. If not provided, a default HBRConf is used.

    Attributes
    ----------
    default_reg_conf : HBRConf
        The default configuration for the regression model.
    regression_model_type : Type[HBR]
        The type of regression model used, which is HBR.
    current_regression_model : HBR
        The current instance of the regression model being used.

    Methods
    -------
    _fit(data: NormData, make_new_model: bool = False) -> None
        Fits the HBR model to the provided data.

    _predict(data: NormData) -> None
        Predicts outcomes using the fitted HBR model.

    _fit_predict(fit_data: NormData, predict_data: NormData) -> None
        Fits the model to the fit_data and predicts outcomes for predict_data.

    _transfer(data: NormData, **kwargs: Any) -> HBR
        Transfers the model to new data with optional freedom parameter.

    _centiles(data: NormData, cdf: np.ndarray, **kwargs: Any) -> xr.DataArray
        Computes centiles for the given data and cumulative distribution function.

    _zscores(data: NormData, resample: bool = False) -> xr.DataArray
        Computes z-scores for the given data.

    n_params() -> int
        Returns the number of parameters in the model.

    from_args(cls, args: Any) -> "NormHBR"
        Creates a NormHBR instance from command line arguments.

    normdata_to_hbrdata(data: NormData) -> hbr_data.HBRData
        Converts NormData to HBRData format.

    Raises
    ------
    RuntimeError
        If the model is not fitted before calling transfer.

    NotImplementedError
        If extend, tune, or merge methods are called.

    Examples
    --------
    >>> norm_conf = NormConf(...)
    >>> hbr_conf = HBRConf(...)
    >>> model = NormHBR(norm_conf, hbr_conf)
    >>> model._fit(data)
    >>> predictions = model._predict(data)
    """

    def __init__(self, norm_conf: NormConf, reg_conf: HBRConf = None): # type: ignore
        super().__init__(norm_conf)
        if reg_conf is None:
            reg_conf = HBRConf()
        self.default_reg_conf: HBRConf = reg_conf
        self.regression_model_type = HBR
        self.current_regression_model: HBR = None  # type: ignore

    def _fit(self, data: NormData, make_new_model: bool = False) -> None:
        hbrdata = self.normdata_to_hbrdata(data)
        self.current_regression_model.fit(hbrdata, make_new_model)

    def _predict(self, data: NormData) -> None:
        hbrdata = self.normdata_to_hbrdata(data)
        self.current_regression_model.predict(hbrdata)

    def _fit_predict(self, fit_data: NormData, predict_data: NormData) -> None:
        fit_hbrdata = self.normdata_to_hbrdata(fit_data)
        predict_hbrdata = self.normdata_to_hbrdata(predict_data)
        self.current_regression_model.fit_predict(fit_hbrdata, predict_hbrdata)

    def _transfer(self, data: NormData, **kwargs: Any) -> HBR:
        freedom = kwargs.get("freedom", 1)
        transferdata = self.normdata_to_hbrdata(data)
        if not self.current_regression_model.is_fitted:
            raise RuntimeError("Model needs to be fitted before it can be transferred")
        new_hbr_model = self.current_regression_model.transfer(
            self.default_reg_conf, transferdata, freedom
        )
        return new_hbr_model

    def _extend(self, data: NormData) -> NormHBR:
        raise NotImplementedError(
            f"Extend method not implemented for {self.__class__.__name__}"
        )

    def _tune(self, data: NormData) -> NormHBR:
        raise NotImplementedError(
            f"Tune method not implemented for {self.__class__.__name__}"
        )

    def _merge(self, other: NormBase) -> NormHBR:
        raise NotImplementedError(
            f"Merge method not implemented for {self.__class__.__name__}"
        )

    def _centiles(self, data: NormData, cdf: np.ndarray, **kwargs: Any) -> xr.DataArray:
        resample = kwargs.get("resample", False)
        hbrdata = self.normdata_to_hbrdata(data)

        return self.current_regression_model.centiles(hbrdata, cdf, resample)

    def _zscores(self, data: NormData, resample: bool = False) -> xr.DataArray:
        hbrdata = self.normdata_to_hbrdata(data)
        return self.current_regression_model.zscores(hbrdata, resample)

    def n_params(self) -> int:
        return sum(
            [i.size.eval() for i in self.current_regression_model.pymc_model.free_RVs]
        )

    @classmethod
    def from_args(cls, args: Any) -> "NormHBR":
        norm_conf = NormConf.from_args(args)
        hbrconf = HBRConf.from_args(args)
        self = cls(norm_conf, hbrconf)
        return self

    @staticmethod
    def normdata_to_hbrdata(data: NormData) -> hbr_data.HBRData:
        """
        Converts NormData to HBRData format.

        This method extracts the necessary components from a NormData object and
        constructs an HBRData object, which is used for Hierarchical Bayesian Regression.

        Parameters
        ----------
        data : NormData
            The NormData object containing the input data, including covariates,
            response variables, and batch effects.

        Returns
        -------
        HBRData
            An HBRData object containing the converted data, ready for use in
            Hierarchical Bayesian Regression.

        Raises
        ------
        AssertionError
            If the response variable in the data has more than one dimension.

        Examples
        --------
        >>> norm_data = NormData(...)
        >>> hbr_data = NormHBR.normdata_to_hbrdata(norm_data)
        """
        if hasattr(data, "Phi") and data.Phi is not None:
            this_X = data.Phi.to_numpy()
            this_covariate_dims = data.basis_functions.to_numpy()
        elif hasattr(data, "scaled_X") and data.scaled_X is not None:
            this_X = data.scaled_X.to_numpy()
            this_covariate_dims = data.covariates.to_numpy()
        else:
            this_X = data.X.to_numpy()
            this_covariate_dims = data.covariates.to_numpy()

        if hasattr(data, "scaled_y") and data.scaled_y is not None:
            this_y = data.scaled_y.to_numpy()
        else:
            this_y = data.y.to_numpy()

        assert (len(data.y.shape) == 1) or (
            data.y.shape[1] == 1
        ), "Only one response variable is supported for HBRdata"

        hbrdata = hbr_data.HBRData(
            X=this_X,
            y=this_y,
            batch_effects=data.batch_effects.to_numpy(),
            response_var=data.response_vars.to_numpy().item(),
            covariate_dims=this_covariate_dims,
            batch_effect_dims=data.batch_effect_dims.to_numpy().tolist(),
            datapoint_coords=data.datapoints.to_numpy().tolist(),
        )
        hbrdata.set_batch_effects_maps(data.attrs["batch_effects_maps"])
        return hbrdata
