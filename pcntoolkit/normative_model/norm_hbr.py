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

from typing import Any, Type

import numpy as np
import pandas as pd
import xarray as xr

from pcntoolkit.dataio.norm_data import NormData
from pcntoolkit.normative_model.norm_base import NormBase
from pcntoolkit.normative_model.norm_conf import NormConf
from pcntoolkit.regression_model.hbr import hbr_data
from pcntoolkit.regression_model.hbr.hbr import HBR
from pcntoolkit.regression_model.hbr.hbr_conf import HBRConf
from pcntoolkit.regression_model.regression_model import RegressionModel
from pcntoolkit.util.output import Errors, Output


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
    reg_conf : Optional[HBRConf], optionafl
        Configuration object for the regression model. If not provided, a default HBRConf is used.

    Attributes
    ----------
    template_reg_conf : HBRConf
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

    def __init__(
        self,
        norm_conf: NormConf,
        reg_conf: HBRConf = None,  # type: ignore
        regression_model_type: Type[RegressionModel] = HBR,
    ):  # type: ignore
        super().__init__(norm_conf)
        if reg_conf is None:
            reg_conf = HBRConf()
        self.template_reg_conf: HBRConf = reg_conf
        self.regression_model_type = regression_model_type
        self.current_regression_model: HBR = None  # type: ignore

    def _fit(self, data: NormData, make_new_model: bool = False) -> None:
        hbrdata = self.normdata_to_hbrdata(data)
        self.focused_model.fit(hbrdata, make_new_model)  # type: ignore

    def _predict(self, data: NormData) -> None:
        hbrdata = self.normdata_to_hbrdata(data)
        self.focused_model.predict(hbrdata)  # type: ignore

    def _fit_predict(self, fit_data: NormData, predict_data: NormData) -> None:
        fit_hbrdata = self.normdata_to_hbrdata(fit_data)
        predict_hbrdata = self.normdata_to_hbrdata(predict_data)
        self.focused_model.fit_predict(fit_hbrdata, predict_hbrdata)  # type: ignore

    def _transfer(self, model_to_transfer_to: NormHBR, data: NormData, **kwargs: Any) -> HBR:
        freedom = kwargs.get("freedom", 1)
        transferdata = model_to_transfer_to.normdata_to_hbrdata(data)
        if not self.focused_model.is_fitted:
            raise Output.error(Errors.ERROR_MODEL_NOT_FITTED)
        reg_conf_dict: dict[str, Any] = model_to_transfer_to.template_reg_conf.to_dict()
        reg_conf_dict["draws"] = kwargs.get("draws", reg_conf_dict["draws"])
        reg_conf_dict["tune"] = kwargs.get("tune", reg_conf_dict["tune"])
        reg_conf_dict["pymc_cores"] = kwargs.get("pymc_cores", reg_conf_dict["pymc_cores"])
        reg_conf_dict["nuts_sampler"] = kwargs.get("nuts_sampler", reg_conf_dict["nuts_sampler"])
        reg_conf_dict["init"] = kwargs.get("init", reg_conf_dict["init"])
        reg_conf_dict["chains"] = kwargs.get("chains", reg_conf_dict["chains"])
        reg_conf = HBRConf.from_dict(reg_conf_dict)
        new_hbr_model = self.focused_model.transfer(  # type: ignore
            reg_conf, transferdata, freedom
        )
        return new_hbr_model

    def _extend(self, data: NormData) -> NormBase:
        hbrdata = self.normdata_to_hbrdata(data)
        return self.focused_model.extend(hbrdata)  # type: ignore

    def _generate_synthetic_data(self, data: NormData, n_synthetic_samples: int = 1000) -> NormData:
        df = pd.DataFrame()
        for c in data.X.coords["covariates"].values:
            c_min = self.inscalers[c].min
            c_max = self.inscalers[c].max
            df[c] = np.random.rand(n_synthetic_samples) * (c_max - c_min) + c_min

        for responsevar in data.response_vars.values:
            if responsevar not in self.response_vars:
                continue
            df[responsevar] = np.zeros(n_synthetic_samples)

        bes = self.sample_batch_effects(n_synthetic_samples)
        df = pd.concat([df, bes], axis=1)

        synthetic_data = NormData.from_dataframe(
            name="synthetic_data",
            dataframe=df,
            covariates=data.covariates.values,
            batch_effects=data.batch_effect_dims.values,
            response_vars=data.response_vars.values,
        )

        self.preprocess(synthetic_data)

        for responsevar in df.columns:
            if responsevar not in self.response_vars:
                continue
            resp_synthetic_data = synthetic_data.sel(response_vars=responsevar)
            self.focus(responsevar)
            hbr_data = self.normdata_to_hbrdata(resp_synthetic_data)
            self.focused_model.generate_synthetic_data(hbr_data)  # type: ignore
            synthetic_data.scaled_y.loc[{"response_vars": responsevar}] = hbr_data.y

        self.scale_backward(synthetic_data)
        return synthetic_data

    def _centiles(self, data: NormData, cdf: np.ndarray, **kwargs: Any) -> xr.DataArray:
        hbrdata = self.normdata_to_hbrdata(data)

        return self.focused_model.centiles(hbrdata, cdf)  # type: ignore

    def _zscores(self, data: NormData, **kwargs: Any) -> xr.DataArray:
        hbrdata = self.normdata_to_hbrdata(data)
        return self.focused_model.zscores(hbrdata)  # type: ignore

    def _logp(self, data: NormData) -> None:
        hbrdata = self.normdata_to_hbrdata(data)
        return self.focused_model.logp(hbrdata)  # type: ignore

    @property
    def focused_model(self) -> HBR:
        """Get the currently focused HBR model.

        Returns
        -------
        HBR
            The currently focused Hierarchical Bayesian Regression model.
        """
        return self[self.focused_var]  # type:ignore

    def n_params(self) -> int:
        return sum(
            [i.size.eval() for i in self.focused_model.pymc_model.free_RVs]  # type: ignore
        )

    def make_serializable(self) -> None:
        for model in self.regression_models.values():
            del model.pymc_model  # type: ignore

    @classmethod
    def from_args(cls, args: Any) -> "NormHBR":
        norm_conf = NormConf.from_args(args)
        hbrconf = HBRConf.from_args(args)
        self = cls(norm_conf, hbrconf)
        return self

    def normdata_to_hbrdata(self, data: NormData) -> hbr_data.HBRData:
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

        assert (len(data.y.shape) == 1) or (data.y.shape[1] == 1), "Only one response variable is supported for HBRdata"

        hbrdata = hbr_data.HBRData(
            X=this_X,
            y=this_y,
            batch_effects=self.map_batch_effects(data),
            unique_batch_effects=self.unique_batch_effects,
            response_var=data.response_vars.to_numpy().item(),
            covariate_dims=this_covariate_dims,
            datapoint_coords=data.datapoints.to_numpy().tolist(),
        )
        return hbrdata
