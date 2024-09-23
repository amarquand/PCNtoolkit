import gc
import json
import os
import warnings
from typing import Union

import arviz as az
import numpy as np
import pymc as pm
import pytensor.tensor as pt
import scipy.stats as stats
import xarray as xr

from pcntoolkit.dataio.norm_data import NormData
from pcntoolkit.normative_model.norm_base import NormBase
from pcntoolkit.normative_model.norm_conf import NormConf
from pcntoolkit.regression_model.hbr import hbr_data
from pcntoolkit.regression_model.hbr.hbr import HBR
from pcntoolkit.regression_model.hbr.hbr_conf import HBRConf
from pcntoolkit.regression_model.hbr.hbr_util import S_inv, m
from pcntoolkit.regression_model.reg_conf import RegConf


class NormHBR(NormBase):
    def __init__(self, norm_conf: NormConf, reg_conf: HBRConf = None):
        super().__init__(norm_conf)
        if reg_conf is None:
            reg_conf = HBRConf
        self.default_reg_conf: HBRConf = reg_conf
        self.regression_model_type = HBR
        self.current_regression_model: HBR = None

    def _fit(self, data: NormData, make_new_model=False):
        # Transform the data to hbrdata
        hbrdata = self.normdata_to_hbrdata(data)

        self.current_regression_model.fit(hbrdata, make_new_model)


    def _predict(self, data: NormData) -> NormData:
        # Assert that the model is fitted
        assert (
            self.current_regression_model.is_fitted
        ), "Model must be fitted before predicting."

        # Transform the data to hbrdata
        hbrdata = self.normdata_to_hbrdata(data)

        self.current_regression_model.predict(hbrdata)


    def _fit_predict(self, fit_data: NormData, predict_data: NormData) -> NormData:


        # Transform the data to hbrdata
        fit_hbrdata = self.normdata_to_hbrdata(fit_data)
        predict_hbrdata = self.normdata_to_hbrdata(predict_data)

        self.current_regression_model.fit_predict(fit_hbrdata, predict_hbrdata)

    def _transfer(self, data: NormData, *args, **kwargs) -> "HBR":

        freedom = kwargs.get("freedom", 1)
        # Transform the data to hbrdata
        transferdata = self.normdata_to_hbrdata(data)

        # Assert that the model is fitted
        if not self.current_regression_model.is_fitted:
            raise RuntimeError("Model needs to be fitted before it can be transferred")
        
        new_hbr_model = self.current_regression_model.transfer(self.default_reg_conf, transferdata, freedom)


        # Return the new model
        return new_hbr_model

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

    def _centiles(
        self, data: NormData, cummulative_densities: list[float], resample=True
    ) -> xr.DataArray:

        hbrdata = self.normdata_to_hbrdata(data)

        return self.current_regression_model.centiles(hbrdata, cummulative_densities, resample)

    

    def _zscores(self, data: NormData, resample=False) -> xr.DataArray:
        hbrdata = self.normdata_to_hbrdata(data)

        return self.current_regression_model.zscores(hbrdata, resample)



    def n_params(self):
        return sum(
            [i.size.eval() for i in self.current_regression_model.pymc_model.free_RVs]
        )

    @classmethod
    def from_args(cls, args):
        """
        Creates a configuration from command line arguments.
        """
        norm_conf = NormConf.from_args(args)
        hbrconf = HBRConf.from_args(args)
        self = cls(norm_conf, hbrconf)
        return self

    @staticmethod
    def normdata_to_hbrdata(data: NormData) -> hbr_data.HBRData:
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

        assert (len(data.y.shape)==1) or (data.y.shape[1] == 1), "Only one response variable is supported for HBRdata"

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

