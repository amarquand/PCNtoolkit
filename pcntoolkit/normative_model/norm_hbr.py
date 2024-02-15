import gc
import json
import os
import warnings
from typing import Union

import arviz as az
import numpy as np
import pymc as pm
import pytensor.tensor as pt
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

        # Make a new model if needed
        if make_new_model or (not self.current_regression_model.pymc_model):
            self.current_regression_model.create_pymc_graph(hbrdata)

        # Sample from pymc model
        with self.current_regression_model.pymc_model:
            self.current_regression_model.idata = pm.sample(
                self.current_regression_model.reg_conf.draws,
                tune=self.current_regression_model.reg_conf.tune,
                cores=self.current_regression_model.reg_conf.cores,
                chains=self.current_regression_model.reg_conf.chains,
            )

        # Set the is_fitted flag to True
        self.current_regression_model.is_fitted = True

    def _predict(self, data: NormData) -> NormData:
        # Assert that the model is fitted
        assert (
            self.current_regression_model.is_fitted
        ), "Model must be fitted before predicting."

        # Transform the data to hbrdata
        hbrdata = self.normdata_to_hbrdata(data)

        # Create a new pymc model if needed
        if not self.current_regression_model.pymc_model:
            self.current_regression_model.create_pymc_graph(hbrdata)

        # Set the data in the model
        hbrdata.set_data_in_existing_model(self.current_regression_model.pymc_model)

        if "posterior_predictive" in self.current_regression_model.idata:
            del self.current_regression_model.idata.posterior_predictive

        # Sample from the posterior predictive
        with self.current_regression_model.pymc_model:
            pm.sample_posterior_predictive(
                self.current_regression_model.idata,
                extend_inferencedata=True,
                var_names=self.get_var_names() + ["y_pred"],
            )

    def _fit_predict(self, fit_data: NormData, predict_data: NormData) -> NormData:
        # Transform the data to hbrdata
        fit_hbrdata = self.normdata_to_hbrdata(fit_data)
        predict_hbrdata = self.normdata_to_hbrdata(predict_data)

        # Make a new model if needed
        if not self.current_regression_model.pymc_model:
            self.current_regression_model.create_pymc_graph(fit_hbrdata)

        # Sample from pymc model
        with self.current_regression_model.pymc_model:
            self.current_regression_model.idata = pm.sample(
                self.current_regression_model.reg_conf.draws,
                tune=self.current_regression_model.reg_conf.tune,
                cores=self.current_regression_model.reg_conf.cores,
                chains=self.current_regression_model.reg_conf.chains,
            )

        # Set the is_fitted flag to True
        self.current_regression_model.is_fitted = True

        # Set the data in the model
        predict_hbrdata.set_data_in_existing_model(
            self.current_regression_model.pymc_model
        )

        # Sample from the posterior predictive
        with self.current_regression_model.pymc_model:
            pm.sample_posterior_predictive(
                self.current_regression_model.idata,
                extend_inferencedata=True,
                var_names=self.get_var_names() + ["y_pred"],
            )

    def _transfer(self, data: NormData, *args, **kwargs) -> "HBR":

        freedom = kwargs.get("freedom", 1)
        # Transform the data to hbrdata
        transferdata = self.normdata_to_hbrdata(data)

        # Assert that the model is fitted
        if not self.current_regression_model.is_fitted:
            raise RuntimeError("Model needs to be fitted before it can be transferred")

        new_hbr_model = HBR(self.current_regression_model.name, self.default_reg_conf)

        # Create a new model, using the idata from the original model to inform the priors
        new_hbr_model.create_pymc_graph(
            transferdata, self.current_regression_model.idata, freedom
        )

        # Sample using the new model
        with new_hbr_model.pymc_model:
            new_hbr_model.idata = pm.sample(
                self.current_regression_model.reg_conf.draws,
                tune=self.current_regression_model.reg_conf.tune,
                cores=self.current_regression_model.reg_conf.cores,
                chains=self.current_regression_model.reg_conf.chains,
            )
            new_hbr_model.is_fitted = True

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

    def _quantiles(self, data: NormData, zscores: list, resample=True) -> xr.DataArray:
        var_names = self.get_var_names()

        hbrdata = self.normdata_to_hbrdata(data)

        # Create a new pymc model if needed
        if not self.current_regression_model.pymc_model:
            self.current_regression_model.create_pymc_graph(hbrdata)

        # Set the data in the model
        hbrdata.set_data_in_existing_model(self.current_regression_model.pymc_model)

        # Delete the posterior predictive if it exists
        if "posterior_predictive" in self.current_regression_model.idata:
            del self.current_regression_model.idata.posterior_predictive

        # Sample from the posterior predictive
        with self.current_regression_model.pymc_model:
            pm.sample_posterior_predictive(
                self.current_regression_model.idata,
                extend_inferencedata=True,
                var_names=var_names + ["y_pred"],
            )

        # Extract the posterior predictive
        post_pred = az.extract(
            self.current_regression_model.idata,
            "posterior_predictive",
            var_names=var_names,
        )

        # Separate the samples into a list so that they can be unpacked
        array_of_vars = list(map(lambda x: np.squeeze(post_pred[x]), var_names))

        # Create an array to hold the quantiles
        n_datapoints, _, n_mcmc_samples = post_pred["mu_samples"].shape
        quantiles = np.zeros((len(zscores), n_datapoints, n_mcmc_samples))

        # Compute the quantile iteratively for each z-score
        for i, j in enumerate(zscores):
            zs = np.full((n_datapoints, n_mcmc_samples), j, dtype=float)
            quantiles[i] = xr.apply_ufunc(
                self.quantile,
                *array_of_vars,
                kwargs={"zs": zs},
            )
        pass

        return xr.DataArray(
            quantiles,
            dims=["quantile_zscores", "datapoints", "sample"],
            coords={"quantile_zscores": zscores},
        ).mean(dim="sample")

    def _zscores(self, data: NormData, resample=False) -> xr.DataArray:
        var_names = self.get_var_names()
        hbrdata = self.normdata_to_hbrdata(data)

        if resample:
            # Create a new pymc model if needed
            if self.current_regression_model.pymc_model is None:
                self.current_regression_model.create_pymc_graph(hbrdata)

            # Set the data in the model
            hbrdata.set_data_in_existing_model(self.current_regression_model.pymc_model)

            # Delete the posterior predictive if it exists
            if "posterior_predictive" in self.current_regression_model.idata:
                del self.current_regression_model.idata.posterior_predictive

            # Sample from the posterior predictive
            with self.current_regression_model.pymc_model:
                pm.sample_posterior_predictive(
                    self.current_regression_model.idata,
                    extend_inferencedata=True,
                    var_names=var_names + ["y_pred"],
                )

        # Extract the posterior predictive
        post_pred = az.extract(
            self.current_regression_model.idata,
            "posterior_predictive",
            var_names=var_names,
        )

        # Separate the samples into a list so that they can be unpacked
        array_of_vars = list(map(lambda x: np.squeeze(post_pred[x]), var_names))

        zscores = xr.apply_ufunc(
            self.zscore,
            *array_of_vars,
            kwargs={"y": hbrdata.y},
        ).mean(dim="sample")

        return zscores

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

        hbrdata = hbr_data.HBRData(
            X=this_X,
            y=this_y,
            batch_effects=data.batch_effects.to_numpy(),
            covariate_dims=this_covariate_dims,
            batch_effect_dims=data.batch_effect_dims.to_numpy(),
            datapoint_coords=data.datapoints.to_numpy(),
        )
        hbrdata.set_batch_effects_maps(data.attrs["batch_effects_maps"])
        return hbrdata

    def get_var_names(self):
        likelihood = self.current_regression_model.reg_conf.likelihood
        # Determine the variables to predict
        if likelihood == "Normal":
            var_names = ["mu_samples", "sigma_samples"]
        elif likelihood.startswith("SHASH"):
            var_names = [
                "mu_samples",
                "sigma_samples",
                "epsilon_samples",
                "delta_samples",
            ]

        else:
            raise RuntimeError("Unsupported likelihood " + likelihood)
        return var_names

    def quantile(self, mu, sigma, epsilon=None, delta=None, zs=0):
        """Auxiliary function for computing quantiles"""
        """Get the zs'th quantiles given likelihood parameters"""
        likelihood = self.current_regression_model.reg_conf.likelihood
        if likelihood == "SHASHo":
            quantiles = S_inv(zs, epsilon, delta) * sigma + mu
        elif likelihood == "SHASHo2":
            sigma_d = sigma / delta
            quantiles = S_inv(zs, epsilon, delta) * sigma_d + mu
        elif likelihood == "SHASHb":
            true_mu = m(epsilon, delta, 1)
            true_sigma = np.sqrt((m(epsilon, delta, 2) - true_mu**2))
            SHASH_c = (S_inv(zs, epsilon, delta) - true_mu) / true_sigma
            quantiles = SHASH_c * sigma + mu
        elif likelihood == "Normal":
            quantiles = zs * sigma + mu
        else:
            exit("Unsupported likelihood")
        return quantiles

    def zscore(self, mu, sigma, epsilon=None, delta=None, y=None):
        """Auxiliary function for computing z-scores"""
        """Get the z-scores of Y, given likelihood parameters"""
        likelihood = self.current_regression_model.reg_conf.likelihood
        if likelihood == "SHASHo":
            SHASH = (y - mu) / sigma
            Z = np.sinh(np.arcsinh(SHASH) * delta - epsilon)
        elif likelihood == "SHASHo2":
            sigma_d = sigma / delta
            SHASH = (y - mu) / sigma_d
            Z = np.sinh(np.arcsinh(SHASH) * delta - epsilon)
        elif likelihood == "SHASHb":
            true_mu = m(epsilon, delta, 1)
            true_sigma = np.sqrt((m(epsilon, delta, 2) - true_mu**2))
            SHASH_c = (y - mu) / sigma
            SHASH = SHASH_c * true_sigma + true_mu
            Z = np.sinh(np.arcsinh(SHASH) * delta - epsilon)
        elif likelihood == "Normal":
            Z = (y - mu) / sigma
        else:
            exit("Unsupported likelihood")
        return Z
