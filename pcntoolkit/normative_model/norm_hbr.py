import json
import os
import warnings

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


class NormHBR(NormBase):
    def __init__(self, norm_conf: NormConf, reg_conf: HBRConf):
        super().__init__(norm_conf)
        self.reg_conf: HBRConf = reg_conf
        self.model_type = HBR
        self.model: HBR = None

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
    def reg_conf_from_args(args):
        return HBRConf.from_args(args)

    def models_to_dict(self, path):
        regression_model_dict = {}

        for k, v in self.models.items():
            regression_model_dict[k] = v.to_dict()
            del regression_model_dict[k]["conf"]
            if v.is_fitted:
                if hasattr(v, "idata"):
                    idata_path = os.path.join(path, f"idata_{k}.nc")
                    self.model.idata.to_netcdf(idata_path)
                else:
                    raise RuntimeError(
                        "HBR model is fitted but does not have idata. This should not happen."
                    )
        return regression_model_dict

    def dict_to_models(self, dict, path):
        for k, v in dict.items():
            self.models[k] = self.model_type(self.reg_conf)
            self.models[k].is_from_dict = dict[k]["is_from_dict"]
            self.models[k].is_fitted = dict[k]["is_fitted"]
            if self.models[k].is_fitted:
                idata_path = os.path.join(path, f"idata_{k}.nc")
                try:
                    self.models[k].idata = az.from_netcdf(idata_path)
                except:
                    raise RuntimeError(f"Could not load idata from {idata_path}.")

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
        )
        hbrdata.set_batch_effects_maps(data.attrs["batch_effects_maps"])
        return hbrdata

    def _fit(self, data: NormData, make_new_model=False):
        # Transform the data to hbrdata
        hbrdata = self.normdata_to_hbrdata(data)

        # Make a new model if needed
        if make_new_model or (not self.model.model):
            self.model.create_pymc_model(hbrdata)

        # Sample from pymc model
        with self.model.model:
            self.model.idata = pm.sample(
                self.reg_conf.draws,
                tune=self.reg_conf.tune,
                cores=self.reg_conf.cores,
                chains=self.reg_conf.chains,
            )

        # Set the is_fitted flag to True
        self.model.is_fitted = True

    def _predict(self, data: NormData) -> NormData:
        # Assert that the model is fitted
        assert self.model.is_fitted, "Model must be fitted before predicting."

        # Transform the data to hbrdata
        hbrdata = self.normdata_to_hbrdata(data)

        # Create a new pymc model if needed
        if self.model.model is None:
            self.model.create_pymc_model(hbrdata)

        # Set the data in the model
        hbrdata.set_data_in_existing_model(self.model.model)

        # Sample from the posterior predictive
        with self.model.model:
            pm.sample_posterior_predictive(
                self.model.idata,
                extend_inferencedata=True,
                var_names=self.get_var_names(),
            )

    def _fit_predict(self, fit_data: NormData, predict_data: NormData) -> NormData:
        # Transform the data to hbrdata
        fit_hbrdata = self.normdata_to_hbrdata(fit_data)
        predict_hbrdata = self.normdata_to_hbrdata(predict_data)

        # Make a new model if needed
        if not self.model.model:
            self.model.create_pymc_model(fit_hbrdata)

        # Sample from pymc model
        with self.model.model:
            self.model.idata = pm.sample(
                self.reg_conf.draws,
                tune=self.reg_conf.tune,
                cores=self.reg_conf.cores,
                chains=self.reg_conf.chains,
            )

        # Set the is_fitted flag to True
        self.model.is_fitted = True

        # Set the data in the model
        predict_hbrdata.set_data_in_existing_model(self.model.model)

        # Sample from the posterior predictive
        with self.model.model:
            pm.sample_posterior_predictive(
                self.model.idata,
                extend_inferencedata=True,
                var_names=self.get_var_names(),
            )

    def _transfer(self, data: NormData) -> "HBR":
        # Transform the data to hbrdata
        transferdata = self.normdata_to_hbrdata(data)

        # Assert that the model is fitted
        if not self.model.is_fitted:
            raise RuntimeError("Model needs to be fitted before it can be transferred")

        new_hbr_model = HBR(self.reg_conf)

        # Create a new model, using the idata from the original model to inform the priors
        new_hbr_model.create_pymc_model(transferdata, self.model.idata)

        # Sample using the new model
        with new_hbr_model.model:
            new_hbr_model.idata = pm.sample(
                draws=self.reg_conf.draws,
                tune=self.reg_conf.tune,
                cores=self.reg_conf.cores,
                chains=self.reg_conf.chains,
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

    def _quantiles(self, data: NormData, z_scores: list) -> xr.DataArray:
        hbrdata = self.normdata_to_hbrdata(data)

        # Create a new pymc model if needed
        if self.model.model is None:
            self.model.create_pymc_model(hbrdata)

        # Set the data in the model
        hbrdata.set_data_in_existing_model(self.model.model)

        var_names = self.get_var_names()

        # Delete the posterior predictive if it exists
        if "posterior_predictive" in self.model.idata:
            del self.model.idata.posterior_predictive

        # Sample from the posterior predictive
        with self.model.model:
            pm.sample_posterior_predictive(
                self.model.idata, extend_inferencedata=True, var_names=var_names
            )

        # Extract the posterior predictive
        post_pred = az.extract(
            self.model.idata, "posterior_predictive", var_names=var_names
        )

        # Separate the samples into a list so that they can be unpacked
        array_of_vars = list(map(lambda x: np.squeeze(post_pred[x]), var_names))

        # Create an array to hold the quantiles
        n_datapoints, _, n_mcmc_samples = post_pred["mu_samples"].shape
        quantiles = np.zeros((len(z_scores), n_datapoints, n_mcmc_samples))

        # Compute the quantile iteratively for each z-score
        for i, j in enumerate(z_scores):
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
            coords={"quantile_zscores": z_scores},
        ).mean(dim="sample")

    def _zscores(self, data: NormData) -> xr.DataArray:
        hbrdata = self.normdata_to_hbrdata(data)

        # Create a new pymc model if needed
        if self.model.model is None:
            self.model.create_pymc_model(hbrdata)

        # Set the data in the model
        hbrdata.set_data_in_existing_model(self.model.model)

        var_names = self.get_var_names()

        # Delete the posterior predictive if it exists
        if "posterior_predictive" in self.model.idata:
            del self.model.idata.posterior_predictive

        # Sample from the posterior predictive
        with self.model.model:
            pm.sample_posterior_predictive(
                self.model.idata, extend_inferencedata=True, var_names=var_names
            )

        # Extract the posterior predictive
        post_pred = az.extract(
            self.model.idata, "posterior_predictive", var_names=var_names
        )

        # Separate the samples into a list so that they can be unpacked
        array_of_vars = list(map(lambda x: np.squeeze(post_pred[x]), var_names))

        z_scores = xr.apply_ufunc(
            self.z_score,
            *array_of_vars,
            kwargs={"y": hbrdata.y},
        ).mean(dim="sample")

        return z_scores

    def get_var_names(self):
        likelihood = self.reg_conf.likelihood
        # Determine the variables to predict
        if likelihood == "Normal":
            var_names = ["mu_samples", "sigma_samples", "y_pred"]
        elif likelihood.startswith("SHASH"):
            var_names = [
                "mu_samples",
                "sigma_samples",
                "epsilon_samples",
                "delta_samples",
                "y_pred",
            ]

        else:
            raise RuntimeError("Unsupported likelihood " + likelihood)
        return var_names

    def quantile(self, mu, sigma, epsilon=None, delta=None, zs=0):
        """Auxiliary function for computing quantiles"""
        """Get the zs'th quantiles given likelihood parameters"""
        likelihood = self.reg_conf.likelihood
        if likelihood.startswith("SHASH"):
            raise NotImplementedError(
                "Quantiles for SHASH likelihoods are not implemented yet."
            )
            # if likelihood == "SHASHo":
            #     quantiles = S_inv(zs, epsilon, delta) * sigma + mu
            # elif likelihood == "SHASHo2":
            #     sigma_d = sigma / delta
            #     quantiles = S_inv(zs, epsilon, delta) * sigma_d + mu
            # elif likelihood == "SHASHb":
            #     true_mu = m(epsilon, delta, 1)
            #     true_sigma = np.sqrt((m(epsilon, delta, 2) - true_mu**2))
            #     SHASH_c = (S_inv(zs, epsilon, delta) - true_mu) / true_sigma
            #     quantiles = SHASH_c * sigma + mu
        elif likelihood == "Normal":
            quantiles = zs * sigma + mu
        else:
            exit("Unsupported likelihood")
        return quantiles

    def z_score(self, mu, sigma, epsilon=None, delta=None, y=None):
        """Auxiliary function for computing z-scores"""
        """Get the z-scores of Y, given likelihood parameters"""
        likelihood = self.reg_conf.likelihood
        if likelihood.startswith("SHASH"):
            raise NotImplementedError(
                "Z-scores for SHASH likelihoods are not implemented yet."
            )
        #     if likelihood == "SHASHo":
        #         SHASH = (y - mu) / sigma
        #         Z = np.sinh(np.arcsinh(SHASH) * delta - epsilon)
        #     elif likelihood == "SHASHo2":
        #         sigma_d = sigma / delta
        #         SHASH = (y - mu) / sigma_d
        #         Z = np.sinh(np.arcsinh(SHASH) * delta - epsilon)
        #     elif likelihood == "SHASHb":
        #         true_mu = m(epsilon, delta, 1)
        #         true_sigma = np.sqrt((m(epsilon, delta, 2) - true_mu**2))
        #         SHASH_c = (y - mu) / sigma
        #         SHASH = SHASH_c * true_sigma + true_mu
        #         Z = np.sinh(np.arcsinh(SHASH) * delta - epsilon)
        elif likelihood == "Normal":
            Z = (y - mu) / sigma
        else:
            exit("Unsupported likelihood")
        return Z

    def n_params(self):
        return 1
