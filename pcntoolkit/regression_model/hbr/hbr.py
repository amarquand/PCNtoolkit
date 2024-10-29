from __future__ import annotations

import os

import arviz as az
import numpy as np
import pymc as pm
import scipy.stats as stats
import xarray as xr

from pcntoolkit.regression_model.hbr.hbr_data import HBRData
from pcntoolkit.regression_model.hbr.hbr_util import centile, zscore
from pcntoolkit.regression_model.hbr.param import Param
from pcntoolkit.regression_model.hbr.shash import SHASHb, SHASHo
from pcntoolkit.regression_model.regression_model import RegressionModel

from .hbr_conf import HBRConf


class HBR(RegressionModel):
    def __init__(
        self, name: str, reg_conf: HBRConf, is_fitted=False, is_from_dict=False
    ):
        """
        Initializes the model.
        Any mutable parameters should be initialized here.
        Any immutable parameters should be initialized in the configuration.
        """
        super().__init__(name, reg_conf, is_fitted, is_from_dict)
        self.idata: az.InferenceData = None
        self.pymc_model = None

    def fit(self, hbrdata: HBRData, make_new_model: bool = True):
        # Make a new model if needed
        if make_new_model or (not self.pymc_model):
            self.create_pymc_graph(hbrdata)

        # Sample from pymc model
        with self.pymc_model:
            self.idata = pm.sample(
                self.reg_conf.draws,
                tune=self.reg_conf.tune,
                cores=self.reg_conf.cores,
                chains=self.reg_conf.chains,
                nuts_sampler=self.reg_conf.nuts_sampler,
                # var_names=["y_pred"],
            )

        # Set the is_fitted flag to True
        self.is_fitted = True

    def predict(self, hbrdata: HBRData):
        # Create a new pymc model if needed
        if not self.pymc_model:
            self.create_pymc_graph(hbrdata)

        # Set the data in the model
        hbrdata.set_data_in_existing_model(self.pymc_model)

        if "posterior_predictive" in self.idata:
            del self.idata.posterior_predictive

        # Sample from the posterior predictive
        with self.pymc_model:
            pm.sample_posterior_predictive(
                self.idata,
                extend_inferencedata=True,
                var_names=self.get_var_names() + ["y_pred"],
            )

    def fit_predict(self, fit_hbrdata: HBRData, predict_hbrdata: HBRData):
        # Make a new model if needed
        if not self.pymc_model:
            self.create_pymc_graph(fit_hbrdata)

        # Sample from pymc model
        with self.pymc_model:
            self.idata = pm.sample(
                self.reg_conf.draws,
                tune=self.reg_conf.tune,
                cores=self.reg_conf.cores,
                chains=self.reg_conf.chains,
                nuts_sampler=self.reg_conf.nuts_sampler,
            )

        # Set the is_fitted flag to True
        self.is_fitted = True

        # Set the data in the model
        predict_hbrdata.set_data_in_existing_model(self.pymc_model)

        # Sample from the posterior predictive
        with self.pymc_model:
            pm.sample_posterior_predictive(
                self.idata,
                extend_inferencedata=True,
                var_names=self.get_var_names() + ["y_pred"],
            )

    def transfer(self, hbrconf, transferdata, freedom):
        new_hbr_model = HBR(self.name, hbrconf)

        # new_hbr_model.transfer(transferdata, freedom)

        # Create a new model, using the idata from the original model to inform the priors
        new_hbr_model.create_pymc_graph(transferdata, self.idata, freedom)

        # Sample using the new model
        with new_hbr_model.pymc_model:
            new_hbr_model.idata = pm.sample(
                hbrconf.draws,
                tune=hbrconf.tune,
                cores=hbrconf.cores,
                chains=hbrconf.chains,
            )
            new_hbr_model.is_fitted = True

        return new_hbr_model

    def centiles(
        self, hbrdata: HBRData, cummulative_densities: list[float], resample=True
    ) -> xr.DataArray:
        var_names = self.get_var_names()

        # Create a new pymc model if needed
        if not self.pymc_model:
            self.create_pymc_graph(hbrdata)

        # Set the data in the model
        hbrdata.set_data_in_existing_model(self.pymc_model)

        # Delete the posterior predictive if it exists
        if "posterior_predictive" in self.idata:
            del self.idata.posterior_predictive

        # Sample from the posterior predictive
        with self.pymc_model:
            pm.sample_posterior_predictive(
                self.idata,
                extend_inferencedata=True,
                var_names=var_names + ["y_pred"],
            )

        # Extract the posterior predictive
        post_pred = az.extract(
            self.idata,
            "posterior_predictive",
            var_names=var_names,
        )

        # Separate the samples into a list so that they can be unpacked
        array_of_vars = [self.likelihood] + list(
            map(lambda x: np.squeeze(post_pred[x]), var_names)
        )

        # Create an array to hold the centiles
        n_datapoints, n_mcmc_samples = post_pred["mu_samples"].shape
        centiles = np.zeros((len(cummulative_densities), n_datapoints, n_mcmc_samples))

        # Compute the centiles iteratively for each cummulative density
        for i, cdf in enumerate(cummulative_densities):
            zs = np.full(
                (n_datapoints, n_mcmc_samples), stats.norm.ppf(cdf), dtype=float
            )
            centiles[i] = xr.apply_ufunc(
                centile,
                *array_of_vars,
                kwargs={"zs": zs},
            )
        pass

        return xr.DataArray(
            centiles,
            dims=["cummulative_densities", "datapoints", "sample"],
            coords={"cummulative_densities": cummulative_densities},
        ).mean(dim="sample")

    def zscores(self, hbrdata: HBRData, resample=False) -> xr.DataArray:
        var_names = self.get_var_names()
        if resample:
            # Create a new pymc model if needed
            if self.pymc_model is None:
                self.create_pymc_graph(hbrdata)

            # Set the data in the model
            hbrdata.set_data_in_existing_model(self.pymc_model)

            # Delete the posterior predictive if it exists
            if "posterior_predictive" in self.idata:
                del self.idata.posterior_predictive

            # Sample from the posterior predictive
            with self.pymc_model:
                pm.sample_posterior_predictive(
                    self.idata,
                    extend_inferencedata=True,
                    var_names=var_names + ["y_pred"],
                )

        # Extract the posterior predictive
        post_pred = az.extract(
            self.idata,
            "posterior_predictive",
            var_names=var_names,
        )

        # Separate the samples into a list so that they can be unpacked
        array_of_vars = [self.likelihood] + list(
            map(lambda x: np.squeeze(post_pred[x]), var_names)
        )

        zscores = xr.apply_ufunc(
            zscore,
            *array_of_vars,
            kwargs={"y": hbrdata.y[:, None]},
        ).mean(dim="sample")

        return zscores

    def get_var_names(self):
        likelihood = self.reg_conf.likelihood
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

    def create_pymc_graph(
        self, data: HBRData, idata: az.InferenceData = None, freedom=1
    ) -> HBRData:
        """
        Creates the pymc model.
        """
        self.pymc_model = pm.Model(coords=data.coords)
        data.add_to_graph(self.pymc_model)
        if self.reg_conf.likelihood == "Normal":
            self.create_normal_pymc_graph(data, idata, freedom)
        elif self.reg_conf.likelihood == "SHASHb":
            self.create_SHASHb_pymc_graph(data, idata, freedom)
        elif self.reg_conf.likelihood == "SHASHo":
            self.create_SHASHo_pymc_graph(data, idata, freedom)
        else:
            raise NotImplementedError(
                f"Likelihood {self.reg_conf.likelihood} not implemented for {self.__class__.__name__}"
            )

    def create_normal_pymc_graph(
        self, data: HBRData, idata: az.InferenceData = None, freedom=1
    ) -> HBRData:
        """
        Creates the pymc model.
        """
        self.reg_conf.mu.create_graph(self.pymc_model, idata, freedom)
        self.reg_conf.sigma.create_graph(self.pymc_model, idata, freedom)
        with self.pymc_model:
            mu_samples = pm.Deterministic(
                "mu_samples",
                self.reg_conf.mu.get_samples(data),
                dims=self.reg_conf.mu.sample_dims,
            )
            sigma_samples = pm.Deterministic(
                "sigma_samples",
                self.reg_conf.sigma.get_samples(data),
                dims=self.reg_conf.sigma.sample_dims,
            )
            y_pred = pm.Normal(
                "y_pred",
                mu=mu_samples,
                sigma=sigma_samples,
                observed=data.pm_y,
                dims=("datapoints",),
            )

    def create_SHASHb_pymc_graph(
        self, data: HBRData, idata: az.InferenceData = None, freedom=1
    ) -> HBRData:
        """
        Creates the pymc model.
        """
        self.reg_conf.mu.create_graph(self.pymc_model, idata, freedom)
        self.reg_conf.sigma.create_graph(self.pymc_model, idata, freedom)
        self.reg_conf.epsilon.create_graph(self.pymc_model, idata, freedom)
        self.reg_conf.delta.create_graph(self.pymc_model, idata, freedom)
        with self.pymc_model:
            mu_samples = pm.Deterministic(
                "mu_samples",
                self.reg_conf.mu.get_samples(data),
                dims=self.reg_conf.mu.sample_dims,
            )
            sigma_samples = pm.Deterministic(
                "sigma_samples",
                self.reg_conf.sigma.get_samples(data),
                dims=self.reg_conf.sigma.sample_dims,
            )
            epsilon_samples = pm.Deterministic(
                "epsilon_samples",
                self.reg_conf.epsilon.get_samples(data),
                dims=self.reg_conf.epsilon.sample_dims,
            )
            delta_samples = pm.Deterministic(
                "delta_samples",
                self.reg_conf.delta.get_samples(data),
                dims=self.reg_conf.delta.sample_dims,
            )
            y_pred = SHASHb(
                "y_pred",
                mu=mu_samples,
                sigma=sigma_samples,
                epsilon=epsilon_samples,
                delta=delta_samples,
                observed=data.pm_y,
                dims=("datapoints",),
            )

    def create_SHASHo_pymc_graph(
        self, data: HBRData, idata: az.InferenceData = None, freedom=1
    ) -> HBRData:
        """
        Creates the pymc model.
        """
        self.reg_conf.mu.create_graph(self.pymc_model, idata, freedom)
        self.reg_conf.sigma.create_graph(self.pymc_model, idata, freedom)
        self.reg_conf.epsilon.create_graph(self.pymc_model, idata, freedom)
        self.reg_conf.delta.create_graph(self.pymc_model, idata, freedom)
        with self.pymc_model:
            mu_samples = pm.Deterministic(
                "mu_samples",
                self.reg_conf.mu.get_samples(data),
                self.reg_conf.mu.sample_dims,
            )
            sigma_samples = pm.Deterministic(
                "sigma_samples",
                self.reg_conf.sigma.get_samples(data),
                self.reg_conf.sigma.sample_dims,
            )
            epsilon_samples = pm.Deterministic(
                "epsilon_samples",
                self.reg_conf.epsilon.get_samples(data),
                dims=("datapoints", "response_vars"),
            )
            delta_samples = pm.Deterministic(
                "delta_samples",
                self.reg_conf.delta.get_samples(data),
                self.reg_conf.delta.sample_dims,
            )
            y_pred = SHASHo(
                "y_pred",
                mu=mu_samples,
                sigma=sigma_samples,
                epsilon=epsilon_samples,
                delta=delta_samples,
                observed=data.pm_y,
                dims=("datapoints",),
            )

    def to_dict(self, path=None):
        """
        Converts the regression model to a dictionary.
        """
        my_dict = super().to_dict()
        if self.is_fitted and (path is not None):
            idata_path = os.path.join(path, f"idata_{self.name}.nc")
            self.save_idata(idata_path)
        return my_dict

    @classmethod
    def from_dict(cls, dict, path=None):
        """
        Creates a configuration from a dictionary.
        Takes an optional path argument to load idata from.
        """
        name = dict["name"]
        conf = HBRConf.from_dict(dict["reg_conf"])
        is_fitted = dict["is_fitted"]
        is_from_dict = True
        self = cls(name, conf, is_fitted, is_from_dict)
        if is_fitted and (path is not None):
            idata_path = os.path.join(path, f"idata_{name}.nc")
            self.load_idata(idata_path)
        return self

    @classmethod
    def from_args(cls, name, args):
        """
        Creates a configuration from command line arguments
        """
        conf = HBRConf.from_args(args)
        is_fitted = args.get("is_fitted", False)
        is_from_dict = True
        self = cls(name, conf, is_fitted, is_from_dict)
        return self

    def save_idata(self, path):
        if self.is_fitted:
            if hasattr(self, "idata"):
                self.remove_samples_from_idata_posterior()
                self.idata.to_netcdf(path, groups="posterior")
            else:
                raise RuntimeError(
                    "HBR model is fitted but does not have idata. This should not happen."
                )

    def load_idata(self, path):
        if self.is_fitted:
            try:
                self.idata = az.from_netcdf(path)
            except:
                raise RuntimeError(f"Could not load idata from {path}.")
            self.replace_samples_in_idata_posterior()

    def remove_samples_from_idata_posterior(self):
        for name in self.idata.posterior.variables.mapping.keys():
            if name.endswith("_samples"):
                self.idata.posterior.drop_vars(name)
                if "removed_samples" not in self.idata.attrs:
                    self.idata.attrs["removed_samples"] = []
                self.idata.attrs["removed_samples"].append(name)

    def replace_samples_in_idata_posterior(self):
        for name in self.idata.attrs["removed_samples"]:
            samples = np.zeros(self.idata.posterior[name].shape)
            self.idata.posterior[name] = xr.DataArray(
                samples,
                dims=self.idata.posterior[name].dims,
            )

    @property
    def mu(self) -> Param:
        return self.reg_conf.mu

    @property
    def sigma(self) -> Param:
        return self.reg_conf.sigma

    @property
    def epsilon(self) -> Param:
        return self.reg_conf.epsilon

    @property
    def delta(self) -> Param:
        return self.reg_conf.delta

    @property
    def likelihood(self) -> str:
        return self.reg_conf.likelihood
