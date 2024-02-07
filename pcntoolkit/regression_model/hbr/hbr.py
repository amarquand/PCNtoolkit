from __future__ import annotations

import os
from typing import Tuple

import arviz as az
import numpy as np
import pymc as pm
import xarray as xr

from pcntoolkit.regression_model.hbr.hbr_data import HBRData
from pcntoolkit.regression_model.hbr.param import Param
from pcntoolkit.regression_model.hbr.shash import SHASHb, SHASHo

from .hbr_conf import HBRConf


class HBR:
    def __init__(self, conf: HBRConf):
        """
        Initializes the model.
        Any mutable parameters should be initialized here.
        Any immutable parameters should be initialized in the configuration.
        """
        self._conf: HBRConf = conf
        self.is_fitted: bool = False
        self.idata: az.InferenceData = None
        self.is_from_dict = False
        self.pymc_model = None

    @property
    def conf(self) -> HBRConf:
        return self._conf

    def create_pymc_model(
        self, data: HBRData, idata: az.InferenceData = None, freedom=1
    ) -> HBRData:
        """
        Creates the pymc model.
        """
        self.pymc_model = pm.Model(
            coords=data.coords, coords_mutable=data.coords_mutable
        )
        data.add_to_model(self.pymc_model)
        if self.conf.likelihood == "Normal":
            self.create_normal_pymc_model(data, idata, freedom)
        elif self.conf.likelihood == "SHASHb":
            self.create_SHASHb_pymc_model(data, idata, freedom)
        elif self.conf.likelihood == "SHASHo":
            self.create_SHASHo_pymc_model(data, idata, freedom)
        else:
            raise NotImplementedError(
                f"Likelihood {self.conf.likelihood} not implemented for {self.__class__.__name__}"
            )

    def create_normal_pymc_model(
        self, data: HBRData, idata: az.InferenceData = None, freedom=1
    ) -> HBRData:
        """
        Creates the pymc model.
        """
        self.conf.mu.create_graph(self.pymc_model, idata, freedom)
        self.conf.sigma.create_graph(self.pymc_model, idata, freedom)
        with self.pymc_model:
            mu_samples = pm.Deterministic(
                "mu_samples",
                self.conf.mu.get_samples(data),
                dims=("datapoints", "response_vars"),
            )
            sigma_samples = pm.Deterministic(
                "sigma_samples",
                self.conf.sigma.get_samples(data),
                dims=("datapoints", "response_vars"),
            )
            y_pred = pm.Normal(
                "y_pred",
                mu=mu_samples,
                sigma=sigma_samples,
                dims=("datapoints", "response_vars"),
                observed=data.pm_y,
            )

    def create_SHASHb_pymc_model(
        self, data: HBRData, idata: az.InferenceData = None, freedom=1
    ) -> HBRData:
        """
        Creates the pymc model.
        """
        self.conf.mu.create_graph(self.pymc_model, idata, freedom)
        self.conf.sigma.create_graph(self.pymc_model, idata, freedom)
        self.conf.epsilon.create_graph(self.pymc_model, idata, freedom)
        self.conf.delta.create_graph(self.pymc_model, idata, freedom)
        with self.pymc_model:
            mu_samples = pm.Deterministic(
                "mu_samples",
                self.conf.mu.get_samples(data),
                dims=("datapoints", "response_vars"),
            )
            sigma_samples = pm.Deterministic(
                "sigma_samples",
                self.conf.sigma.get_samples(data),
                dims=("datapoints", "response_vars"),
            )
            epsilon_samples = pm.Deterministic(
                "epsilon_samples",
                self.conf.epsilon.get_samples(data),
                dims=("datapoints", "response_vars"),
            )
            delta_samples = pm.Deterministic(
                "delta_samples",
                self.conf.delta.get_samples(data),
                dims=("datapoints", "response_vars"),
            )
            y_pred = SHASHb(
                "y_pred",
                mu=mu_samples,
                sigma=sigma_samples,
                epsilon=epsilon_samples,
                delta=delta_samples,
                dims=("datapoints", "response_vars"),
                observed=data.pm_y,
            )

    def create_SHASHo_pymc_model(
        self, data: HBRData, idata: az.InferenceData = None, freedom=1
    ) -> HBRData:
        """
        Creates the pymc model.
        """
        self.conf.mu.create_graph(self.pymc_model, idata, freedom)
        self.conf.sigma.create_graph(self.pymc_model, idata, freedom)
        self.conf.epsilon.create_graph(self.pymc_model, idata, freedom)
        self.conf.delta.create_graph(self.pymc_model, idata, freedom)
        with self.pymc_model:
            mu_samples = pm.Deterministic(
                "mu_samples",
                self.conf.mu.get_samples(data),
                dims=("datapoints", "response_vars"),
            )
            sigma_samples = pm.Deterministic(
                "sigma_samples",
                self.conf.sigma.get_samples(data),
                dims=("datapoints", "response_vars"),
            )
            epsilon_samples = pm.Deterministic(
                "epsilon_samples",
                self.conf.epsilon.get_samples(data),
                dims=("datapoints", "response_vars"),
            )
            delta_samples = pm.Deterministic(
                "delta_samples",
                self.conf.delta.get_samples(data),
                dims=("datapoints", "response_vars"),
            )
            y_pred = SHASHo(
                "y_pred",
                mu=mu_samples,
                sigma=sigma_samples,
                epsilon=epsilon_samples,
                delta=delta_samples,
                dims=("datapoints", "response_vars"),
                observed=data.pm_y,
            )

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
                self.replace_samples_in_idata_posterior()
            except:
                raise RuntimeError(f"Could not load idata from {path}.")

    def remove_samples_from_idata_posterior(self):
        for name in self.idata.posterior.variables.mapping.keys():
            if name.endswith("_samples"):
                self.idata.posterior.drop_vars(name)
                if "removed_samples" not in self.idata.attrs:
                    self.idata.attrs["removed_samples"] = []
                self.idata.attrs["removed_samples"].append(name)

    def replace_samples_in_idata_posterior(self):
        for name in self.idata.attrs["removed_samples"]:
            samples = np.zeros(
                (
                    self.idata.posterior.chain.size,
                    self.idata.posterior.draw.size,
                    self.idata.posterior.datapoints.size,
                    self.idata.posterior.response_vars.size,
                )
            )
            self.idata.posterior[name] = xr.DataArray(
                samples,
                dims=["chain", "draw", "datapoints", "response_vars"],
            )

    def to_dict(self):
        """
        Converts the configuration to a dictionary.
        """
        my_dict = {}
        my_dict["conf"] = self.conf.to_dict()
        my_dict["is_fitted"] = self.is_fitted
        my_dict["is_from_dict"] = self.is_from_dict
        return my_dict

    @classmethod
    def from_dict(cls, dict):
        """
        Creates a configuration from a dictionary.
        """
        conf = HBRConf.from_dict(dict["conf"])
        self = cls(conf)
        self.is_fitted = dict["is_fitted"]
        self.is_from_dict = True
        return self

    @classmethod
    def from_args(cls, args):
        """
        Creates a configuration from command line arguments
        """
        conf = HBRConf.from_args(args)
        self = cls(conf)
        self.is_from_dict = True
        self.is_fitted = args.get("is_fitted", False)
        return self

    @property
    def mu(self) -> Param:
        return self.conf.mu

    @property
    def sigma(self) -> Param:
        return self.conf.sigma

    @property
    def epsilon(self) -> Param:
        return self.conf.epsilon

    @property
    def delta(self) -> Param:
        return self.conf.delta

    @property
    def likelihood(self) -> str:
        return self.conf.likelihood
