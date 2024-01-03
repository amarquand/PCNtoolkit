from typing import Tuple

import pymc as pm
import numpy as np
from pcntoolkit.regression_model.hbr.hbr_data import HBRData
from pcntoolkit.regression_model.hbr.paramconf import ParamConf
from pcntoolkit.regression_model.hbr.prior import DeterministicNode, LinearNode, Prior, DistWrapper
from .hbr_conf import HBRConf


class HBR:

    def __init__(self, conf: HBRConf):
        """
        Initializes the model.
        Any mutable parameters should be initialized here.
        Any immutable parameters should be initialized in the configuration.
        """
        self._conf: HBRConf = conf
        self.current_param_conf: ParamConf = None
        self.is_fitted: bool = False
        self.idata = None

    @property
    def conf(self) -> HBRConf:
        return self._conf

    def fit(self, data: HBRData):
        """
        Fits the model.
        """
        if not self.is_fitted:

            # Create pymc model
            self.create_pymc_model(data)

            # Sample from pymc model
            with self.model:
                self.idata = pm.sample(
                    self.conf.n_samples, tune=self.conf.n_tune, cores=self.conf.n_cores)

            self.is_fitted = True

        else:
            raise RuntimeError("Model is already fitted.")

    def create_pymc_model(self, data: HBRData) -> HBRData:
        """
        Creates the pymc model.
        """
        if self.conf.likelihood == "Normal":
            self.create_pymc_model_normal(data)
        else:
            raise NotImplementedError(
                f"Likelihood {self.conf.likelihood} not implemented for {self.__class__.__name__}")

    def create_pymc_model_normal(self, data: HBRData) -> pm.Model:
        """
        Creates the pymc model with normal likelihood.
        """

        # Create model
        self.model = pm.Model(coords=data.coords,
                              coords_mutable=data.coords_mutable)

        # Add data to pymc model
        data.add_to_pymc_model(self.model)

        # Create likelihood parameters
        with self.model:
            mu = self.build_param(self.conf.mu, data)
            sigma = self.build_param(self.conf.mu, data)
            mu_samples = pm.Deterministic("mu_samples", mu.get_samples(data))
            sigma_samples = pm.Deterministic(
                "sigma_samples", sigma.get_samples(data))
            print(f"{mu_samples.shape.eval()=}")
            print(f"{sigma_samples.shape.eval()=}")
            # Create likelihood
            likelihood = pm.Normal(
                "likelihood", mu=mu_samples, sigma=sigma_samples, observed=data.pm_y, dims=('datapoints', 'response_vars'))

    def build_param(self, conf: ParamConf, data: HBRData, dims=()):
        """
        Creates a parameter.
        Always returns either an array with the number of data samples in the 0th dimension, or a scalar.
        """
        # If the parameter has a linear effect
        if getattr(self.c, conf.linear, False):

            return self.linear_parameter(name, data, dims)

        # If the parameter has a random effect
        elif getattr(self.conf, f"random_{name}", False):

            # If the parameter is centered
            if getattr(self.conf, f"centered_{name}", False):
                return self.centered_random_parameter(name, data, dims)

            # If the parameter is non-centered
            else:
                return self.non_centered_random_parameter(name, data, dims)
        # This parameter is fixed, i.e. not linear or random
        else:
            return self.fixed_parameter(name, data, dims)

    def fixed_parameter(self, name: str, data: HBRData, dims: Tuple[str] = ()):
        """
        Creates a fixed prior.
        """
        return Prior(self, name, dims)

    def centered_random_parameter(self, name: str,  data: HBRData, dims: Tuple[str]):

        mu_prior = Prior(self, f"mu_{name}", dims)
        sigma_prior = Prior(self, f"sigma_{name}", dims)
        node_dims = (*data.batch_effect_dims, *dims)

        with self.model:
            node = pm.Normal(name, mu=mu_prior.dist,
                             sigma=sigma_prior.dist, dims=node_dims)

        distwrapper = DistWrapper(self, dims=node_dims)
        distwrapper.dist = node
        return distwrapper

    def non_centered_random_parameter(self, name: str,  data: HBRData, dims: Tuple[str]):
        mu_prior = Prior(self, f"mu_{name}", dims)

        sigma_prior = Prior(self, f"sigma_{name}", dims)

        offset_dims = (*data.batch_effect_dims, *dims)

        offset_prior = Prior(
            self, f"offset_{name}", offset_dims)

        node = DeterministicNode(self, name, lambda: mu_prior.dist +
                                 offset_prior.dist * sigma_prior.dist, dims=offset_dims)
        return node

    def linear_parameter(self, name: str, data: HBRData, dims: Tuple[str]):
        """
        Creates a linear prior.
        """
        # Create a prior for the slope
        slope_dims = (*dims, 'covariates')
        slope_prior = self.build_param(f"slope_{name}", data, slope_dims)

        # Create a prior for the intercept
        intercept_prior = self.build_param(f"intercept_{name}", data, dims)

        linear_parameter = LinearNode(self, slope_prior,
                                      intercept_prior)

        return linear_parameter

    def predict(self, data: HBRData) -> HBRData:
        """
        Predicts on new data.
        """
        # some prediction logic
        # ...
        raise NotImplementedError(
            f"Predict method not implemented for {self.__class__.__name__}")

    def fit_predict(self, fit_data: HBRData, predict_data: HBRData) -> HBRData:
        """
        Fits and predicts the model.
        """
        # some fit_predict logic
        # ...
        raise NotImplementedError(
            f"Fit-predict method not implemented for {self.__class__.__name__}")
