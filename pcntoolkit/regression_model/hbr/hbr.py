import numpy as np

import pymc as pm

from pcntoolkit.regression_model.hbr.hbr_data import HBRData
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
            self.model = self.create_pymc_model(data)

            # Sample from pymc model
            self.is_fitted = True

        else:
            raise RuntimeError("Model is already fitted.")

    def create_pymc_model(self, data: HBRData) -> HBRData:
        """
        Creates the pymc model.
        """
        #

        if self.conf.likelihood == "Normal":
            return self.create_pymc_model_normal(data)
        else:
            raise NotImplementedError(
                f"Likelihood {self.conf.likelihood} not implemented for {self.__class__.__name__}")

    def create_pymc_model_normal(self, data: HBRData) -> pm.Model:
        """
        Creates the pymc model with normal likelihood.
        """

        # Create model
        with pm.Model() as model:

            # Add data to pymc model
            data.add_to_pymc_model(model)

            # Create priors
            mu = self.create_prior("mu", data)
            sigma = self.create_prior("sigma", data)

            # Create likelihood
            likelihood = pm.Normal(
                "likelihood", mu=mu, sigma=sigma, observed=data.pm_y)

        return model

    def create_prior(self, name: str, data: HBRData):
        """
        Creates a prior.
        """

        # If the prior is linear
        if getattr(self.conf, f"linear_{name.lower()}"):
            return self.create_linear_prior(name, data)
        # If the prior is random
        elif getattr(self.conf, f"random_{name.lower()}"):
            # If the prior is centered
            if getattr(self.conf, f"centered_{name.lower()}"):
                return self.create_centered_prior(name, data)
            # If the prior is non-centered
            else:
                return self.create_non_centered_prior(name, data)            
        # This parameter is fixed, i.e. not linear or random
        else:
            return self.create_fixed_prior(name, data)

    def make_pymc_dist(self, name: str, data: HBRData):
        distmap = {'Normal': pm.Normal,
                   'HalfNormal': pm.HalfNormal, 'Uniform': pm.Uniform}
        dist = distmap[getattr(self.conf, f"{name.lower()}_dist")]
        params = getattr(self.conf, f"{name.lower()}_dist_params")
        prior = dist(name, **params)

    def create_fixed_prior(self, name, data):
        pass

    def create_non_centered_prior(self, name, data):
        mu_dist = self.make_pymc_dist(f"mu_{name}", data)
        sigma_dist = self.make_pymc_dist(f"sigma_{name}", data)
        offset = pm.Normal(f"offset_{name}", mu=0, sigma=1)
        mu = pm.Deterministic(f"{name}", mu_dist + offset * sigma_dist)
        return mu

    def create_centered_prior(self, name, data):
        mu_dist = self.make_pymc_dist(f"mu_{name}", data)
        sigma_dist = self.make_pymc_dist(f"sigma_{name}", data)
        mu = pm.Normal(f"{name}", mu=mu_dist, sigma=sigma_dist)
        return mu

    def create_linear_prior(self, name: str, data: HBRData):
        """
        Creates a linear prior.
        """
        # Create a prior for the slope
        slope_prior = self.create_prior(f"slope_{name.lower()}", HBRData)
        # Create a prior for the intercept
        intercept_prior = self.create_prior(
            f"intercept_{name.lower()}", HBRData)

        # Create a linear prior
        return slope_prior * data.pm_X + intercept_prior

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
