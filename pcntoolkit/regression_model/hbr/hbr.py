from __future__ import annotations
from typing import Tuple

import pymc as pm
import numpy as np
from pcntoolkit.regression_model.hbr.hbr_data import HBRData
from pcntoolkit.regression_model.hbr.param import Param
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
        self.is_from_args = False
        self.args = None
        self.model = None


    @classmethod
    def from_args(cls, args):
        """
        Creates a configuration from command line arguments.
        """
        conf = HBRConf.from_args(args)
        self = cls(conf)
        self.is_from_args = True
        return self

    @property
    def conf(self) -> HBRConf:
        return self._conf

    def fit(self, data: HBRData):
        """
        Fits the model.
        """
        if not self.is_fitted:

            # Sample from pymc model
            if not self.model:
                self.create_new_pymc_model(data)

            with self.model:
                self.idata = pm.sample(
                    self.conf.n_samples, tune=self.conf.n_tune, cores=self.conf.n_cores)
            self.is_fitted = True

        else:
            raise RuntimeError("Model is already fitted.")
        
    def predict(self, data: HBRData) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predicts the response variable for the given data.
        """
        assert self.is_fitted, "Model must be fitted before predicting."

        data.set_data_in_existing_model(self.model)
        graph =self.model.to_graphviz()
        graph.render('hbr_predict_model_graph', format='png')
        with self.model:
            self.idata_pred = pm.sample_posterior_predictive(self.idata, var_names=['y_pred'])

        return self.idata_pred['y_pred'].mean(axis=0), self.idata_pred['y_pred'].std(axis=0)

        
        

    def create_new_pymc_model(self, data: HBRData) -> HBRData:
        """
        Creates the pymc model.
        """
        self.model = pm.Model(coords=data.coords, coords_mutable=data.coords_mutable)
        data.add_to_model(self.model)
        if self.conf.likelihood == "Normal":
            self.create_normal_pymc_model(data)
        else:
            raise NotImplementedError(
                f"Likelihood {self.conf.likelihood} not implemented for {self.__class__.__name__}")

    def create_normal_pymc_model(self, data: HBRData) -> HBRData:
        """
        Creates the pymc model.
        """
        self.conf.mu.add_to(self.model)
        self.conf.sigma.add_to(self.model)
        with self.model:
            mu_samples = self.conf.mu.get_samples(data)
            # mu_samples = pm.Deterministic('mu_samples', mu_samples, dims=('datapoints', 'response_vars'))
            sigma_samples = self.conf.sigma.get_samples(data)
            # sigma_samples = pm.Deterministic('sigma_samples', sigma_samples, dims=('datapoints', 'response_vars'))
            pm.Normal("y_pred", mu=mu_samples, sigma=sigma_samples, dims=('datapoints', 'response_vars'), observed=data.pm_y)


