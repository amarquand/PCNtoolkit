from __future__ import annotations

from typing import Tuple

import arviz as az
import numpy as np
import pymc as pm

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
        self.idata: az.InferenceData = None
        self.is_from_dict = False
        self.model = None

    @property
    def conf(self) -> HBRConf:
        return self._conf

    def create_pymc_model(self, data: HBRData, idata: az.InferenceData = None) -> HBRData:
        """
        Creates the pymc model.
        """
        self.model = pm.Model(coords=data.coords,
                              coords_mutable=data.coords_mutable)
        data.add_to_model(self.model)
        if self.conf.likelihood == "Normal":
            self.create_normal_pymc_model(data, idata)
        else:
            raise NotImplementedError(
                f"Likelihood {self.conf.likelihood} not implemented for {self.__class__.__name__}")

    def create_normal_pymc_model(self, data: HBRData, idata: az.InferenceData = None) -> HBRData:
        """
        Creates the pymc model.
        """
        self.conf.mu.add_to(self.model, idata)
        self.conf.sigma.add_to(self.model, idata)
        with self.model:
            mu_samples = self.conf.mu.get_samples(data)
            # mu_samples = pm.Deterministic('mu_samples', mu_samples, dims=('datapoints', 'response_vars'))
            sigma_samples = self.conf.sigma.get_samples(data)
            # sigma_samples = pm.Deterministic('sigma_samples', sigma_samples, dims=('datapoints', 'response_vars'))
            y_pred = pm.Normal("y_pred", mu=mu_samples, sigma=sigma_samples, dims=(
                'datapoints', 'response_vars'), observed=data.pm_y)

    def to_dict(self):
        """
        Converts the configuration to a dictionary.
        """
        my_dict = {}
        my_dict['conf'] = self.conf.to_dict()
        my_dict['is_fitted'] = self.is_fitted
        my_dict['is_from_dict'] = self.is_from_dict
        return my_dict

    @classmethod
    def from_dict(cls, args):
        """
        Creates a configuration from command line arguments or a dict
        """
        conf = HBRConf.from_dict(args)
        self = cls(conf)
        self.is_from_dict = True
        self.is_fitted = args.get('is_fitted', False)
        return self
