from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, Tuple
import pymc as pm
import numpy as np
import scipy.stats as stats
import arviz as az
from pcntoolkit.regression_model.hbr.hbr_data import HBRData


@dataclass
class Param:
    name: str
    dims: Tuple[str] = ()

    dist_name: str = "Normal"
    dist_params: tuple = (0, 1)

    linear: bool = False
    slope: Param = None
    intercept: Param = None
    mapping: str = 'identity'

    random: bool = False
    centered: bool = False
    mu: Param = None
    sigma: Param = None

    has_covariate_dim: bool = field(init=False, default=False)
    has_random_effect: bool = field(init=False, default=False)
    distmap: Dict[str, pm.Distribution] = field(init=False, default=False)
    dist: pm.Distribution = field(init=False, default=False)

    def __post_init__(self):
        self.has_covariate_dim = False if not self.dims else "covariates" in self.dims
        self.distmap = {"Normal": pm.Normal,
                        "Cauchy": pm.Cauchy, 
                        "HalfNormal": pm.HalfNormal,
                        "HalfCauchy": pm.HalfCauchy,
                        "Uniform": pm.Uniform}

        if self.linear:
            self.set_linear_params()

        elif self.random:
            if self.centered:
                self.set_centered_random_params()
            else:
                self.set_noncentered_random_params()

        else:
            # If the parameter is really only a single number, we need to add an empty dimension so our outputs are always 2D
            if self.dims == ():
                # self.dims = None
                self.shape = (1,)
            else:
                self.shape = None
                if type(self.dims) is str:
                    self.dims = (self.dims,)

    def add_to(self, model, idata=None):
        self.distmap = {"Normal": pm.Normal,
                        'Cauchy': pm.Cauchy, 
                        "HalfNormal": pm.HalfNormal}

        with model:
            if self.linear:
                self.slope.add_to(model,idata)
                self.intercept.add_to(model, idata)
            elif self.random:
                if self.centered:
                    self.mu.add_to(model, idata)
                    self.sigma.add_to(model, idata)
                    self.dist = pm.Normal(self.name, mu=self.mu.dist, sigma=self.sigma.dist, dims=(
                        *model.custom_batch_effect_dims, *self.dims))
                else:
                    self.mu.add_to(model, idata)
                    self.sigma.add_to(model, idata)
                    self.offset = pm.Normal(f"offset_" + self.name, mu=0, sigma=1, dims=(
                        *model.custom_batch_effect_dims, *self.dims))
                    self.dist = pm.Deterministic(self.name, self.mu.dist + self.offset*
                                                 self.sigma.dist, dims=(*model.custom_batch_effect_dims, *self.dims))
            else:
                if idata is None:
                    self.dist = self.distmap[self.dist_name](self.name, *self.dist_params, shape=self.shape, dims=self.dims)                
                else:
                    self.dist = self.approximate_marginal(model, az.extract(idata, var_names = self.name))

  
    def approximate_marginal(self, model, dist_name:str, samples, freedom=1):
        """
        use scipy stats.XXX.fit to get the parameters of the marginal distribution
        """
        """At some point, we want to average over all dimensions except the covariate dimension."""
        with model:
            if dist_name == "Normal":
                temp = stats.norm.fit(samples)
                return pm.Normal(self.name, mu=temp[0], sigma=freedom*temp[1], shape=self.shape, dims=self.dims)
            elif dist_name == "HalfNormal":
                temp = stats.halfnorm.fit(samples)
                return pm.HalfNormal(self.name, sigma=freedom*temp[1], shape=self.shape, dims=self.dims)
            elif dist_name == "LogNormal":
                temp = stats.lognorm.fit(samples)
                return pm.Lognormal(self.name, mu=temp[0], sigma=freedom*temp[1], shape=self.shape, dims=self.dims)
            elif dist_name == "Cauchy":
                temp = stats.cauchy.fit(samples)
                return pm.Cauchy(self.name, loc=temp[0], scale=freedom*temp[1], shape=self.shape, dims=self.dims)
            elif dist_name == "HalfCauchy":
                temp = stats.halfcauchy.fit(samples)
                return pm.HalfCauchy(self.name, sigma=freedom*temp[1], shape=self.shape, dims=self.dims)
            elif dist_name == "Uniform":
                temp = stats.uniform.fit(samples)
                return pm.Uniform(self.name, lower=temp[0], upper=temp[1], shape=self.shape, dims=self.dims)
            else:
                raise ValueError(f"Unknown distribution name {dist_name}")
            

    @classmethod
    def from_dict(cls, name: str, param_dict: Dict[str, any], dims=()):
        if param_dict.get(f'linear_{name}', False):
            slope = cls.from_dict(
                f"slope_{name}", param_dict, dims=(*dims, 'covariates'))
            intercept = cls.from_dict(
                f"intercept_{name}",  param_dict, dims=dims)
            return cls(name, dims=dims, linear=True, slope=slope, intercept=intercept)
        elif param_dict.get(f'random_{name}', False):
            if param_dict.get(f'centered_{name}', False):
                mu = cls.from_dict(f"mu_{name}", param_dict, dims=dims)
                sigma = cls.from_dict(f"sigma_{name}", param_dict, dims=dims)
                return cls(name,  dims=dims, random=True, centered=True, mu=mu, sigma=sigma)
            else:
                mu = cls.from_dict(f"mu_{name}",  param_dict, dims=dims)
                sigma = cls.from_dict(f"sigma_{name}", param_dict, dims=dims)
                return cls(name, dims=dims, random=True, centered=False, mu=mu, sigma=sigma)
        else:
            (default_dist, default_params) = ("HalfNormal", (1.,)
                                              ) if name.startswith("sigma") else ("Normal", (0., 1.))
            return cls(name, dims=dims, dist_name=param_dict.get(f'{name}_dist_name', default_dist), dist_params=param_dict.get(f'{name}_dist_params', default_params))

    def set_noncentered_random_params(self):
        if not self.mu:
            self.mu = Param(f"mu_{self.name}", dims=self.dims)
        if not self.sigma:
            self.sigma = Param(f"sigma_{self.name}",dims=self.dims, dist_name="HalfNormal", dist_params=(1,))

    def set_centered_random_params(self):
        if not self.mu:
            self.mu = Param(f"mu_{self.name}", dims=self.dims)
        if not self.sigma:
            self.sigma = Param(
                f"sigma_{self.name}", dims=self.dims, dist_name="HalfNormal", dist_params=(1.,))

    def set_linear_params(self):
        if not self.slope:
            self.slope = Param(f"slope_{self.name}", dims=(
                *self.dims, "covariates"))
        if not self.intercept:
            self.intercept = Param(
                f"intercept_{self.name}",  dims=self.dims)

    def get_samples(self, data: HBRData):
        if self.linear:
            slope_samples = self.slope.get_samples(data)
            intercept_samples = self.intercept.get_samples(data)
            return pm.math.sum(slope_samples *data.pm_X, axis=1, keepdims=True) + intercept_samples
        elif self.random:
            if self.has_covariate_dim:
                return self.dist[*data.pm_batch_effect_indices]
            else:
                return self.dist[*data.pm_batch_effect_indices, None]
        else:
            return self.dist

    def to_dict(self):
        param_dict = {'name': self.name,
                    'dims': self.dims,
                    'dist_name': self.dist_name, 
                    'dist_params': self.dist_params, 
                    'linear': self.linear, 
                    'random': self.random, 
                    'centered': self.centered,
                    "has_covariate_dim": self.has_covariate_dim,
                    "has_random_effect": self.has_random_effect,}
        if self.linear:
            param_dict['slope'] = self.slope.to_dict()
            param_dict['intercept'] = self.intercept.to_dict()
        elif self.random:
            param_dict['mu'] = self.mu.to_dict()
            param_dict['sigma'] = self.sigma.to_dict()
        return param_dict
