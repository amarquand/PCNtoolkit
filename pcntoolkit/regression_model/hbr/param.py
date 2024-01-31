from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Tuple

import arviz as az
import numpy as np
import pymc as pm
import scipy.stats as stats
from pytensor.tensor.extra_ops import repeat

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
    mapping: str = "identity"
    mapping_params: tuple = (0, 1)

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
        self.distmap = {
            "Normal": pm.Normal,
            "Cauchy": pm.Cauchy,
            "HalfNormal": pm.HalfNormal,
            "HalfCauchy": pm.HalfCauchy,
            "Uniform": pm.Uniform,
        }

        if self.linear:
            self.set_linear_params()

        elif self.random:
            if self.centered:
                self.set_centered_random_params()
            else:
                self.set_noncentered_random_params()

        else:
            # If the parameter is really only a single number, we need to add an empty dimension so our outputs are always 2D
            if (self.dims == ()) or (self.dims == []):
                # self.dims = None
                self.shape = (1,)
            else:
                self.shape = None
                if type(self.dims) is str:
                    self.dims = (self.dims,)

    def add_to(self, model, idata=None):
        self.distmap = {
            "Normal": pm.Normal,
            "Cauchy": pm.Cauchy,
            "HalfNormal": pm.HalfNormal,
        }

        with model:
            if self.linear:
                self.slope.add_to(model, idata)
                self.intercept.add_to(model, idata)
            elif self.random:
                if self.centered:
                    self.mu.add_to(model, idata)
                    self.sigma.add_to(model, idata)
                    self.dist = pm.Normal(
                        self.name,
                        mu=self.mu.dist,
                        sigma=self.sigma.dist,
                        dims=(*model.custom_batch_effect_dims, *self.dims),
                    )
                else:
                    self.mu.add_to(model, idata)
                    self.sigma.add_to(model, idata)
                    self.offset = pm.Normal(
                        f"offset_" + self.name,
                        mu=0,
                        sigma=1,
                        dims=(*model.custom_batch_effect_dims, *self.dims),
                    )
                    self.dist = pm.Deterministic(
                        self.name,
                        self.mu.dist + self.offset * self.sigma.dist,
                        dims=(*model.custom_batch_effect_dims, *self.dims),
                    )
            else:
                if idata is None:
                    self.dist = self.distmap[self.dist_name](
                        self.name, *self.dist_params, shape=self.shape, dims=self.dims
                    )
                else:
                    self.dist = self.approximate_marginal(
                        model, self.dist_name, az.extract(idata, var_names=self.name)
                    )

    def approximate_marginal(self, model, dist_name: str, samples, freedom=1):
        """
        use scipy stats.XXX.fit to get the parameters of the marginal distribution
        """
        """At some point, we want to average over all dimensions except the covariate dimension."""
        with model:
            if dist_name == "Normal":
                temp = stats.norm.fit(samples)
                return pm.Normal(
                    self.name,
                    mu=temp[0],
                    sigma=freedom * temp[1],
                    shape=self.shape,
                    dims=self.dims,
                )
            elif dist_name == "HalfNormal":
                temp = stats.halfnorm.fit(samples)
                return pm.HalfNormal(
                    self.name, sigma=freedom * temp[1], shape=self.shape, dims=self.dims
                )
            elif dist_name == "LogNormal":
                temp = stats.lognorm.fit(samples)
                return pm.Lognormal(
                    self.name,
                    mu=temp[1],
                    sigma=freedom * temp[2],
                    shape=self.shape,
                    dims=self.dims,
                )
            elif dist_name == "Cauchy":
                temp = stats.cauchy.fit(samples)
                return pm.Cauchy(
                    self.name,
                    alpha=temp[0],
                    beta=freedom * temp[1],
                    shape=self.shape,
                    dims=self.dims,
                )
            elif dist_name == "HalfCauchy":
                temp = stats.halfcauchy.fit(samples)
                return pm.HalfCauchy(
                    self.name, beta=freedom * temp[1], shape=self.shape, dims=self.dims
                )
            elif dist_name == "Uniform":
                temp = stats.uniform.fit(samples)
                return pm.Uniform(
                    self.name,
                    lower=temp[0],
                    upper=temp[1],
                    shape=self.shape,
                    dims=self.dims,
                )
            else:
                raise ValueError(f"Unknown distribution name {dist_name}")

    @classmethod
    def from_args(cls, name: str, args: Dict[str, any], dims=()):
        if args.get(f"linear_{name}", False):
            slope = cls.from_dict(f"slope_{name}", args, dims=(*dims, "covariates"))
            intercept = cls.from_args(f"intercept_{name}", args, dims=dims)
            return cls(
                name,
                dims=dims,
                linear=True,
                slope=slope,
                intercept=intercept,
                mapping=args.get(f"mapping_{name}", "identity"),
            )
        elif args.get(f"random_{name}", False):
            if args.get(f"centered_{name}", False):
                mu = cls.from_args(f"mu_{name}", args, dims=dims)
                sigma = cls.from_args(f"sigma_{name}", args, dims=dims)
                return cls(
                    name, dims=dims, random=True, centered=True, mu=mu, sigma=sigma
                )
            else:
                mu = cls.from_args(f"mu_{name}", args, dims=dims)
                sigma = cls.from_args(f"sigma_{name}", args, dims=dims)
                return cls(
                    name, dims=dims, random=True, centered=False, mu=mu, sigma=sigma
                )
        else:
            (default_dist, default_params) = (
                ("HalfNormal", (1.0,))
                if name.startswith("sigma")
                else ("Normal", (0.0, 1.0))
            )
            return cls(
                name,
                dims=dims,
                dist_name=args.get(f"{name}_dist_name", default_dist),
                dist_params=args.get(f"{name}_dist_params", default_params),
            )

    @classmethod
    def from_dict(cls, dict):
        if dict.get("linear", False):
            slope = cls.from_dict(dict["slope"])
            intercept = cls.from_dict(dict["intercept"])
            return cls(
                dict["name"],
                dims=dict["dims"],
                linear=True,
                slope=slope,
                intercept=intercept,
                mapping=dict.get("mapping", "identity"),
            )
        elif dict.get("random", False):
            if dict.get("centered", False):
                mu = cls.from_dict(dict["mu"])
                sigma = cls.from_dict(dict["sigma"])
                return cls(
                    dict["name"],
                    dims=dict["dims"],
                    random=True,
                    centered=True,
                    mu=mu,
                    sigma=sigma,
                )
            else:
                mu = cls.from_dict(dict["mu"])
                sigma = cls.from_dict(dict["sigma"])
                return cls(
                    dict["name"],
                    dims=dict["dims"],
                    random=True,
                    centered=False,
                    mu=mu,
                    sigma=sigma,
                )
        else:
            return cls(
                dict["name"],
                dims=dict["dims"],
                dist_name=dict.get("dist_name", "Normal"),
                dist_params=dict.get("dist_params", (0.0, 1.0)),
                mapping=dict.get("mapping", "identity"),
                mapping_params=dict.get("mapping_params", (0.0, 1.0)),
            )

    def set_noncentered_random_params(self):
        if not self.mu:
            self.mu = Param(f"mu_{self.name}", dims=self.dims)
        if not self.sigma:
            self.sigma = Param(
                f"sigma_{self.name}",
                dims=self.dims,
                dist_name="HalfNormal",
                dist_params=(1,),
            )

    def set_centered_random_params(self):
        if not self.mu:
            self.mu = Param(f"mu_{self.name}", dims=self.dims)
        if not self.sigma:
            self.sigma = Param(
                f"sigma_{self.name}",
                dims=self.dims,
                dist_name="HalfNormal",
                dist_params=(1.0,),
            )

    def set_linear_params(self):
        if not self.slope:
            self.slope = Param(f"slope_{self.name}", dims=(*self.dims, "covariates"))
        if not self.intercept:
            self.intercept = Param(f"intercept_{self.name}", dims=self.dims)

    def get_samples(self, data: HBRData):
        if self.linear:
            slope_samples = self.slope.get_samples(data)
            intercept_samples = self.intercept.get_samples(data)
            result = (
                pm.math.sum(slope_samples * data.pm_X, axis=1, keepdims=True)
                + intercept_samples
            )
            return self.apply_mapping(result)

        elif self.random:
            if self.has_covariate_dim:
                return self.dist[*data.pm_batch_effect_indices]
            else:
                return self.dist[*data.pm_batch_effect_indices, None]
        else:
            return repeat(self.dist[None, :], data.pm_X.shape[0], axis=0)

    def apply_mapping(self, x):
        if self.mapping == "identity":
            return x
        elif self.mapping == "exp":
            return pm.math.exp(x)
        elif self.mapping == "softplus":
            return pm.math.log1pexp(x)
        else:
            raise ValueError(f"Unknown mapping {self.mapping}")

    def to_dict(self):
        param_dict = {
            "name": self.name,
            "dims": self.dims,
            "dist_name": self.dist_name,
            "dist_params": self.dist_params,
            "linear": self.linear,
            "random": self.random,
            "centered": self.centered,
            "has_covariate_dim": self.has_covariate_dim,
            "has_random_effect": self.has_random_effect,
            "mapping": self.mapping,
            "mapping_params": self.mapping_params,
        }
        if self.linear:
            param_dict["slope"] = self.slope.to_dict()
            param_dict["intercept"] = self.intercept.to_dict()
        elif self.random:
            param_dict["mu"] = self.mu.to_dict()
            param_dict["sigma"] = self.sigma.to_dict()
        return param_dict
