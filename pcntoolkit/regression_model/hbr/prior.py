from __future__ import annotations

import copy
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Tuple, Union

import arviz as az
import numpy as np
import pymc as pm
import scipy.stats as stats
import xarray as xr
from pymc import math

from pcntoolkit.regression_model.hbr.hbr_data import HBRData
from pcntoolkit.util.output import Errors, Output

PM_DISTMAP = {
    "Normal": pm.Normal,
    "Cauchy": pm.Cauchy,
    "HalfNormal": pm.HalfNormal,
    "HalfCauchy": pm.HalfCauchy,
    "Uniform": pm.Uniform,
    "Gamma": pm.Gamma,
    "InvGamma": pm.InverseGamma,
    "LogNormal": pm.LogNormal,
}

DEFAULT_PRIOR_ARGS = {
    # For all models
    "linear_mu": True,
    "random_intercept_mu": True,
    "slope_mu_dist": "Normal",
    "slope_mu_params": (0, 10.0),
    "mu_intercept_mu_dist": "Normal",
    "mu_intercept_mu_params": (0, 10.0),
    "sigma_intercept_mu_dist": "HalfNormal",
    "sigma_intercept_mu_params": (2.0,),
    "linear_sigma": True,
    "mapping_sigma": "softplus",
    "mapping_params_sigma": (0, 2.0),
    "slope_sigma_dist": "Normal",
    "slope_sigma_params": (0, 10.0),
    "intercept_sigma_dist": "HalfNormal",
    "intercept_sigma_params": (2.0,),
    # For SHASH models
    "linear_epsilon": False,
    "random_epsilon": False,
    "epsilon_dist": "Normal",
    "epsilon_params": (0, 1.0),
    "linear_delta": False,
    "random_delta": False,
    "delta_dist": "Normal",
    "delta_params": (1.0, 2.0),
    "mapping_delta": "softplus",
    "mapping_params_delta": (0, 3.0, 0.3),
}


def make_prior(name: str = "theta", **kwargs) -> BasePrior:
    if kwargs.pop("linear", False):
        return LinearPrior(name, **kwargs)
    elif kwargs.pop("random", False):
        return RandomPrior(name, **kwargs)
    else:
        return Prior(name, **kwargs)


def prior_from_args(name: str, args: Dict[str, Any], dims: Optional[Union[Tuple[str, ...], str]] = None) -> BasePrior:
    my_args = DEFAULT_PRIOR_ARGS | args
    mapping = my_args.get(f"mapping_{name}", "identity")
    mapping_params = my_args.get(f"mapping_params_{name}", (0, 1))
    if name.split("_")[0] in ["sigma", "delta"]:
        dist_name = my_args.get(f"dist_name_{name}", "HalfNormal")
        dist_params = my_args.get(f"dist_params_{name}", (1.0,))
        if my_args.get(f"linear_{name}", False) or my_args.get(f"random_{name}", False):
            assert mapping != "identity", "Sigma and delta need a mapping if they are linear or random"
        else:
            assert my_args.get(f"dist_name_{name}", None) not in ["Normal", "Cauchy"] or (
                my_args.get(f"dist_name_{name}", None) == "Uniform" and my_args.get(f"dist_params_{name}", None)[0] > 0
            ), "Sigma and delta need a positive distribution if they are not linear or random"
    else:
        dist_name = my_args.get(f"dist_name_{name}", "Normal")
        dist_params = my_args.get(f"dist_params_{name}", (0, 1))

    dims = my_args.get(f"dims_{name}", dims)
    if my_args.get(f"linear_{name}", False):
        slope = prior_from_args(f"slope_{name}", my_args, dims=dims)
        intercept = prior_from_args(f"intercept_{name}", my_args, dims=dims)
        return LinearPrior(name, dims, mapping, mapping_params, slope, intercept)
    elif my_args.get(f"random_{name}", False):
        mu = prior_from_args(f"mu_{name}", my_args, dims=dims)
        sigma = prior_from_args(f"sigma_{name}", my_args, dims=dims)
        return RandomPrior(name, dims, mapping, mapping_params, mu, sigma)
    else:
        return Prior(name, dims, mapping, mapping_params, dist_name, dist_params)


class BasePrior(ABC):
    def __init__(
        self,
        name: str = "theta",
        dims: Optional[Union[Tuple[str, ...], str]] = None,
        mapping: str = "identity",
        mapping_params: tuple = (0, 1),
        **kwargs,
    ):
        self.name = name
        self._dims = dims
        has_covariate_dim = False if not self.dims else "covariates" in self.dims
        if self.name.startswith("slope") and not has_covariate_dim:
            if self.dims is None:
                self.dims = ("covariates",)
            else:
                self.dims = (dims, "covariates")
        self.mapping = mapping
        self.mapping_params = mapping_params
        self.sample_dims = ()

    @property
    def dims(self):
        return self._dims

    @dims.setter
    def dims(self, value):
        self._dims = value

    def sample(self, data) -> Any:
        samples = self._sample(data)
        return self.apply_mapping(samples)

    def apply_mapping(self, x: Any) -> Any:
        a, b = self.mapping_params[0], self.mapping_params[1]
        if self.mapping == "identity":
            toreturn = x
        elif self.mapping == "exp":
            toreturn = math.exp(a + x / b) * b
        elif self.mapping == "softplus":
            toreturn = math.log(1 + math.exp(a + x / b)) * b  # type: ignore
        else:
            raise Output.error(Errors.ERROR_UNKNOWN_MAPPING, mapping=self.mapping)
        if len(self.mapping_params) > 2:
            toreturn = toreturn + self.mapping_params[2]
        return toreturn

    @abstractmethod
    def compile(
        self,
        model: pm.Model,
        idata: Optional[az.InferenceData] = None,
        freedom: float = 1,
    ):
        pass

    @abstractmethod
    def _sample(self, data) -> Any:
        pass

    @property
    @abstractmethod
    def has_random_effect(self) -> bool:
        pass

    @abstractmethod
    def set_name(self, name: str) -> None:
        pass

    def to_dict(self):
        dct = copy.deepcopy(self.__dict__)
        dct["dims"] = self.dims
        del dct["sample_dims"]
        return dct | {"type": self.__class__.__name__}

    @classmethod
    def from_dict(cls, dict: dict) -> BasePrior:
        return globals()[dict.pop("type")].from_dict(dict)

    def __eq__(self, other: BasePrior):
        return self.to_dict() == other.to_dict()


class Prior(BasePrior):
    def __init__(
        self,
        name: str = "theta",
        dims: Optional[Union[Tuple[str, ...], str]] = None,
        mapping: str = "identity",
        mapping_params: tuple[float, ...] = (0.0, 1.0),
        dist_name: str = "Normal",
        dist_params: Tuple[float | int | list[float | int], ...] = (0, 10.0),
        **kwargs,
    ):
        super().__init__(name, dims, mapping, mapping_params, **kwargs)
        self.dist_name = dist_name
        self.dist_params = dist_params
        self.sample_dims = ()

    def compile(
        self,
        model: pm.Model,
        idata: Optional[az.InferenceData] = None,
        freedom: float = 1,
    ):
        with model:
            if idata is not None:
                self.approximate(
                    model,
                    self.dist_name,
                    az.extract(idata, var_names=self.name),
                    freedom,
                )
            self.dist = PM_DISTMAP[self.dist_name](self.name, *self.dist_params, dims=self.dims)

    def _sample(self, data: HBRData):
        return self.dist

    def approximate(
        self,
        model: Any,
        dist_name: str,
        samples: xr.DataArray,
        freedom: float | int = 1,
    ) -> None:
        # TODO At some point, we want to flatten over all dimensions except the covariate dimension."""
        def infer_params(s):
            with model:
                if dist_name == "Normal":
                    temp = stats.norm.fit(s)
                    return (temp[0], freedom * temp[1])
                elif dist_name == "HalfNormal":
                    temp = stats.halfnorm.fit(s)
                    return (freedom * temp[1],)
                elif dist_name == "LogNormal":
                    temp = stats.lognorm.fit(s)
                    return (temp[0], freedom * temp[1])
                elif dist_name == "Cauchy":
                    temp = stats.cauchy.fit(s)
                    return (temp[0], freedom * temp[1])
                elif dist_name == "HalfCauchy":
                    temp = stats.halfcauchy.fit(s)
                    return (freedom * temp[1],)
                elif dist_name == "Uniform":
                    temp = stats.uniform.fit(s)
                    return (temp[0], temp[1])
                elif dist_name == "Gamma":
                    temp = stats.gamma.fit(s)
                    return (temp[0], temp[1], freedom * temp[2])
                elif dist_name == "InvGamma":
                    temp = stats.invgamma.fit(s)
                    return (temp[0], temp[1], freedom * temp[2])
                else:
                    raise Output.error(Errors.ERROR_UNKNOWN_DISTRIBUTION, dist_name=dist_name)

        if "covariates" in samples.dims:
            params = [infer_params(samples.sel(covariates=i)) for i in samples.coords["covariates"]]
            self.dist_params = [i.tolist() for i in np.array(params).T]
        else:
            self.dist_params = infer_params(samples)

    @property
    def has_random_effect(self):
        return False

    def to_dict(self):
        dct = super().to_dict()
        dct.pop("dist", None)
        dct["type"] = self.__class__.__name__
        return dct

    @classmethod
    def from_dict(cls, dct: dict):
        return cls(
            **{
                k: v
                for k, v in dct.items()
                if k
                in [
                    "name",
                    "dims",
                    "mapping",
                    "mapping_params",
                    "dist_name",
                    "dist_params",
                ]
            }
        )

    def set_name(self, name: str) -> None:
        self.name = name


class RandomPrior(BasePrior):
    def __init__(
        self,
        name: str = "theta",
        dims: Optional[Union[Tuple[str, ...], str]] = None,
        mapping: str = "identity",
        mapping_params: tuple[float, ...] = (0.0, 1.0),
        mu: Optional[BasePrior] = None,
        sigma: Optional[BasePrior] = None,
        **kwargs,
    ):
        super().__init__(name, dims, mapping, mapping_params, **kwargs)
        self.mu = mu or get_default_sub_mu(dims)
        self.mu.set_name(f"mu_{self.name}")
        self.sigma = sigma or get_default_sub_sigma(dims)
        self.sigma.set_name(f"sigma_{self.name}")
        self.sample_dims = ("datapoints",)

    @property
    def dims(self):
        return self._dims

    @dims.setter
    def dims(self, value):
        if hasattr(self, "mu"):
            self.mu.dims = value
        if hasattr(self, "sigma"):
            self.sigma.dims = value
        self._dims = value

    def compile(
        self,
        model: pm.Model,
        idata: Optional[az.InferenceData] = None,
        freedom: float = 1,
    ):
        outdims = "datapoints" if not self.dims else ("datapoints", *self.dims)
        with model:
            self.mu.compile(model, idata, freedom)
            if self.dims:
                acc = self.mu.dist[None]  # type: ignore
            else:
                acc = self.mu.dist  # type: ignore
            self.sigmas: dict[str, BasePrior] = {}
            self.offsets = {}
            self.scaled_offsets = {}
            for be in model.custom_batch_effect_dims:  # type:ignore
                be_dims = be if not self.dims else (be, *self.dims)
                self.sigmas[be] = copy.deepcopy(self.sigma)
                self.sigmas[be].set_name(f"{be}_sigma_{self.name}")
                self.sigmas[be].compile(model, idata, freedom)
                self.scaled_offsets[be] = pm.Deterministic(
                    f"{be}_offset_{self.name}",
                    self.sigmas[be].dist  # type: ignore
                    * pm.Normal(
                        f"normalized_{be}_offset_{self.name}",
                        dims=be_dims,  # type:ignore
                    ),
                    dims=be_dims,
                )
                acc += self.scaled_offsets[be][model[f"{be}_data"]]
            self.dist = pm.Deterministic(self.name, acc, dims=outdims)

    def set_name(self, name: str):
        self.name = name
        self.mu.set_name(f"mu_{self.name}")
        self.sigma.set_name(f"sigma_{self.name}")

    def _sample(self, data: HBRData):
        return self.dist

    @property
    def has_random_effect(self):
        return True

    def to_dict(self):
        dct = super().to_dict()
        dct["mu"] = self.mu.to_dict()
        dct["sigma"] = self.sigma.to_dict()
        if hasattr(self, "sigmas"):
            for k, v in self.sigmas.items():
                dct[f"{k}_sigma"] = v.to_dict()

        for thing in ["sigmas", "offsets", "scaled_offsets", "dist"]:
            if hasattr(self, thing):
                del dct[thing]
        return dct

    @classmethod
    def from_dict(cls, dct):
        mu = BasePrior.from_dict(dct["mu"])
        sigma = BasePrior.from_dict(dct["sigma"])
        instance = cls(
            mu=mu,
            sigma=sigma,
            **{k: v for k, v in dct.items() if k in ["name", "dims", "mapping", "mapping_params"]},
        )
        instance.sigmas = {k: BasePrior.from_dict(v) for k, v in dct.items() if k.endswith("_sigma")}
        # instance.scaled_offsets = {k: Param.from_dict(v) for k, v in dct.items() if k.endswith("_offset")}
        return instance


class LinearPrior(BasePrior):
    def __init__(
        self,
        name: str = "theta",
        dims: Optional[Union[Tuple[str, ...], str]] = None,
        mapping: str = "identity",
        mapping_params: tuple[float, ...] = (0.0, 1.0),
        slope: Optional[BasePrior] = None,
        intercept: Optional[BasePrior] = None,
        **kwargs,
    ):
        super().__init__(name, dims, mapping, mapping_params, **kwargs)
        self.slope = slope or get_default_slope()
        self.slope.dims = ("covariates",) if not self.dims else ("covariates", *self.dims)
        self.intercept = intercept or get_default_intercept()
        self.intercept.dims = self.dims
        self.sample_dims = ("datapoints",)
        self.set_name(self.name)

    @property
    def dims(self):
        return self._dims

    @dims.setter
    def dims(self, value):
        if hasattr(self, "slope"):
            self.slope.dims = value
        if hasattr(self, "intercept"):
            self.intercept.dims = value
        self._dims = value

    def compile(
        self,
        model: pm.Model,
        idata: Optional[az.InferenceData] = None,
        freedom: float = 1,
    ):
        self.one_dimensional = len(model.coords["covariates"]) == 1  # type: ignore
        self.slope.compile(model, idata, freedom)
        self.intercept.compile(model, idata, freedom)

    def _sample(self, data: HBRData):
        if self.one_dimensional:
            return (self.slope.sample(data) * data.pm_X)[:, 0] + self.intercept.sample(data)
        else:
            return math.sum(self.slope.sample(data) * data.pm_X, axis=1, keepdims=False) + self.intercept.sample(data)

    def to_dict(self):
        dct = super().to_dict()
        dct["slope"] = self.slope.to_dict()
        dct["intercept"] = self.intercept.to_dict()
        return dct

    @classmethod
    def from_dict(cls, dct):
        slope = BasePrior.from_dict(dct["slope"])
        intercept = BasePrior.from_dict(dct["intercept"])
        return cls(
            slope=slope,
            intercept=intercept,
            **{k: v for k, v in dct.items() if k in ["name", "dims", "mapping", "mapping_params"]},
        )

    @property
    def has_random_effect(self):
        return self.slope.has_random_effect or self.intercept.has_random_effect

    def set_name(self, name):
        self.name = name
        self.slope.set_name(f"slope_{self.name}")
        self.intercept.set_name(f"intercept_{self.name}")


"≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠"
"≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠"
"≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠"
"≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠"


def get_default_mu(dims: Optional[Union[Tuple[str, ...], str]] = None) -> BasePrior:
    return LinearPrior(
        "mu",
        dims=dims,
        slope=get_default_slope(dims),
        intercept=get_default_intercept(dims),
    )


def get_default_sigma(dims: Optional[Union[Tuple[str, ...], str]] = None) -> BasePrior:
    slope = get_default_slope(dims)
    intercept = get_default_intercept(dims)
    return LinearPrior(
        slope=slope,
        intercept=intercept,
        mapping="softplus",
        mapping_params=(0.0, 3.0),
    )


def get_default_epsilon() -> BasePrior:
    return Prior(
        dist_name="Normal",
        dist_params=(
            0.0,
            1.0,
        ),
    )


def get_default_delta() -> BasePrior:
    return Prior(
        dist_name="Normal",
        dist_params=(
            0.0,
            2.0,
        ),
        mapping="softplus",
        mapping_params=(0.0, 3.0, 0.3),
    )


def get_default_sub_mu(dims: Optional[Union[Tuple[str, ...], str]] = None) -> BasePrior:
    return Prior(
        dims=dims,
        dist_name="Normal",
        dist_params=(
            0.0,
            10.0,
        ),
    )


def get_default_sub_sigma(dims: Optional[Union[Tuple[str, ...], str]] = None) -> BasePrior:
    return Prior(
        dims=dims,
        dist_name="LogNormal",
        dist_params=(2.0,),
    )


def get_default_slope(
    dims: Optional[Union[Tuple[str, ...], str]] = ("covariates",),
) -> BasePrior:
    return Prior(
        dims=dims,
        dist_name="Normal",
        dist_params=(0, 10.0),
    )


def get_default_intercept(dims: Optional[Union[Tuple[str, ...], str]] = None) -> BasePrior:
    return Prior(
        dims=dims,
        dist_name="Normal",
        dist_params=(0, 10.0),
    )
