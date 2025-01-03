from __future__ import annotations

import copy
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Tuple, Union

import arviz as az
import pymc as pm
import scipy.stats as stats
import xarray as xr
from pymc import math

from pcntoolkit.regression_model.hbr.hbr_data import HBRData

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


def make_param(name: str = "theta", **kwargs) -> Param:
    if kwargs.pop("linear", False):
        return LinearParam(name, **kwargs)
    elif kwargs.pop("random", False):
        return RandomParam(name, **kwargs)
    else:
        return FixedParam(name, **kwargs)


def param_from_args(name: str, args: Dict[str, Any]) -> Param:
    dims = args.get(f"dims_{name}", None)
    mapping = args.get(f"mapping_{name}", "identity")
    mapping_params = args.get(f"mapping_params_{name}", (0, 1))
    if args.get(f"linear_{name}", False):
        slope = param_from_args(f"slope_{name}", args)
        intercept = param_from_args(f"intercept_{name}", args)
        return LinearParam(name, dims, mapping, mapping_params, slope, intercept)
    elif args.get(f"random_{name}", False):
        mu = param_from_args(f"mu_{name}", args)
        sigma = param_from_args(f"sigma_{name}", args)
        return RandomParam(name, dims, mapping, mapping_params, mu, sigma)
    else:
        return FixedParam(name, dims, mapping, mapping_params)


class Param(ABC):
    def __init__(
        self,
        name: str = "theta",
        dims: Optional[Union[Tuple[str, ...], str]] = None,
        mapping: str = "identity",
        mapping_params: tuple = (0, 1),
    ):
        self.name = name
        self.dims = dims
        has_covariate_dim = False if not self.dims else "covariates" in self.dims
        if self.name.startswith("slope") and not has_covariate_dim:
            if self.dims is None:
                self.dims = ("covariates",)
            else:
                self.dims = (dims, "covariates")
        self.mapping = mapping
        self.mapping_params = mapping_params
        self.sample_dims = ()

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
            raise ValueError(f"Unknown mapping {self.mapping}")
        if len(self.mapping_params) > 2:
            toreturn = toreturn + self.mapping_params[2]
        return toreturn

    @abstractmethod
    def create_graph(
        self, model: pm.Model, idata: Optional[az.InferenceData], freedom: float = 1
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
        del dct["sample_dims"]
        return dct | {"type": self.__class__.__name__}

    @classmethod
    def from_dict(cls, dict: dict) -> Param:
        return globals()[dict.pop("type")].from_dict(dict)

    def __eq__(self, other: Param):
        return self.to_dict() == other.to_dict()


class FixedParam(Param):
    def __init__(
        self,
        name: str = "theta",
        dims: Optional[Union[Tuple[str, ...], str]] = None,
        mapping: str = "identity",
        mapping_params: tuple[float, ...] = (0.0, 1.0),
        dist_name: str = "Normal",
        dist_params: Tuple[float | int, ...] = (0, 10.0),
    ):
        super().__init__(name, dims, mapping, mapping_params)
        self.dist_name = dist_name
        self.dist_params = dist_params
        self.sample_dims = ()

    def create_graph(
        self, model: pm.Model, idata: Optional[az.InferenceData], freedom: float = 1
    ):
        with model:
            if idata is not None:
                self.approximate_marginal(
                    model, self.dist_name, az.extract(idata, var_names=self.name), freedom
                )
            self.dist = PM_DISTMAP[self.dist_name](
                self.name, *self.dist_params, dims=self.dims
            )

    def _sample(self, data: HBRData):
        return self.dist

    def approximate_marginal(
        self,
        model: Any,
        dist_name: str,
        samples: xr.DataArray,
        freedom: float | int = 1,
    ) -> None:
        # TODO At some point, we want to flatten over all dimensions except the covariate dimension."""
        samples_flat = samples.to_numpy().flatten()
        with model:
            if dist_name == "Normal":
                temp = stats.norm.fit(samples_flat)
                self.dist_params = (temp[0], freedom * temp[1])
            elif dist_name == "HalfNormal":
                temp = stats.halfnorm.fit(samples_flat)
                self.dist_params = (freedom * temp[1],)
            elif dist_name == "LogNormal":
                temp = stats.lognorm.fit(samples_flat)
                self.dist_params = (temp[0], freedom * temp[1])
            elif dist_name == "Cauchy":
                temp = stats.cauchy.fit(samples_flat)
                self.dist_params = (temp[0], freedom * temp[1])
            elif dist_name == "HalfCauchy":
                temp = stats.halfcauchy.fit(samples_flat)
                self.dist_params = (freedom * temp[1],)
            elif dist_name == "Uniform":
                temp = stats.uniform.fit(samples_flat)
                self.dist_params = (temp[0], temp[1])
            elif dist_name == "Gamma":
                temp = stats.gamma.fit(samples_flat)
                self.dist_params = (temp[0], temp[1], freedom * temp[2])
            elif dist_name == "InvGamma":
                temp = stats.invgamma.fit(samples_flat)
                self.dist_params = (temp[0], temp[1], freedom * temp[2])
            else:
                raise ValueError(f"Unknown distribution name {dist_name}")

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


class RandomParam(Param):
    def __init__(
        self,
        name: str = "theta",
        dims: Optional[Union[Tuple[str, ...], str]] = None,
        mapping: str = "identity",
        mapping_params: tuple[float, ...] = (0.0, 1.0),
        mu: Optional[Param] = None,
        sigma: Optional[Param] = None,
    ):
        super().__init__(name, dims, mapping, mapping_params)
        self.mu = mu or get_default_sub_mu()
        self.mu.set_name(f"mu_{self.name}")
        self.sigma = sigma or get_default_sub_sigma()
        self.sigma.set_name(f"sigma_{self.name}")
        self.sample_dims = ("datapoints",)

    def create_graph(
        self, model: pm.Model, idata: Optional[az.InferenceData], freedom: float = 1
    ):
        self.model_reference = model
        with model:
            self.mu.create_graph(model, idata, freedom)
            self.sigmas: dict[str, Param] = {}
            self.offsets = {}
            self.scaled_offsets = {}
            for be in model.custom_batch_effect_dims:  # type:ignore
                self.sigmas[be] = copy.deepcopy(self.sigma)
                self.sigmas[be].set_name(f"{be}_sigma_{self.name}")
                self.sigmas[be].create_graph(model, idata, freedom)
                self.offsets[be] = pm.Normal(
                    f"{be}_offset_{self.name}",
                    dims=be if (not self.dims) else (*self.dims, be),  # type:ignore
                )

    def set_name(self, name: str):
        self.name = name
        self.mu.set_name(f"mu_{self.name}")
        self.sigma.set_name(f"sigma_{self.name}")

    def _sample(self, data: HBRData):
        with self.model_reference:
            acc = self.mu.sample(data)
            for k in self.sigmas.keys():
                if not hasattr(self.model_reference, f"scaled_{k}_offset_{self.name}"):
                    self.scaled_offsets[k] = pm.Deterministic(f"scaled_{k}_offset_{self.name}", self.sigmas[k].sample(data) * self.offsets[k][data.pm_batch_effect_indices[k]], dims=("datapoints",))
                acc += self.scaled_offsets[k]
            return pm.Deterministic(self.name, acc, dims=("datapoints",) if self.dims is None or self.dims == () else self.dims)

    @property
    def has_random_effect(self):
        return True

    def to_dict(self):
        dct = super().to_dict()
        if hasattr(self, "model_reference"):
            del dct['model_reference']
        dct["mu"] = self.mu.to_dict()
        dct["sigma"] = self.sigma.to_dict()
        if hasattr(self, "sigmas"):
            for k, v in self.sigmas.items():
                dct[f"{k}_sigma"] = v.to_dict()
        # if hasattr(self, "scaled_offsets"):
        #     for k, v in self.scaled_offsets.items():
        #         dct[f"scaled_{k}_offset"] = v.to_dict()

        del dct["sigmas"]
        del dct["offsets"]
        del dct["scaled_offsets"]
        return dct

    @classmethod
    def from_dict(cls, dct):
        mu = Param.from_dict(dct["mu"])
        sigma = Param.from_dict(dct["sigma"])
        instance = cls( 
            mu=mu,
            sigma=sigma,
            **{
                k: v
                for k, v in dct.items()
                if k in ["name", "dims", "mapping", "mapping_params"]
            },
        )
        instance.sigmas = {k: Param.from_dict(v) for k, v in dct.items() if k.endswith("_sigma")}
        # instance.scaled_offsets = {k: Param.from_dict(v) for k, v in dct.items() if k.endswith("_offset")}
        return instance


class LinearParam(Param):
    def __init__(
        self,
        name: str = "theta",
        dims: Optional[Union[Tuple[str, ...], str]] = None,
        mapping: str = "identity",
        mapping_params: tuple[float, ...] = (0.0, 1.0),
        slope: Optional[Param] = None,
        intercept: Optional[Param] = None,
    ):
        super().__init__(name, dims, mapping, mapping_params)
        self.slope = slope or get_default_slope()
        self.slope.dims = ('covariates',) if self.slope.dims is None else self.slope.dims
        self.intercept = intercept or get_default_intercept()
        self.intercept.dims = dims if self.intercept.dims is None else self.intercept.dims
        self.sample_dims = ("datapoints",)
        self.set_name(self.name)

    def create_graph(
        self, model: pm.Model, idata: az.InferenceData | None, freedom: float = 1
    ):
        self.slope.create_graph(model, idata, freedom)
        self.intercept.create_graph(model, idata, freedom)

    def _sample(self, data: HBRData):
        return math.sum(
            self.slope.sample(data) * data.pm_X, axis=1
        ) + self.intercept.sample(data)

    def to_dict(self):
        dct = super().to_dict()
        dct["slope"] = self.slope.to_dict()
        dct["intercept"] = self.intercept.to_dict()
        return dct

    @classmethod
    def from_dict(cls, dct):
        slope = Param.from_dict(dct["slope"])
        intercept = Param.from_dict(dct["intercept"])
        return cls(
            slope=slope,
            intercept=intercept,
            **{
                k: v
                for k, v in dct.items()
                if k in ["name", "dims", "mapping", "mapping_params"]
            },
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


def get_default_mu(dims: Optional[Union[Tuple[str, ...], str]] = None) -> Param:
    return LinearParam(
        "mu",
        dims=dims,
        slope=get_default_slope(dims),
        intercept=get_default_intercept(dims),
    )


def get_default_sigma(dims: Optional[Union[Tuple[str, ...], str]] = None) -> Param:
    slope = get_default_slope(dims)
    intercept = get_default_intercept(dims)
    return LinearParam(
        slope=slope,
        intercept=intercept,
        mapping="softplus",
        mapping_params=(0.0, 3.0),
    )


def get_default_epsilon() -> Param:
    return FixedParam(
        dist_name="Normal",
        dist_params=(
            0.0,
            1.0,
        ),
    )


def get_default_delta() -> Param:
    return FixedParam(
        dist_name="Normal",
        dist_params=(
            0.0,
            2.0,
        ),
        mapping="softplus",
        mapping_params=(0.0, 3.0, 0.3),
    )

def get_default_sub_mu(dims: Optional[Union[Tuple[str, ...], str]] = None) -> Param:
    return FixedParam(
        dims=dims,
        dist_name="Normal",
        dist_params=(
            0.0,
            10.0,
        ),
    )


def get_default_sub_sigma(dims: Optional[Union[Tuple[str, ...], str]] = None) -> Param:
    return FixedParam(
        dims=dims,
        dist_name="LogNormal",
        dist_params=(2.0,),
    )



def get_default_slope(
    dims: Optional[Union[Tuple[str, ...], str]] = ("covariates",),
) -> Param:
    return FixedParam(
        dims=dims,
        dist_name="Normal",
        dist_params=(0, 10.0),
    )


def get_default_intercept(dims: Optional[Union[Tuple[str, ...], str]] = None) -> Param:
    return FixedParam(
        dims=dims,
        dist_name="Normal",
        dist_params=(0, 10.0),
    )
