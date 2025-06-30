from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List

import arviz as az  # type: ignore
import numpy as np
import pymc as pm  # type: ignore
import scipy.stats as stats
import xarray as xr

from pcntoolkit.math_functions.basis_function import BsplineBasisFunction
from pcntoolkit.math_functions.factorize import *
from pcntoolkit.math_functions.prior import BasePrior, make_prior, prior_from_args
from pcntoolkit.math_functions.shash import S, S_inv, SHASHb, SHASHo, SHASHo2, m


class Likelihood(ABC):
    def __init__(self, name: str):
        self.name = name

    def compile(self, X: xr.DataArray, be: xr.DataArray, be_maps: dict[str, dict[str, int]], Y: xr.DataArray) -> pm.Model:
        model = self.create_model_with_data(X, be, be_maps, Y)
        self._compile(model, X, be, be_maps, Y)
        return model

    def create_model_with_data(self, X, be, be_maps, Y) -> pm.Model:
        coords = {"batch_effect_dims": be.coords["batch_effect_dims"].values, "observations": X.coords["observations"].values}
        for _be, _map in be_maps.items():
            coords[_be] = [k for k in sorted(_map.keys(), key=(lambda v: _map[v]))]

        model = pm.Model(coords=coords)
        with model:
            for be_name in be.coords["batch_effect_dims"].values:
                pm.Data(
                    f"{be_name}_data",
                    be.sel(batch_effect_dims=be_name).values,
                    dims=("observations",),
                )
            pm.Data("Y", Y.values, dims=("observations",))
        return model

    def update_data(
        self, model: pm.Model, X: xr.DataArray, be: xr.DataArray, be_maps: dict[str, dict[str, int]], Y: xr.DataArray
    ):
        with model:
            model.set_data(name="Y", values=Y.values, coords={"observations": Y.coords["observations"].values})
            for be_name in be.coords["batch_effect_dims"].values:
                model.set_data(
                    name=f"{be_name}_data",
                    values=be.sel(batch_effect_dims=be_name).values,
                )
        self._update_data(model, X, be, be_maps, Y)

    @abstractmethod
    def _update_data(
        self, model: pm.Model, X: xr.DataArray, be: xr.DataArray, be_maps: dict[str, dict[str, int]], Y: xr.DataArray
    ):
        pass

    @abstractmethod
    def transfer(self, idata: az.InferenceData, **kwargs) -> "Likelihood":
        pass

    @abstractmethod
    def _compile(
        self,
        model: pm.Model,
        X: xr.DataArray,
        be: xr.DataArray,
        be_maps: dict[str, dict[str, int]],
        Y: xr.DataArray,
    ) -> pm.Model:
        pass

    @abstractmethod
    def compile_params(
        self,
        model: pm.Model,
        X: xr.DataArray,
        be: xr.DataArray,
        be_maps: dict[str, dict[str, int]],
        Y: xr.DataArray,
    ) -> dict[str, Any]:
        pass

    @abstractmethod
    def forward(self, *args, **kwargs):
        pass

    @abstractmethod
    def backward(self, *args, **kwargs):
        pass

    @abstractmethod
    def yhat(self, *args, **kwargs):
        pass

    @staticmethod
    def from_dict(dct: Dict[str, Any]) -> "Likelihood":
        likelihood = dct.pop("name", "Normal")
        match likelihood:
            case "Normal":
                return NormalLikelihood._from_dict(dct)
            case "SHASHb":
                return SHASHbLikelihood._from_dict(dct)
            # case "SHASHo":
            #     return SHASHoLikelihood._from_dict(dct)
            # case "SHASHo2":
            #     return SHASHo2Likelihood._from_dict(dct)
            case "beta":
                return BetaLikelihood._from_dict(dct)
            case _:
                raise ValueError(f"Unknown likelihood: {likelihood}")

    @staticmethod
    def from_args(args: Dict[str, Any]) -> "Likelihood":
        likelihood = args.pop("likelihood", "Normal")
        match likelihood:
            case "Normal":
                return NormalLikelihood._from_args(args)
            case "SHASHb":
                return SHASHbLikelihood._from_args(args)
            # case "SHASHo":
            #     return SHASHoLikelihood._from_args(args)
            # case "SHASHo2":
            #     return SHASHo2Likelihood._from_args(args)
            case "beta":
                return BetaLikelihood._from_args(args)
            case _:
                raise ValueError(f"Unknown likelihood: {likelihood}")

    @abstractmethod
    def to_dict(self) -> Dict[str, Any]:
        pass

    @classmethod
    @abstractmethod
    def _from_dict(cls, dct: Dict[str, Any]) -> "Likelihood":
        pass

    @classmethod
    @abstractmethod
    def _from_args(cls, args: Dict[str, Any]) -> "Likelihood":
        pass

    @abstractmethod
    def has_random_effect(self) -> bool:
        pass


class NormalLikelihood(Likelihood):
    def __init__(self, mu: BasePrior, sigma: BasePrior):
        super().__init__(name="Normal")
        self.mu = mu
        self.mu.set_name("mu")
        self.sigma = sigma
        self.sigma.set_name("sigma")

    def _compile(
        self,
        model: pm.Model,
        X: xr.DataArray,
        be: xr.DataArray,
        be_maps: dict[str, dict[str, int]],
        Y: xr.DataArray,
    ) -> pm.Model:
        compiled_params = self.compile_params(model, X, be, be_maps, Y)
        compiled_params = {k: v[0] for k, v in compiled_params.items()}
        with model:
            pm.Normal("Yhat", **compiled_params, observed=model["Y"], dims="observations")
        return model

    def compile_params(
        self,
        model: pm.Model,
        X: xr.DataArray,
        be: xr.DataArray,
        be_maps: dict[str, dict[str, int]],
        Y: xr.DataArray,
    ) -> dict[str, Any]:
        return {
            "mu": (self.mu.compile(model, X, be, be_maps, Y), self.mu.sample_dims),
            "sigma": (self.sigma.compile(model, X, be, be_maps, Y), self.sigma.sample_dims),
        }

    def transfer(self, idata: az.InferenceData, **kwargs) -> "Likelihood":
        new_mu = self.mu.transfer(idata, **kwargs)
        new_sigma = self.sigma.transfer(idata, **kwargs)
        return NormalLikelihood(new_mu, new_sigma)

    def _update_data(
        self, model: pm.Model, X: xr.DataArray, be: xr.DataArray, be_maps: dict[str, dict[str, int]], Y: xr.DataArray
    ):
        self.mu.update_data(model, X, be, be_maps, Y)
        self.sigma.update_data(model, X, be, be_maps, Y)

    def forward(self, *args, **kwargs):
        mu, sigma = args
        Y = kwargs.get("Y", None)
        return (Y - mu) / sigma

    def backward(self, *args, **kwargs):
        mu, sigma = args
        Z = kwargs.get("Z")
        return Z * sigma + mu

    def yhat(self, *args, **kwargs):
        mu, _ = args
        return mu

    def to_dict(self) -> Dict[str, Any]:
        return {"name": self.name, "mu": self.mu.to_dict(), "sigma": self.sigma.to_dict()}

    @classmethod
    def _from_dict(cls, dct: Dict[str, Any]) -> "NormalLikelihood":
        return cls(mu=BasePrior.from_dict(dct["mu"]), sigma=BasePrior.from_dict(dct["sigma"]))

    @classmethod
    def _from_args(cls, args: Dict[str, Any]) -> "NormalLikelihood":
        return cls(mu=prior_from_args("mu", args), sigma=prior_from_args("sigma", args))

    def has_random_effect(self) -> bool:
        return self.mu.has_random_effect or self.sigma.has_random_effect


class SHASHbLikelihood(Likelihood):
    def __init__(self, mu: BasePrior, sigma: BasePrior, epsilon: BasePrior, delta: BasePrior):
        super().__init__(name="SHASHb")
        self.mu = mu
        self.mu.set_name("mu")
        self.sigma = sigma
        self.sigma.set_name("sigma")
        self.epsilon = epsilon
        self.epsilon.set_name("epsilon")
        self.delta = delta
        self.delta.set_name("delta")

    def _compile(
        self,
        model: pm.Model,
        X: xr.DataArray,
        be: xr.DataArray,
        be_maps: dict[str, dict[str, int]],
        Y: xr.DataArray,
    ) -> pm.Model:
        compiled_params = self.compile_params(model, X, be, be_maps, Y)
        compiled_params = {k: v[0] for k, v in compiled_params.items()}
        with model:
            SHASHb("Yhat", **compiled_params, observed=model["Y"], dims="observations")
        return model

    def compile_params(
        self,
        model: pm.Model,
        X: xr.DataArray,
        be: xr.DataArray,
        be_maps: dict[str, dict[str, int]],
        Y: xr.DataArray,
    ) -> dict[str, Any]:
        return {
            "mu": (self.mu.compile(model, X, be, be_maps, Y), self.mu.sample_dims),
            "sigma": (self.sigma.compile(model, X, be, be_maps, Y), self.sigma.sample_dims),
            "epsilon": (self.epsilon.compile(model, X, be, be_maps, Y), self.epsilon.sample_dims),
            "delta": (self.delta.compile(model, X, be, be_maps, Y), self.delta.sample_dims),
        }

    def transfer(self, idata: az.InferenceData, **kwargs) -> "SHASHbLikelihood":
        new_mu = self.mu.transfer(idata, **kwargs)
        new_sigma = self.sigma.transfer(idata, **kwargs)
        new_epsilon = self.epsilon.transfer(idata, **kwargs)
        new_delta = self.delta.transfer(idata, **kwargs)
        return SHASHbLikelihood(new_mu, new_sigma, new_epsilon, new_delta)

    def _update_data(
        self, model: pm.Model, X: xr.DataArray, be: xr.DataArray, be_maps: dict[str, dict[str, int]], Y: xr.DataArray
    ):
        self.mu.update_data(model, X, be, be_maps, Y)
        self.sigma.update_data(model, X, be, be_maps, Y)
        self.epsilon.update_data(model, X, be, be_maps, Y)
        self.delta.update_data(model, X, be, be_maps, Y)

    def has_random_effect(self) -> bool:
        return (
            self.mu.has_random_effect
            or self.sigma.has_random_effect
            or self.epsilon.has_random_effect
            or self.delta.has_random_effect
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "mu": self.mu.to_dict(),
            "sigma": self.sigma.to_dict(),
            "epsilon": self.epsilon.to_dict(),
            "delta": self.delta.to_dict(),
        }

    @classmethod
    def _from_dict(cls, dct: Dict[str, Any]) -> "SHASHbLikelihood":
        return cls(
            mu=BasePrior.from_dict(dct["mu"]),
            sigma=BasePrior.from_dict(dct["sigma"]),
            epsilon=BasePrior.from_dict(dct["epsilon"]),
            delta=BasePrior.from_dict(dct["delta"]),
        )

    @classmethod
    def _from_args(cls, args: Dict[str, Any]) -> "SHASHbLikelihood":
        return cls(
            mu=prior_from_args("mu", args),
            sigma=prior_from_args("sigma", args),
            epsilon=prior_from_args("epsilon", args),
            delta=prior_from_args("delta", args),
        )

    def get_var_names(self) -> List[str]:
        return ["mu_samples", "sigma_samples", "epsilon_samples", "delta_samples"]

    def forward(self, *args, **kwargs):
        mu, sigma, epsilon, delta = args
        Y = kwargs.get("Y", None)
        true_mu = m(epsilon, delta, 1)
        true_sigma = np.sqrt((m(epsilon, delta, 2) - true_mu**2))
        SHASH_centered = (Y - mu) / sigma
        SHASH_uncentered = SHASH_centered * true_sigma + true_mu
        Z = S(SHASH_uncentered, epsilon, delta)
        return Z

    def backward(self, *args, **kwargs):
        mu, sigma, epsilon, delta = args
        Z = kwargs.get("Z", None)
        true_mu = m(epsilon, delta, 1)
        true_sigma = np.sqrt((m(epsilon, delta, 2) - true_mu**2))
        SHASH_uncentered = S_inv(Z, epsilon, delta)
        SHASH_centered = (SHASH_uncentered - true_mu) / true_sigma
        Y = SHASH_centered * sigma + mu
        return Y

    def yhat(self, *args, **kwargs):
        mu, _, _, _ = args
        return mu


class SHASHoLikelihood(Likelihood):
    def __init__(self, mu: BasePrior, sigma: BasePrior, epsilon: BasePrior, delta: BasePrior):
        super().__init__(name="SHASHo")
        self.mu = mu
        self.mu.set_name("mu")
        self.sigma = sigma
        self.sigma.set_name("sigma")
        self.epsilon = epsilon
        self.epsilon.set_name("epsilon")
        self.delta = delta
        self.delta.set_name("delta")

    def _compile(
        self,
        model: pm.Model,
        X: xr.DataArray,
        be: xr.DataArray,
        be_maps: dict[str, dict[str, int]],
        Y: xr.DataArray,
    ) -> pm.Model:
        with model:
            mu_samples = self.mu.compile(model, X, be, be_maps, Y)
            sigma_samples = self.sigma.compile(model, X, be, be_maps, Y)
            epsilon_samples = self.epsilon.compile(model, X, be, be_maps, Y)
            delta_samples = self.delta.compile(model, X, be, be_maps, Y)
            mu_samples = pm.Deterministic("mu_samples", mu_samples, dims=self.mu.sample_dims)
            sigma_samples = pm.Deterministic("sigma_samples", sigma_samples, dims=self.sigma.sample_dims)
            epsilon_samples = pm.Deterministic("epsilon_samples", epsilon_samples, dims=self.epsilon.sample_dims)
            delta_samples = pm.Deterministic("delta_samples", delta_samples, dims=self.delta.sample_dims)
            SHASHo(
                "Yhat",
                mu=mu_samples,
                sigma=sigma_samples,
                epsilon=epsilon_samples,
                delta=delta_samples,
                observed=model["Y"],
                dims="observations",
            )
        return model

    def has_random_effect(self) -> bool:
        return (
            self.mu.has_random_effect
            or self.sigma.has_random_effect
            or self.epsilon.has_random_effect
            or self.delta.has_random_effect
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "mu": self.mu.to_dict(),
            "sigma": self.sigma.to_dict(),
            "epsilon": self.epsilon.to_dict(),
            "delta": self.delta.to_dict(),
        }

    @classmethod
    def _from_dict(cls, dct: Dict[str, Any]) -> "SHASHoLikelihood":
        return cls(
            mu=BasePrior.from_dict(dct["mu"]),
            sigma=BasePrior.from_dict(dct["sigma"]),
            epsilon=BasePrior.from_dict(dct["epsilon"]),
            delta=BasePrior.from_dict(dct["delta"]),
        )

    @classmethod
    def _from_args(cls, args: Dict[str, Any]) -> "SHASHoLikelihood":
        return cls(
            mu=prior_from_args("mu", args),
            sigma=prior_from_args("sigma", args),
            epsilon=prior_from_args("epsilon", args),
            delta=prior_from_args("delta", args),
        )

    def get_var_names(self) -> List[str]:
        return ["mu_samples", "sigma_samples", "epsilon_samples", "delta_samples"]

    def forward(self, *args, **kwargs):
        mu, sigma, epsilon, delta = args
        y = kwargs.get("Y", None)
        SHASH = (y - mu) / sigma
        Z = np.sinh(np.arcsinh(SHASH) * delta - epsilon)
        return Z

    def backward(self, *args, **kwargs):
        mu, sigma, epsilon, delta = args
        Z = kwargs.get("Z", None)
        SHASH = S_inv(Z, epsilon, delta)
        Y = SHASH * sigma + mu
        return Y


class SHASHo2Likelihood(Likelihood):
    def __init__(self, mu: BasePrior, sigma: BasePrior, epsilon: BasePrior, delta: BasePrior):
        super().__init__(name="SHASHo2")
        self.mu = mu
        self.mu.set_name("mu")
        self.sigma = sigma
        self.sigma.set_name("sigma")
        self.epsilon = epsilon
        self.epsilon.set_name("epsilon")
        self.delta = delta
        self.delta.set_name("delta")

    def _compile(
        self,
        model: pm.Model,
        X: xr.DataArray,
        be: xr.DataArray,
        be_maps: dict[str, dict[str, int]],
        Y: xr.DataArray,
    ) -> pm.Model:
        with model:
            mu_samples = self.mu.compile(model, X, be, be_maps, Y)
            sigma_samples = self.sigma.compile(model, X, be, be_maps, Y)
            epsilon_samples = self.epsilon.compile(model, X, be, be_maps, Y)
            delta_samples = self.delta.compile(model, X, be, be_maps, Y)
            mu_samples = pm.Deterministic("mu_samples", mu_samples, dims=self.mu.sample_dims)
            sigma_samples = pm.Deterministic("sigma_samples", sigma_samples, dims=self.sigma.sample_dims)
            epsilon_samples = pm.Deterministic("epsilon_samples", epsilon_samples, dims=self.epsilon.sample_dims)
            delta_samples = pm.Deterministic("delta_samples", delta_samples, dims=self.delta.sample_dims)
            SHASHo2(
                "Yhat",
                mu=mu_samples,
                sigma=sigma_samples,
                epsilon=epsilon_samples,
                delta=delta_samples,
                observed=model["Y"],
                dims="observations",
            )
        return model

    def has_random_effect(self) -> bool:
        return (
            self.mu.has_random_effect
            or self.sigma.has_random_effect
            or self.epsilon.has_random_effect
            or self.delta.has_random_effect
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "mu": self.mu.to_dict(),
            "sigma": self.sigma.to_dict(),
            "epsilon": self.epsilon.to_dict(),
            "delta": self.delta.to_dict(),
        }

    @classmethod
    def _from_dict(cls, dct: Dict[str, Any]) -> "SHASHo2Likelihood":
        return cls(
            mu=BasePrior.from_dict(dct["mu"]),
            sigma=BasePrior.from_dict(dct["sigma"]),
            epsilon=BasePrior.from_dict(dct["epsilon"]),
            delta=BasePrior.from_dict(dct["delta"]),
        )

    @classmethod
    def _from_args(cls, args: Dict[str, Any]) -> "SHASHo2Likelihood":
        return cls(
            mu=prior_from_args("mu", args),
            sigma=prior_from_args("sigma", args),
            epsilon=prior_from_args("epsilon", args),
            delta=prior_from_args("delta", args),
        )

    def get_var_names(self) -> List[str]:
        return ["mu_samples", "sigma_samples", "epsilon_samples", "delta_samples"]

    def forward(self, *args, **kwargs):
        mu, sigma, epsilon, delta = args
        sigma_d = sigma / delta
        Y = kwargs.get("Y", None)
        SHASH = (Y - mu) / sigma_d
        Z = S(SHASH, epsilon, delta)
        return Z

    def backward(self, *args, **kwargs):
        mu, sigma, epsilon, delta = args
        sigma_d = sigma / delta
        Z = kwargs.get("Z", None)
        SHASH = S_inv(Z, epsilon, delta)
        Y = SHASH * sigma_d + mu
        return Y


class BetaLikelihood(Likelihood):
    def __init__(self, alpha: BasePrior, beta: BasePrior):
        super().__init__(name="beta")
        self.alpha = alpha
        self.alpha.set_name("alpha")
        self.beta = beta
        self.beta.set_name("beta")

    def _compile(
        self,
        model: pm.Model,
        X: xr.DataArray,
        be: xr.DataArray,
        be_maps: dict[str, dict[str, int]],
        Y: xr.DataArray,
    ) -> pm.Model:
        with model:
            alpha_samples = self.alpha.compile(model, X, be, be_maps, Y)
            beta_samples = self.beta.compile(model, X, be, be_maps, Y)

            alpha_samples = pm.Deterministic("alpha_samples", alpha_samples, dims=self.alpha.sample_dims)
            beta_samples = pm.Deterministic("beta_samples", beta_samples, dims=self.beta.sample_dims)
            pm.Beta(
                "Yhat",
                alpha=alpha_samples,
                beta=beta_samples,
                observed=model["Y"],
                dims="observations",
            )
        return model

    def compile_params(
        self,
        model: pm.Model,
        X: xr.DataArray,
        be: xr.DataArray,
        be_maps: dict[str, dict[str, int]],
        Y: xr.DataArray,
    ) -> dict[str, Any]:
        return {
            "alpha": (self.alpha.compile(model, X, be, be_maps, Y), self.alpha.sample_dims),
            "beta": (self.beta.compile(model, X, be, be_maps, Y), self.beta.sample_dims),
        }

    def transfer(self, idata: az.InferenceData, **kwargs) -> "BetaLikelihood":
        new_alpha = self.alpha.transfer(idata, **kwargs)
        new_beta = self.beta.transfer(idata, **kwargs)
        return BetaLikelihood(new_alpha, new_beta)

    def _update_data(
        self, model: pm.Model, X: xr.DataArray, be: xr.DataArray, be_maps: dict[str, dict[str, int]], Y: xr.DataArray
    ):
        self.alpha.update_data(model, X, be, be_maps, Y)
        self.beta.update_data(model, X, be, be_maps, Y)

    def forward(self, *args, **kwargs):
        alpha, beta = args
        Y = kwargs.get("Y", None)
        cdf = stats.beta.cdf(Y, alpha, beta)
        Z = stats.norm.ppf(cdf)
        return Z

    def backward(self, *args, **kwargs):
        alpha, beta = args
        Z = kwargs.get("Z", None)
        cdf_norm = stats.norm.cdf(Z)
        quantiles = stats.beta.ppf(cdf_norm, alpha, beta)
        return quantiles

    def yhat(self, *args, **kwargs):
        alpha, beta = args
        return alpha / (alpha + beta)

    def has_random_effect(self) -> bool:
        return self.alpha.has_random_effect or self.beta.has_random_effect

    def to_dict(self) -> Dict[str, Any]:
        return {"name": self.name, "alpha": self.alpha.to_dict(), "beta": self.beta.to_dict()}

    @classmethod
    def _from_dict(cls, dct: Dict[str, Any]) -> "BetaLikelihood":
        return cls(alpha=BasePrior.from_dict(dct["alpha"]), beta=BasePrior.from_dict(dct["beta"]))

    @classmethod
    def _from_args(cls, args: Dict[str, Any]) -> "BetaLikelihood":
        return cls(alpha=prior_from_args("alpha", args), beta=prior_from_args("beta", args))

    def get_var_names(self) -> List[str]:
        return ["alpha_samples", "beta_samples"]


def get_default_normal_likelihood() -> NormalLikelihood:
    "Return a normal likelihood with a random intercept of the mean, and heteroskedasticity"
    return NormalLikelihood(
        mu=make_prior(
            "mu",
            linear=True,
            slope=make_prior(dist_name="Normal", dist_params=(0, 10.0)),
            intercept=make_prior(
                random=True,
                mu=make_prior(
                    "mu_intercept_mu",
                    dist_name="Normal",
                    dist_params=(0, 1.0),
                    sigma=make_prior(
                        "sigma_intercept_mu",
                        dist_name="Normal",
                        dist_params=(1.0, 2, 0),
                        mapping="softplus",
                        mapping_params=(0.0, 3.0),
                    ),
                ),
            ),
            basis_function=BsplineBasisFunction(0, 3, 5),
        ),
        sigma=make_prior(
            "sigma",
            linear=True,
            slope=make_prior(dist_name="Normal", dist_params=(0, 10.0)),
            intercept=make_prior(dist_name="Normal", dist_params=(1.0, 3.0), mapping="softplus", mapping_params=(0.0, 3.0)),
            basis_function=BsplineBasisFunction(0, 3, 5),
        ),
    )
