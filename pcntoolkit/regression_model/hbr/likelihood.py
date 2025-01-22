from abc import ABC, abstractmethod
from typing import Any, Dict, List

import arviz as az
import numpy as np
import pymc as pm
import scipy.stats as stats

from pcntoolkit.regression_model.hbr.hbr_data import HBRData
from pcntoolkit.regression_model.hbr.hbr_util import S_inv, m
from pcntoolkit.regression_model.hbr.prior import BasePrior, prior_from_args
from pcntoolkit.regression_model.hbr.shash import SHASHb, SHASHo, SHASHo2


class Likelihood(ABC):
    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    def compile(self, data: HBRData, idata: az.InferenceData) -> pm.Model:
        pass

    @abstractmethod
    def zscore(self, *args, **kwargs):
        pass

    @abstractmethod
    def centile(self, *args, **kwargs):
        pass

    @staticmethod
    def from_dict(dct: Dict[str, Any]) -> "Likelihood":
        if dct["name"] == "Normal":
            return NormalLikelihood._from_dict(dct)
        elif dct["name"] == "SHASHb":
            return SHASHbLikelihood._from_dict(dct)
        elif dct["name"] == "SHASHo":
            return SHASHoLikelihood._from_dict(dct)
        elif dct["name"] == "SHASHo2":
            return SHASHo2Likelihood._from_dict(dct)
        elif dct["name"] == "beta":
            return BetaLikelihood._from_dict(dct)
        raise ValueError(f"Unknown likelihood: {dct['name']}")

    @staticmethod
    def from_args(args: Dict[str, Any]) -> "Likelihood":
        if args["likelihood"] == "Normal":
            return NormalLikelihood._from_args(args)
        elif args["likelihood"] == "SHASHb":
            return SHASHbLikelihood._from_args(args)
        elif args["likelihood"] == "SHASHo":
            return SHASHoLikelihood._from_args(args)
        elif args["likelihood"] == "SHASHo2":
            return SHASHo2Likelihood._from_args(args)
        elif args["likelihood"] == "beta":
            return BetaLikelihood._from_args(args)
        raise ValueError(f"Unknown likelihood: {args['likelihood']}")

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
    def get_var_names(self) -> List[str]:
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

    def compile(self, data: HBRData, idata: az.InferenceData) -> pm.Model:
        model = pm.Model(coords=data.coords)
        data.set_data_in_new_model(model)
        with model:
            self.mu.compile(model, idata)
            self.sigma.compile(model, idata)
            mu_samples = pm.Deterministic("mu_samples", self.mu.sample(data), dims=self.mu.sample_dims)
            sigma_samples = pm.Deterministic("sigma_samples", self.sigma.sample(data), dims=self.sigma.sample_dims)
            pm.Normal("y_pred", mu=mu_samples, sigma=sigma_samples, observed=data.pm_y, dims="datapoints")
        return model

    def zscore(self, *args, **kwargs):
        mu, sigma = args
        y = kwargs.get("y", None)
        return (y - mu) / sigma

    def centile(self, *args, **kwargs):
        mu, sigma = args
        zs = kwargs.get("zs", None)
        return zs * sigma + mu

    def to_dict(self) -> Dict[str, Any]:
        return {"name": self.name, "mu": self.mu.to_dict(), "sigma": self.sigma.to_dict()}

    @classmethod
    def _from_dict(cls, dct: Dict[str, Any]) -> "NormalLikelihood":
        return cls(mu=BasePrior.from_dict(dct["mu"]), sigma=BasePrior.from_dict(dct["sigma"]))

    @classmethod
    def _from_args(cls, args: Dict[str, Any]) -> "NormalLikelihood":
        return cls(mu=prior_from_args("mu", args), sigma=prior_from_args("sigma", args))

    def get_var_names(self) -> List[str]:
        return ["mu_samples", "sigma_samples"]

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

    def compile(self, data: HBRData, idata: az.InferenceData) -> pm.Model:
        model = pm.Model(coords=data.coords)
        data.set_data_in_new_model(model)
        with model:
            self.mu.compile(model, idata)
            self.sigma.compile(model, idata)
            self.epsilon.compile(model, idata)
            self.delta.compile(model, idata)
            mu_samples = pm.Deterministic("mu_samples", self.mu.sample(data), dims=self.mu.sample_dims)
            sigma_samples = pm.Deterministic("sigma_samples", self.sigma.sample(data), dims=self.sigma.sample_dims)
            epsilon_samples = pm.Deterministic("epsilon_samples", self.epsilon.sample(data), dims=self.epsilon.sample_dims)
            delta_samples = pm.Deterministic("delta_samples", self.delta.sample(data), dims=self.delta.sample_dims)
            SHASHb(
                "y_pred",
                mu=mu_samples,
                sigma=sigma_samples,
                epsilon=epsilon_samples,
                delta=delta_samples,
                observed=data.pm_y,
                dims="datapoints",
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

    def zscore(self, *args, **kwargs):
        mu, sigma, epsilon, delta = args
        y = kwargs.get("y", None)
        true_mu = m(epsilon, delta, 1)
        true_sigma = np.sqrt((m(epsilon, delta, 2) - true_mu**2))
        SHASH_c = (y - mu) / sigma
        SHASH = SHASH_c * true_sigma + true_mu
        Z = np.sinh(np.arcsinh(SHASH) * delta - epsilon)
        return Z

    def centile(self, *args, **kwargs):
        mu, sigma, epsilon, delta = args
        zs = kwargs.get("zs", None)
        true_mu = m(epsilon, delta, 1)
        true_sigma = np.sqrt((m(epsilon, delta, 2) - true_mu**2))
        SHASH_c = (S_inv(zs, epsilon, delta) - true_mu) / true_sigma
        quantiles = SHASH_c * sigma + mu
        return quantiles


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

    def compile(self, data: HBRData, idata: az.InferenceData) -> pm.Model:
        model = pm.Model(coords=data.coords)
        data.set_data_in_new_model(model)
        with model:
            self.mu.compile(model, idata)
            self.sigma.compile(model, idata)
            self.epsilon.compile(model, idata)
            self.delta.compile(model, idata)
            mu_samples = pm.Deterministic("mu_samples", self.mu.sample(data), dims=self.mu.sample_dims)
            sigma_samples = pm.Deterministic("sigma_samples", self.sigma.sample(data), dims=self.sigma.sample_dims)
            epsilon_samples = pm.Deterministic("epsilon_samples", self.epsilon.sample(data), dims=self.epsilon.sample_dims)
            delta_samples = pm.Deterministic("delta_samples", self.delta.sample(data), dims=self.delta.sample_dims)
            SHASHo(
                "y_pred",
                mu=mu_samples,
                sigma=sigma_samples,
                epsilon=epsilon_samples,
                delta=delta_samples,
                observed=data.pm_y,
                dims="datapoints",
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

    def zscore(self, *args, **kwargs):
        mu, sigma, epsilon, delta = args
        y = kwargs.get("y", None)
        SHASH = (y - mu) / sigma
        Z = np.sinh(np.arcsinh(SHASH) * delta - epsilon)
        return Z

    def centile(self, *args, **kwargs):
        mu, sigma, epsilon, delta = args
        zs = kwargs.get("zs", None)
        quantiles = S_inv(zs, epsilon, delta) * sigma + mu
        return quantiles


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

    def compile(self, data: HBRData, idata: az.InferenceData) -> pm.Model:
        model = pm.Model(coords=data.coords)
        data.set_data_in_new_model(model)
        with model:
            self.mu.compile(model, idata)
            self.sigma.compile(model, idata)
            self.epsilon.compile(model, idata)
            self.delta.compile(model, idata)
            mu_samples = pm.Deterministic("mu_samples", self.mu.sample(data), dims=self.mu.sample_dims)
            sigma_samples = pm.Deterministic("sigma_samples", self.sigma.sample(data), dims=self.sigma.sample_dims)
            epsilon_samples = pm.Deterministic("epsilon_samples", self.epsilon.sample(data), dims=self.epsilon.sample_dims)
            delta_samples = pm.Deterministic("delta_samples", self.delta.sample(data), dims=self.delta.sample_dims)
            SHASHo2(
                "y_pred",
                mu=mu_samples,
                sigma=sigma_samples,
                epsilon=epsilon_samples,
                delta=delta_samples,
                observed=data.pm_y,
                dims="datapoints",
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

    def zscore(self, *args, **kwargs):
        mu, sigma, epsilon, delta = args
        sigma_d = sigma / delta
        y = kwargs.get("y", None)
        SHASH = (y - mu) / sigma_d
        Z = np.sinh(np.arcsinh(SHASH) * delta - epsilon)
        return Z

    def centile(self, *args, **kwargs):
        mu, sigma, epsilon, delta = args
        sigma_d = sigma / delta
        zs = kwargs.get("zs", None)
        quantiles = S_inv(zs, epsilon, delta) * sigma_d + mu
        return quantiles


class BetaLikelihood(Likelihood):
    def __init__(self, alpha: BasePrior, beta: BasePrior):
        super().__init__(name="beta")
        self.alpha = alpha
        self.alpha.set_name("alpha")
        self.beta = beta
        self.beta.set_name("beta")

    def compile(self, data: HBRData, idata: az.InferenceData) -> pm.Model:
        model = pm.Model(coords=data.coords)
        data.set_data_in_new_model(model)
        with model:
            self.alpha.compile(model, idata)
            self.beta.compile(model, idata)
            alpha_samples = pm.Deterministic("alpha_samples", self.alpha.sample(data), dims=self.alpha.sample_dims)
            beta_samples = pm.Deterministic("beta_samples", self.beta.sample(data), dims=self.beta.sample_dims)
            pm.Beta("y_pred", alpha=alpha_samples, beta=beta_samples, observed=data.pm_y, dims="datapoints")
        return model

    def zscore(self, *args, **kwargs):
        alpha, beta = args
        y = kwargs.get("y", None)
        cdf = stats.beta.cdf(y, alpha, beta)
        Z = stats.norm.ppf(cdf)
        return Z

    def centile(self, *args, **kwargs):
        alpha, beta = args
        zs = kwargs.get("zs", None)
        cdf_norm = stats.norm.cdf(zs)
        quantiles = stats.beta.ppf(cdf_norm, alpha, beta)
        return quantiles

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
