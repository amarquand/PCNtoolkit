from abc import ABC, abstractmethod
from typing import Any, Dict, List

import arviz as az
import pymc as pm
import pytensor.tensor as tensor

from pcntoolkit.regression_model.hbr.hbr_data import HBRData
from pcntoolkit.regression_model.hbr.prior import BasePrior, prior_from_args


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
    def get_samples(data: HBRData, prior: BasePrior):
        samples = prior.sample(data)
        shape = samples.shape.eval()
        if len(shape) == 1:
            if shape[0] == 1:
                return tensor.repeat(samples, data.y.shape[0], axis=0)
        return samples

    @staticmethod
    def from_dict(dct: Dict[str, Any]) -> "Likelihood":
        if dct["name"] == "Normal":
            return NormalLikelihood(mu=BasePrior.from_dict(dct["mu"]), sigma=BasePrior.from_dict(dct["sigma"]))
        raise ValueError(f"Unknown likelihood: {dct['name']}")
        # elif dct["name"].startswith("SHASH"):
        #     return SHASHLikelihood(mu=BasePrior.from_dict(dct["mu"]), sigma=BasePrior.from_dict(dct["sigma"]), epsilon=BasePrior.from_dict(dct["epsilon"]), delta=BasePrior.from_dict(dct["delta"]))
        # elif dct["name"] == "beta":
        #     return BetaLikelihood(alpha=BasePrior.from_dict(dct["alpha"]), beta=BasePrior.from_dict(dct["beta"]))
        # return Likelihood(name=dct["name"])

    @staticmethod
    def from_args(args: Dict[str, Any]) -> "Likelihood":
        if args["likelihood"] == "Normal":
            return NormalLikelihood(mu=prior_from_args("mu", args), sigma=prior_from_args("sigma", args))
        raise ValueError(f"Unknown likelihood: {args['likelihood']}")

    @abstractmethod
    def to_dict(self) -> Dict[str, Any]:
        pass

    @abstractmethod
    def get_var_names(self) -> List[str]:
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
            mu_samples = pm.Deterministic("mu_samples", self.get_samples(data, self.mu), dims=("datapoints",))
            sigma_samples = pm.Deterministic("sigma_samples", self.get_samples(data, self.sigma), dims=("datapoints",))
            pm.Normal("y_pred", mu=mu_samples, sigma=sigma_samples, observed=data.y, dims=("datapoints",))
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

    def get_var_names(self) -> List[str]:
        return ["mu_samples", "sigma_samples"]
