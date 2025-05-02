from __future__ import annotations

import copy
import os
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple, Union

import arviz as az  # type: ignore
import matplotlib.pyplot as plt
import numpy as np
import pymc as pm  # type: ignore
import scipy.stats as stats
import xarray as xr
from pymc import math

from pcntoolkit.math_functions.basis_function import BasisFunction, LinearBasisFunction
from pcntoolkit.math_functions.factorize import *
from pcntoolkit.math_functions.shash import S, S_inv, SHASHb, SHASHo, SHASHo2, m
from pcntoolkit.regression_model.regression_model import RegressionModel
from pcntoolkit.util.output import Errors, Output


class HBR(RegressionModel):
    """
    Hierarchical Bayesian Regression model implementation.

    This class implements a Bayesian hierarchical regression model using PyMC for
    posterior sampling. It supports multiple likelihood functions and provides
    methods for model fitting, prediction, and analysis.
    """

    def __init__(
        self,
        name: str,
        likelihood: Likelihood,
        draws: int = 1500,
        tune: int = 500,
        cores: int = 4,
        chains: int = 4,
        nuts_sampler: str = "nutpie",
        init: str = "jitter+adapt_diag",
        progressbar: bool = True,
        is_fitted: bool = False,
        is_from_dict: bool = False,
    ):
        """
        This class implements a Bayesian hierarchical regression model using PyMC for
        posterior sampling.

        Parameters
        ----------
        name : str
            Unique identifier for the model instance
        likelihood : Likelihood
            Likelihood function to use for the model
        draws : int, optional
            Number of samples to draw from the posterior distribution per chain, by default 1000
        tune : int, optional
            Number of tuning samples to draw from the posterior distribution per chain, by default 500
        cores : int, optional
            Number of cores to use for parallel sampling, by default 4
        chains : int, optional
            Number of chains to use for parallel sampling, by default 4
        nuts_sampler : str, optional
            NUTS sampler to use for parallel sampling, by default "nutpie"
        init : str, optional
            Initialization method for the model, by default "jitter+adapt_diag"
        progressbar : bool, optional
            Whether to display a progress bar during sampling, by default True
        is_fitted : bool, optional
            Whether the model has been fitted, by default False
        is_from_dict : bool, optional
            Whether the model was created from a dictionary, by default False

        """
        super().__init__(name, is_fitted, is_from_dict)
        self.likelihood = likelihood or NormalLikelihood(get_default_mu(), get_default_sigma())
        self.draws = draws
        self.tune = tune
        self.cores = cores
        self.chains = chains
        self.nuts_sampler = nuts_sampler
        self.init = init
        self.progressbar = progressbar
        self.idata: az.InferenceData = None  # type: ignore
        self.pymc_model: pm.Model = None  # type: ignore

    def fit(self, X: xr.DataArray, be: xr.DataArray, be_maps: dict[str, dict[str, int]], Y: xr.DataArray) -> None:
        """
        Fit the model to training data using MCMC sampling.

        Parameters
        ----------
        X : xr.DataArray
            Covariate data
        be : xr.DataArray
            Batch effect data
        be_maps : dict[str, dict[str, int]]
            Batch effect maps
        Y : xr.DataArray
            Response variable data

        Returns
        -------
        None
        """
        self.pymc_model: pm.Model = self.likelihood.compile(X, be, be_maps, Y)
        with self.pymc_model:
            self.idata = pm.sample(
                self.draws,
                tune=self.tune,
                cores=self.cores,
                chains=self.chains,
                nuts_sampler=self.nuts_sampler,  # type: ignore
                init=self.init,
                progressbar=self.progressbar,
            )
        self.is_fitted = True

    def forward(self, X: xr.DataArray, be: xr.DataArray, be_maps: dict[str, dict[str, int]], Y: xr.DataArray) -> xr.DataArray:
        """
        Map Y values to Z space using MCMC samples

        Parameters
        ----------
        X : xr.DataArray
            Covariate data
        be : xr.DataArray
            Batch effect data
        be_maps : dict[str, dict[str, int]]
            Batch effect maps
        Y : xr.DataArray
            Response variable data

        Returns
        -------
        xr.DataArray
            Z-values mapped to Y space
        """
        fn = self.likelihood.forward
        kwargs = {"Y": np.squeeze(Y.values)[:, None]}
        return self.generic_MCMC_apply(X, be, be_maps, Y, fn, kwargs)

    def backward(self, X, be, be_maps, Z) -> xr.DataArray:  # type: ignore
        """
        Map Z values to Y space using MCMC samples

        Parameters
        ----------
        X : xr.DataArray
            Covariate data
        be : xr.
            Batch effect data
        be_maps : dict[str, dict[str, int]]
            Batch effect maps
        Z : xr.DataArray
            Z-score data

        Returns
        -------
        xr.DataArray
            Z-values mapped to Y space
        """
        Y = xr.DataArray(np.zeros_like(Z.values), dims=Z.dims)
        fn = self.likelihood.backward
        kwargs = kwargs = {"Z": np.squeeze(Z.values)[:, None]}
        return self.generic_MCMC_apply(X, be, be_maps, Y, fn, kwargs)

    def generic_MCMC_apply(self, X, be, be_maps, Y, fn, kwargs):
        """
        Apply a generic function to likelihood parameters
        """
        if not self.is_fitted:
            raise ValueError(Output.error(Errors.HBR_MODEL_NOT_FITTED))

        model = self.likelihood.create_model_with_data(X, be, be_maps, Y)
        params = self.likelihood.compile_params(model, X, be, be_maps, Y)
        var_names = [f"{k}_per_subject" for k, _ in params.items()]
        with model:
            for param_name, (value, dims) in params.items():
                pm.Deterministic(f"{param_name}_per_subject", value, dims=dims)
            idata = pm.sample_posterior_predictive(
                self.idata,
                extend_inferencedata=False,
                var_names=var_names,
                progressbar=False,
            )

        post_pred = az.extract(
            idata,
            "posterior_predictive",
            var_names=var_names,
        )

        n_subjects = model.dim_lengths["subjects"].eval().item()
        array_of_vars = list(map(lambda x: self.extract_and_reshape(post_pred, n_subjects, x), var_names))
        result = xr.apply_ufunc(fn, *array_of_vars, kwargs=kwargs).mean(dim="sample")
        return result

    def elemwise_logp(self, X, be, be_maps, Y) -> xr.DataArray:  # type: ignore
        """
        Compute log-probabilities for each observation in the data.

        Parameters
        ----------
        X : xr.DataArray
            Covariate data
        be : xr.DataArray
            Batch effect data
        be_maps : dict[str, dict[str, int]]
            Batch effect maps
        Y : xr.DataArray
            Response variable data

        Returns
        -------
        xr.DataArray
            Log-probabilities of the data
        """

        if not self.is_fitted:
            raise ValueError(Output.error(Errors.HBR_MODEL_NOT_FITTED))
        if not self.pymc_model:
            self.pymc_model = self.likelihood.compile(X, be, be_maps, Y)
        else:
            self.likelihood.update_data(self.pymc_model, X, be, be_maps, Y)
        with self.pymc_model:
            logp = pm.compute_log_likelihood(
                self.idata,
                var_names=["Yhat"],
                extend_inferencedata=False,
                progressbar=False,
            )
        return az.extract(logp, "log_likelihood", var_names=["Yhat"]).mean("sample")

    def model_specific_evaluation(self, path: str) -> None:
        """
        Save model-specific evaluation metrics.
        """
        plotdir = os.path.join(path, "plots")
        os.makedirs(plotdir, exist_ok=True)
        resultsdir = os.path.join(path, "results")
        os.makedirs(resultsdir, exist_ok=True)
        if self.is_fitted:
            if self.idata is not None:
                az.summary(self.idata, fmt="wide", var_names=["~_per_subject"], filter_vars="like").to_csv(
                    os.path.join(resultsdir, self.name + "_summary.csv")
                )
                az.plot_trace(self.idata, var_names="~_per_subject", filter_vars="like")
                plt.tight_layout()
                plt.savefig(os.path.join(plotdir, self.name + "_trace.png"))
                plt.close()
                az.plot_autocorr(self.idata, var_names="~_per_subject", filter_vars="like")
                plt.tight_layout()
                plt.savefig(os.path.join(plotdir, self.name + "_autocorr.png"))
                plt.close()
                if hasattr(self.idata, "posterior_predictive"):
                    az.plot_ppc(self.idata)
                    plt.tight_layout()
                    plt.savefig(os.path.join(plotdir, self.name + "_ppc.png"))
                    plt.close()
            if self.pymc_model is not None:
                self.pymc_model.to_graphviz(save=os.path.join(plotdir, self.name + "_model.png"))
        else:
            raise ValueError(Output.error(Errors.HBR_MODEL_NOT_FITTED))

    def transfer(
        self,
        X: xr.DataArray,
        be: xr.DataArray,
        be_maps: dict[str, dict[str, int]],
        Y: xr.DataArray,
        **kwargs,
    ) -> HBR:
        """
                Perform transfer learning using existing model as prior.

                Parameters
                ----------
                hbrconf : HBRConf
                    Configuration for new model
                transferdata : HBRData
                    Data for transfer learning
                freedom : float
                    Parameter controlling influence of prior model (0-1)
        x
                Returns
                -------
                HBR
                    New model instance with transferred knowledge
        """
        new_likelihood = self.likelihood.transfer(self.idata, **kwargs)
        new_hbr_model = HBR(
            self.name,
            new_likelihood,
            self.draws,
            self.tune,
            self.cores,
            self.chains,
            self.nuts_sampler,
            self.init,
            self.progressbar,
            self.is_fitted,
            self.is_from_dict,
        )
        new_hbr_model_model = new_hbr_model.likelihood.compile(X, be, be_maps, Y)
        with new_hbr_model_model:
            new_hbr_model.idata = pm.sample(
                kwargs.get("draws", self.draws),
                tune=kwargs.get("tune", self.tune),
                cores=kwargs.get("cores", self.cores),
                chains=kwargs.get("chains", self.chains),
                nuts_sampler=kwargs.get("nuts_sampler", self.nuts_sampler),  # type: ignore
                progressbar=kwargs.get("progressbar", self.progressbar),
            )
            new_hbr_model.is_fitted = True
        new_hbr_model.pymc_model = new_hbr_model_model
        return new_hbr_model

    def has_batch_effect(self) -> bool:
        return False

    def extract_and_reshape(self, post_pred, subjects, var_name: str) -> xr.DataArray:
        preds = post_pred[var_name].values
        if len(preds.shape) == 1:
            preds = np.repeat(preds[None, :], subjects, axis=0)
        return xr.DataArray(np.squeeze(preds), dims=["subjects", "sample"])

    def to_dict(self, path: Optional[str] = None) -> Dict[str, Any]:
        """
        Serialize model to dictionary format.

        Parameters
        ----------
        path : Optional[str], optional
            Path to save inference data, by default None

        Returns
        -------
        Dict[str, Any]
            Dictionary containing serialized model
        """
        my_dict = self.regmodel_dict
        my_dict["likelihood"] = self.likelihood.to_dict()
        for key, value in self.__dict__.items():
            if key not in ["likelihood", "pymc_model", "idata"]:
                my_dict[key] = value
        if self.is_fitted and (path is not None):
            idata_path = os.path.join(path, "idata.nc")
            self.save_idata(idata_path)
            my_dict["idata_path"] = idata_path
        return my_dict

    @classmethod
    def from_dict(cls, my_dict: Dict[str, Any], path: Optional[str] = None) -> "HBR":
        """
        Create model instance from serialized dictionary.

        Parameters
        ----------
        dict : Dict[str, Any]
            Dictionary containing serialized model
        path : Optional[str], optional
            Path to load inference data from, by default None

        Returns
        -------
        HBR
            New model instance
        """
        name: str = my_dict["name"]
        likelihood: Likelihood = Likelihood.from_dict(my_dict["likelihood"])
        draws: int = my_dict["draws"]
        tune: int = my_dict["tune"]
        cores: int = my_dict["cores"]
        chains: int = my_dict["chains"]
        nuts_sampler: str = my_dict["nuts_sampler"]
        init: str = my_dict["init"]
        progressbar: bool = my_dict["progressbar"]
        is_fitted: bool = my_dict["is_fitted"]
        is_from_dict: bool = True
        self = cls(name, likelihood, draws, tune, cores, chains, nuts_sampler, init, progressbar, is_fitted, is_from_dict)
        if is_fitted and (path is not None):
            idata_path = os.path.join(path, "idata.nc")
            self.load_idata(idata_path)
        return self

    @classmethod
    def from_args(cls, name: str, args: Dict[str, Any]) -> "HBR":
        """
        Create model instance from command line arguments.

        Parameters
        ----------
        name : str
            Name for new model instance
        args : Dict[str, Any]
            Dictionary of command line arguments

        Returns
        -------
        HBR
            New model instance
        """
        likelihood = Likelihood.from_args(args)
        draws = args.get("draws", 1000)
        tune = args.get("tune", 1000)
        cores = args.get("cores", 1)
        chains = args.get("chains", 1)
        nuts_sampler = args.get("nuts_sampler", "pymc")
        init = args.get("init", "auto")
        progressbar = args.get("progressbar", True)
        is_fitted = args.get("is_fitted", False)
        is_from_dict = True
        self = cls(name, likelihood, draws, tune, cores, chains, nuts_sampler, init, progressbar, is_fitted, is_from_dict)
        return self

    def save_idata(self, path: str) -> None:
        """
        Save inference data to NetCDF file.

        Parameters
        ----------
        path : str
            Path to save inference data to. Should end in '.nc'

        Returns
        -------
        None

        Raises
        ------
        RuntimeError
            If model is fitted but does not have inference data
        """
        if self.is_fitted:
            if hasattr(self, "idata"):
                self.idata.to_netcdf(path, groups=["posterior"])
            else:
                raise ValueError(Output.error(Errors.ERROR_HBR_FITTED_BUT_NO_IDATA))

    def load_idata(self, path: str) -> None:
        """
        Load inference data from NetCDF file.

        Parameters
        ----------
        path : str
            Path to load inference data from. Should end in '.nc'

        Returns
        -------
        None

        Raises
        ------
        RuntimeError
            If model is fitted but inference data cannot be loaded from path
        """
        if self.is_fitted:
            try:
                self.idata = az.from_netcdf(path)
            except Exception as exc:
                raise ValueError(Output.error(Errors.ERROR_HBR_COULD_NOT_LOAD_IDATA, path=path)) from exc


PM_DISTMAP = {
    "Normal": pm.Normal,
    # "Cauchy": pm.Cauchy,
    "HalfNormal": pm.HalfNormal,
    # "HalfCauchy": pm.HalfCauchy,
    "Uniform": pm.Uniform,
    "Gamma": pm.Gamma,
    # "InvGamma": pm.InverseGamma,
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
        if mapping == "identity":
            if dist_name in ["Normal", "Cauchy"] or (dist_name == "Uniform" and dist_params[0] <= 0):
                raise ValueError(Output.error(Errors.ENSURE_POSITIVE_DISTRIBUTION, name=name))
    else:
        dist_name = my_args.get(f"dist_name_{name}", "Normal")
        dist_params = my_args.get(f"dist_params_{name}", (0, 1))

    dims = my_args.get(f"dims_{name}", dims)
    if my_args.get(f"linear_{name}", False):
        slope = prior_from_args(f"slope_{name}", my_args, dims=dims)
        intercept = prior_from_args(f"intercept_{name}", my_args, dims=dims)
        basis_function = BasisFunction.from_args(f"basis_function_{name}", my_args)
        return LinearPrior(name, dims, mapping, mapping_params, slope, intercept, basis_function)
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
        mapping_params: tuple = None,  # type: ignore
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
        self.mapping_params = mapping_params or (0, 1)
        self.sample_dims = ()

    @property
    def dims(self):
        return self._dims

    @dims.setter
    def dims(self, value):
        self._dims = value

    def apply_mapping(self, x: Any) -> Any:
        a, b = self.mapping_params[0], self.mapping_params[1]
        if self.mapping == "identity":
            toreturn = x
        elif self.mapping == "exp":
            toreturn = math.exp((x - a) / b) * b
        elif self.mapping == "softplus":
            toreturn = math.log(1 + math.exp((x - a) / b)) * b  # type: ignore
        else:
            raise ValueError(Output.error(Errors.ERROR_UNKNOWN_MAPPING, mapping=self.mapping))
        if len(self.mapping_params) > 2:
            toreturn = toreturn + self.mapping_params[2]
        return toreturn

    def compile(
        self, model: pm.Model, X: xr.DataArray, be: xr.DataArray, be_maps: dict[str, dict[str, int]], Y: xr.DataArray
    ) -> Any:
        samples = self._compile(model, X, be, be_maps, Y)
        return self.apply_mapping(samples)

    @abstractmethod
    def transfer(self, idata: az.InferenceData, **kwargs) -> "BasePrior":
        pass

    @abstractmethod
    def _compile(self, model, X, be, be_maps, Y) -> Any:
        pass

    @abstractmethod
    def update_data(self, model, X, be, be_maps, Y):
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
        mapping_params: tuple[float, ...] = None,  # type: ignore
        dist_name: str = "Normal",
        dist_params: Tuple[float | int | list[float | int], ...] = None,  # type: ignore
        **kwargs,
    ):
        super().__init__(name, dims, mapping, mapping_params, **kwargs)
        self.dist_name = dist_name
        self.dist_params = dist_params or (0, 10.0)
        self.sample_dims = ()

    def _compile(
        self,
        model: pm.Model,
        X: xr.DataArray,
        be: xr.DataArray,
        be_maps: dict[str, dict[str, int]],
        Y: xr.DataArray,
    ):
        with model:
            self.dist = PM_DISTMAP[self.dist_name](self.name, *self.dist_params, dims=self.dims)
        return self.dist

    def transfer(self, idata: az.InferenceData, **kwargs) -> "Prior":
        new_prior = Prior(self.name, self.dims, self.mapping, self.mapping_params, self.dist_name, self.dist_params)
        freedom = kwargs.get("freedom", 1)

        def infer_params(s):
            if self.dist_name == "Normal":
                return factorize_normal(s, freedom)
            elif self.dist_name == "HalfNormal":
                return factorize_halfnormal(s, freedom)
            elif self.dist_name == "LogNormal":
                return factorize_lognormal(s, freedom)
            elif self.dist_name == "Uniform":
                return factorize_uniform(s, freedom)
            elif self.dist_name == "Gamma":
                return factorize_gamma(s, freedom)
            # elif self.dist_name == "InvGamma":
            #     return factorize_invgamma(s, freedom)
            else:
                raise ValueError(Output.error(Errors.ERROR_UNKNOWN_DISTRIBUTION, dist_name=self.dist_name))

        samples = az.extract(idata, var_names=self.name)
        covariate_dims = [i for i in samples.dims if i.endswith("covariates")]
        if len(covariate_dims) == 1:
            params = [infer_params(samples.sel(**{covariate_dims[0]: i})) for i in samples.coords[covariate_dims[0]]]
            new_prior.dist_params = [i.tolist() for i in np.array(params).T]
        elif len(covariate_dims) == 0:
            new_prior.dist_params = infer_params(samples)
        else:
            raise ValueError(Output.error(Errors.ERROR_MULTIPLE_COVARIATE_DIMS, covariate_dims=covariate_dims))
        return new_prior

    def update_data(
        self, model: pm.Model, X: xr.DataArray, be: xr.DataArray, be_maps: dict[str, dict[str, int]], Y: xr.DataArray
    ):
        pass

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
        mapping_params: tuple[float, ...] = None,  # type: ignore
        mu: Optional[BasePrior] = None,
        sigma: Optional[BasePrior] = None,
        **kwargs,
    ):
        super().__init__(name, dims, mapping, mapping_params, **kwargs)
        self.mu = mu or get_default_sub_mu(dims)
        self.mu.set_name(f"mu_{self.name}")
        self.sigma = sigma or get_default_sub_sigma(dims)
        self.sigma.set_name(f"sigma_{self.name}")
        self.sigmas = {}
        self.offsets = {}
        self.scaled_offsets = {}
        self.sample_dims = ("subjects",)

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

    def _compile(
        self,
        model: pm.Model,
        X: xr.DataArray,
        be: xr.DataArray,
        be_maps: dict[str, dict[str, int]],
        Y: xr.DataArray,
    ):
        # outdims = "subjects" if not self.dims else ("subjects", *self.dims)
        with model:
            self.mu.compile(model, X, be, be_maps, Y)
            if self.dims:
                acc = self.mu.dist[None]  # type: ignore
            else:
                acc = self.mu.dist  # type: ignore
            for be_i in model.coords["batch_effect_dims"]:  # type:ignore
                be_dims = (be_i,) if not self.dims else (be_i, *self.dims)
                if be_i not in self.sigmas:
                    self.sigmas[be_i] = copy.deepcopy(self.sigma)
                    self.sigmas[be_i].set_name(f"{be_i}_sigma_{self.name}")
                self.sigmas[be_i].compile(model, X, be, be_maps, Y)
                self.scaled_offsets[be_i] = pm.Deterministic(
                    f"{be_i}_offset_{self.name}",
                    self.sigmas[be_i].dist  # type: ignore
                    * pm.Normal(
                        f"normalized_{be_i}_offset_{self.name}",
                        dims=be_dims,  # type:ignore
                    ),
                    dims=be_dims,
                )
                acc += self.scaled_offsets[be_i][model[f"{be_i}_data"]]
            self.dist = acc
        return self.dist

    def transfer(self, idata: az.InferenceData, **kwargs) -> "RandomPrior":
        new_mu = self.mu.transfer(idata, **kwargs)
        new_sigma = copy.deepcopy(self.sigma)
        new_prior = RandomPrior(
            name=self.name, dims=self.dims, mapping=self.mapping, mapping_params=self.mapping_params, mu=new_mu, sigma=new_sigma
        )
        for be_i in self.sigmas.keys():
            new_prior.sigmas[be_i] = self.sigmas[be_i].transfer(idata, **kwargs)
        return new_prior

    def update_data(
        self, model: pm.Model, X: xr.DataArray, be: xr.DataArray, be_maps: dict[str, dict[str, int]], Y: xr.DataArray
    ):
        pass

    def set_name(self, name: str):
        self.name = name
        self.mu.set_name(f"mu_{self.name}")
        self.sigma.set_name(f"sigma_{self.name}")

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
        instance.sigmas = {k.split("_")[0]: BasePrior.from_dict(v) for k, v in dct.items() if k.endswith("_sigma")}
        # instance.scaled_offsets = {k: Param.from_dict(v) for k, v in dct.items() if k.endswith("_offset")}
        return instance


class LinearPrior(BasePrior):
    def __init__(
        self,
        name: str = "theta",
        dims: Optional[Union[Tuple[str, ...], str]] = None,
        mapping: str = "identity",
        mapping_params: tuple[float, ...] = None,  # type: ignore
        slope: Optional[BasePrior] = None,
        intercept: Optional[BasePrior] = None,
        basis_function: BasisFunction = LinearBasisFunction(),
        **kwargs,
    ):
        super().__init__(name, dims, mapping, mapping_params, **kwargs)
        self.slope = slope or get_default_slope()
        self.slope.dims = ("covariates",) if not self.dims else ("covariates", *self.dims)
        self.intercept = intercept or get_default_intercept()
        self.intercept.dims = self.dims
        self.sample_dims = ("subjects",)
        self.set_name(self.name)
        self.basis_function = basis_function

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

    def _compile(self, model: pm.Model, X: xr.DataArray, be: xr.DataArray, be_maps: dict[str, dict[str, int]], Y: xr.DataArray):
        if not self.basis_function.is_fitted:
            self.basis_function.fit(X.values)  # Do this indexing to avoid ordering issues
        mapped_X = self.basis_function.transform(X.values)
        covs = f"{self.name}_covariates"
        self.covariate_dims = [f"{covs}_{i}" for i in range(mapped_X.shape[1])]

        if not model.coords.get(covs):
            model.add_coords({covs: self.covariate_dims})
        with model:
            pm_X = pm.Data(f"{self.name}_X", mapped_X, dims=("subjects", covs))

        self.slope.dims = (covs,) if not self.dims else (covs, *self.dims)
        self.intercept.dims = self.dims
        self.one_dimensional = len(self.covariate_dims) == 1  # type: ignore
        slope_samples = self.slope.compile(model, X, be, be_maps, Y)
        intercept_samples = self.intercept.compile(model, X, be, be_maps, Y)
        if self.one_dimensional:
            return (slope_samples * pm_X)[:, 0] + intercept_samples
        else:
            return math.sum(slope_samples * pm_X, axis=1, keepdims=False) + intercept_samples

    def transfer(self, idata: az.InferenceData, **kwargs) -> "LinearPrior":
        new_slope = self.slope.transfer(idata, **kwargs)
        new_intercept = self.intercept.transfer(idata, **kwargs)
        new_basis_function = copy.deepcopy(self.basis_function)
        new_basis_function.compute_min = False
        new_basis_function.compute_max = False
        new_prior = LinearPrior(
            name=self.name,
            dims=self.dims,
            mapping=self.mapping,
            mapping_params=self.mapping_params,
            slope=new_slope,
            intercept=new_intercept,
            basis_function=new_basis_function,
        )
        return new_prior

    def update_data(
        self, model: pm.Model, X: xr.DataArray, be: xr.DataArray, be_maps: dict[str, dict[str, int]], Y: xr.DataArray
    ):
        mapped_X = self.basis_function.transform(X.values)
        model.set_data(f"{self.name}_X", mapped_X, coords={"subjects": X.coords["subjects"].values})

    def to_dict(self):
        dct = super().to_dict()
        dct["slope"] = self.slope.to_dict()
        dct["intercept"] = self.intercept.to_dict()
        dct["basis_function"] = self.basis_function.to_dict()
        return dct

    @classmethod
    def from_dict(cls, dct):
        slope = BasePrior.from_dict(dct["slope"])
        intercept = BasePrior.from_dict(dct["intercept"])
        basis_function = BasisFunction.from_dict(dct["basis_function"])
        return cls(
            slope=slope,
            intercept=intercept,
            basis_function=basis_function,
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


def get_default_alpha() -> BasePrior:
    return Prior(
        dist_name="Gamma",
        dist_params=(
            2.0,
            0.5,
        ),
    )


def get_default_beta() -> BasePrior:
    return Prior(
        dist_name="Gamma",
        dist_params=(
            2.0,
            0.5,
        ),
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


# ≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠
# Likelihoods
# ≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠


class Likelihood(ABC):
    def __init__(self, name: str):
        self.name = name

    def compile(self, X: xr.DataArray, be: xr.DataArray, be_maps: dict[str, dict[str, int]], Y: xr.DataArray) -> pm.Model:
        model = self.create_model_with_data(X, be, be_maps, Y)
        self._compile(model, X, be, be_maps, Y)
        return model

    def create_model_with_data(self, X, be, be_maps, Y) -> pm.Model:
        coords = {"batch_effect_dims": be.coords["batch_effect_dims"].values, "subjects": X.coords["subjects"].values}
        for _be, _map in be_maps.items():
            coords[_be] = [k for k in sorted(_map.keys(), key=(lambda v: _map[v]))]

        model = pm.Model(coords=coords)
        with model:
            for be_name in be.coords["batch_effect_dims"].values:
                pm.Data(
                    f"{be_name}_data",
                    be.sel(batch_effect_dims=be_name).values,
                    dims=("subjects",),
                )
            pm.Data("Y", Y.values, dims=("subjects",))
        return model

    def update_data(
        self, model: pm.Model, X: xr.DataArray, be: xr.DataArray, be_maps: dict[str, dict[str, int]], Y: xr.DataArray
    ):
        with model:
            model.set_data(name="Y", values=Y.values, coords={"subjects": Y.coords["subjects"].values})
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
            # case "beta":
            #     return BetaLikelihood._from_dict(dct)
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
            # case "beta":
            #     return BetaLikelihood._from_args(args)
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
            pm.Normal("Yhat", **compiled_params, observed=model["Y"], dims="subjects")
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
        Z = kwargs.get("Z", None)
        return Z * sigma + mu

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
            SHASHb("Yhat", **compiled_params, observed=model["Y"], dims="subjects")
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
                dims="subjects",
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
                dims="subjects",
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
                dims="subjects",
            )
        return model

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
