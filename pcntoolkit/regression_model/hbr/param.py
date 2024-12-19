"""
Parameter handling module for Hierarchical Bayesian Regression models.

This module provides the Param class which handles parameter definitions,
distributions, and transformations for hierarchical Bayesian regression models.
It supports linear parameters, random effects, and various probability distributions.

The module integrates with PyMC for Bayesian modeling and provides utilities for
parameter initialization, transformation, and serialization.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple, Union

import arviz as az
import pymc as pm  # type: ignore
import scipy.stats as stats  # type: ignore
import xarray as xr
from pymc import math

from pcntoolkit.regression_model.hbr.hbr_data import HBRData


@dataclass
class Param:
    """
    A class representing parameters in hierarchical Bayesian regression models.

    This class handles parameter definitions including their distributions,
    linear relationships, random effects, and transformations. It supports
    both centered and non-centered parameterizations.

    Parameters
    ----------
    name : str
        Name of the parameter. Defaults to "theta". Used to infer the names of the sub-parameters.
    dims : Optional[Union[Tuple[str, ...], str]], optional
        Dimension names for the parameter, by default None
    dist_name : str, optional
        Name of the probability distribution, by default "Normal"
    dist_params : tuple, optional
        Parameters for the probability distribution, by default (0, 10.0)
    linear : bool, optional
        Whether parameter has linear relationship, by default False
    slope : Param, optional
        Slope parameter for linear relationships, by default None
    intercept : Param, optional
        Intercept parameter for linear relationships, by default None
    mapping : str, optional
        Transformation mapping type, by default "identity"
    mapping_params : tuple, optional
        Parameters for the transformation mapping, by default (0, 1)
    random : bool, optional
        Whether parameter has random effects, by default False
    centered : bool, optional
        Whether to use centered parameterization, by default False
    mu : Param, optional
        Mean parameter for random effects, by default None
    sigma : Param, optional
        Standard deviation parameter for random effects, by default None
    freedom : float, optional
        Degrees of freedom parameter, by default 1.0

    Attributes
    ----------
    has_covariate_dim : bool
        Whether parameter has covariate dimensions
    has_random_effect : bool
        Whether parameter has random effects
    distmap : Dict[str, Any]
        Mapping of distribution names to PyMC distribution classes
    dist : Any
        PyMC distribution object
    """

    name: str = "theta"
    dims: Optional[Union[Tuple[str, ...], str]] = None
    dist_name: str = "Normal"
    dist_params: tuple = (0, 10.0)

    linear: bool = False
    slope: Param = None  # type: ignore
    intercept: Param = None  # type: ignore
    mapping: str = "identity"
    mapping_params: tuple = (0, 1)

    random: bool = False
    centered: bool = False
    mu: Param = field(default=None)  # type: ignore
    sigma: Param = field(default=None)  # type: ignore
    offset: pm.TensorVariable = field(default=None)  # type: ignore

    has_random_effect: bool = field(init=False, default=False)
    distmap: Dict[str, Any] = field(init=False, default_factory=dict)
    dist: Any = field(init=False, default=None)

    freedom: float = 1.0

    def __post_init__(self) -> None:
        """
        Initialize parameter attributes after instance creation.

        Validates parameter configuration and sets up appropriate parameter structure
        based on whether it's linear, random, or basic parameter.

        Raises
        ------
        ValueError
            If slope parameter is missing required covariate dimension
        """

        self.distmap = {
            "Normal": pm.Normal,
            "Cauchy": pm.Cauchy,
            "HalfNormal": pm.HalfNormal,
            "HalfCauchy": pm.HalfCauchy,
            "Uniform": pm.Uniform,
            "Gamma": pm.Gamma,
            "InvGamma": pm.InverseGamma,
            "LogNormal": pm.LogNormal,
        }

        if self.linear:
            self.set_linear_params()
            self.sample_dims = ("datapoints",)

        elif self.random:
            if self.centered:
                self.set_centered_random_params()
            else:
                self.set_noncentered_random_params()
            self.sample_dims = ("datapoints",)

        else:
            # If the parameter is really only a single number, we need to add an empty dimension so our outputs are always 2D
            if (self.dims == ()) or (self.dims == []):
                self.dims = None
                self.shape = None
            else:
                self.shape = None
                if isinstance(self.dims, str):
                    self.dims = (self.dims,)
            self.sample_dims = ()  # type: ignore

        self.has_random_effect = (self.random and not self.linear) or (
            self.linear and (self.slope.random or self.intercept.random)
        )
        self.set_name(self.name)


    def create_graph(
        self, model: Any, idata: Optional[Any] = None, freedom: float = 1
    ) -> None:
        """
        Create PyMC computational graph for the parameter.

        Parameters
        ----------
        model : Any
            PyMC model object
        idata : Optional[Any], optional
            Inference data for parameter initialization, by default None
        freedom : float, optional
            Degrees of freedom parameter, by default 1

        Notes
        -----
        Creates appropriate PyMC variables based on parameter configuration
        (linear, random, or basic) and adds them to the model graph.
        """
        has_covariate_dim = False if not self.dims else "covariates" in self.dims
        if self.name.startswith("slope") and not has_covariate_dim:
            #! This is a bit of a hack to make sure that the slope parameter has a covariate dimension
            if self.dims is None:
                self.dims = ("covariates",)
            else:
                self.dims = (*self._dims, "covariates")
        self.freedom = freedom
        with model:
            if self.linear:
                self.slope.create_graph(model, idata, freedom)
                self.intercept.create_graph(model, idata, freedom)
            elif self.random:
                if self.centered:
                    self.mu.create_graph(model, idata, freedom)
                    self.sigma.create_graph(model, idata, freedom)
                    self.dist = pm.Normal(
                        self.name,
                        mu=self.mu.dist,
                        sigma=self.sigma.dist,
                        dims=(*model.custom_batch_effect_dims, *self._dims),
                    )
                else:
                    self.mu.create_graph(model, idata, freedom)
                    self.sigma.create_graph(model, idata, freedom)
                    self.offset = pm.Normal(
                        "offset_" + self.name,
                        mu=0,
                        sigma=1,
                        dims=(*model.custom_batch_effect_dims, *self._dims),
                    )
                    self.dist = pm.Deterministic(
                        self.name,
                        self.mu.dist + self.offset * self.sigma.dist,
                        dims=(*model.custom_batch_effect_dims, *self._dims),
                    )
            else:
                if idata is not None:
                    self.approximate_marginal(
                        model,
                        self.dist_name,
                        az.extract(idata, var_names=self.name),
                    )
                self.dist = self.distmap[self.dist_name](
                    self.name, *self.dist_params, shape=self.shape, dims=self.dims
                )

    def approximate_marginal(
        self, model: Any, dist_name: str, samples: xr.DataArray
    ) -> None:
        """
        Approximate marginal distribution parameters from MCMC samples.

        Parameters
        ----------
        model : Any
            PyMC model object
        dist_name : str
            Name of distribution to fit
        samples : xr.DataArray
            MCMC samples to fit distribution to

        Raises
        ------
        ValueError
            If distribution name is not recognized

        Notes
        -----
        Uses scipy.stats to fit distribution parameters to the samples.
        """
        """#TODO At some point, we want to flatten over all dimensions except the covariate dimension."""
        samples_flat = samples.to_numpy().flatten()
        with model:
            if dist_name == "Normal":
                temp = stats.norm.fit(samples_flat)
                self.dist_params = (temp[0], self.freedom * temp[1])
            elif dist_name == "HalfNormal":
                temp = stats.halfnorm.fit(samples_flat)
                self.dist_params = (self.freedom * temp[1],)
            elif dist_name == "LogNormal":
                temp = stats.lognorm.fit(samples_flat)
                self.dist_params = (temp[0], self.freedom * temp[1])
            elif dist_name == "Cauchy":
                temp = stats.cauchy.fit(samples_flat)
                self.dist_params = (temp[0], self.freedom * temp[1])
            elif dist_name == "HalfCauchy":
                temp = stats.halfcauchy.fit(samples_flat)
                self.dist_params = (self.freedom * temp[1],)
            elif dist_name == "Uniform":
                temp = stats.uniform.fit(samples_flat)
                self.dist_params = (temp[0], temp[1])
            elif dist_name == "Gamma":
                temp = stats.gamma.fit(samples_flat)
                self.dist_params = (temp[0], temp[1], self.freedom * temp[2])
            elif dist_name == "InvGamma":
                temp = stats.invgamma.fit(samples_flat)
                self.dist_params = (temp[0], temp[1], self.freedom * temp[2])
            else:
                raise ValueError(f"Unknown distribution name {dist_name}")

    def set_noncentered_random_params(self) -> None:
        """
        Set up non-centered parameterization for random effects.

        Creates default mu and sigma parameters if they don't exist.
        For non-centered parameterization, parameters are expressed as:
        param = mu + sigma * offset, where offset ~ Normal(0,1)
        """
        if not self.mu:
            self.mu = Param.default_sub_mu(self.dims)
        if not self.sigma:
            self.sigma = Param.default_sub_sigma(self.dims)
            self.sigma = Param(
                dims=self.dims,
                dist_name="LogNormal",
                dist_params=(2.0,),
            )

    def set_centered_random_params(self) -> None:
        """
        Set up centered parameterization for random effects.

        Currently delegates to non-centered parameterization setup.
        """
        self.set_noncentered_random_params()

    def set_linear_params(self) -> None:
        """
        Set up parameters for linear relationships.

        Creates default slope and intercept parameters if they don't exist.
        """
        if not self.slope:
            self.slope = Param.default_slope(dims=(*self._dims, "covariates"))
        if not self.intercept:
            self.intercept = Param.default_intercept(self._dims)

    def get_samples(self, data: HBRData) -> Any:
        """
        Generate samples from the parameter distribution.

        Parameters
        ----------
        data : HBRData
            Data object containing covariates and batch effects

        Returns
        -------
        Any
            PyMC distribution or transformed random variable
        """
        if self.linear:
            slope_samples = self.slope.get_samples(data)
            intercept_samples = self.intercept.get_samples(data)
            result = math.sum(slope_samples * data.pm_X, axis=1) + intercept_samples
            return self.apply_mapping(result)

        elif self.random:
            return self.dist[data.pm_batch_effect_indices]
        else:
            return self.dist

    def apply_mapping(self, x: Any) -> Any:
        """
        Apply transformation mapping to parameter values.

        Parameters
        ----------
        x : Any
            Input value to transform

        Returns
        -------
        Any
            Transformed value

        Raises
        ------
        ValueError
            If mapping type is not recognized
        """
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

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert parameter configuration to dictionary format.

        Returns
        -------
        Dict[str, Any]
            Dictionary containing parameter configuration
        """
        param_dict: dict[str, Any] = {
            "name": self.name,
            "dims": self.dims,
            "linear": self.linear,
            "random": self.random,
            "centered": self.centered,
            "has_random_effect": self.has_random_effect,
        }
        if self.linear:
            param_dict["slope"] = self.slope.to_dict()
            param_dict["intercept"] = self.intercept.to_dict()
            param_dict["mapping"] = self.mapping
            param_dict["mapping_params"] = self.mapping_params
        elif self.random:
            param_dict["mu"] = self.mu.to_dict()
            param_dict["sigma"] = self.sigma.to_dict()
        else:
            param_dict["dist_name"] = self.dist_name
            param_dict["dist_params"] = self.dist_params
        return param_dict

    @property
    def _dims(self) -> Tuple[str, ...]:
        """
        Get parameter dimensions as tuple.

        Returns
        -------
        Tuple[str, ...]
            Parameter dimensions as tuple, empty tuple if no dimensions
        """
        return self.dims if self.dims else ()  # type: ignore

    @classmethod
    def from_args(
        cls,
        name: str,
        args: Dict[str, Any],
        dims: Optional[Union[Tuple[str, ...], str]] = None,
    ) -> Param:
        """
        Create parameter from command line arguments.

        Parameters
        ----------
        name : str
            Parameter name
        args : Dict[str, Any]
            Dictionary of arguments
        dims : Optional[Union[Tuple[str, ...], str]], optional
            Parameter dimensions, by default None

        Returns
        -------
        Param
            New parameter instance
        """
        tupdims = dims if dims else ()
        if args.get(f"linear_{name}", False):
            slope = cls.from_args(f"slope_{name}", args, dims=(*tupdims, "covariates"))
            intercept = cls.from_args(f"intercept_{name}", args, dims=dims)
            instance = cls(
                dims=dims,
                linear=True,
                slope=slope,
                intercept=intercept,
                mapping=args.get(f"mapping_{name}", "identity"),
                mapping_params=args.get(f"mapping_params_{name}", (0.0, 1.0)),
            )        
        elif args.get(f"random_{name}", False):
            if args.get(f"centered_{name}", False):
                mu = cls.from_args(f"mu_{name}", args, dims=dims)
                sigma = cls.from_args(f"sigma_{name}", args, dims=dims)
                instance = cls(
                    dims=dims,
                    random=True,
                    centered=True,
                    mu=mu,
                    sigma=sigma,
                )
            else:
                mu = cls.from_args(f"mu_{name}", args, dims=dims)
                sigma = cls.from_args(f"sigma_{name}", args, dims=dims)
                instance = cls(
                    dims=dims,
                    random=True,
                    centered=False,
                    mu=mu,
                    sigma=sigma,
                )
        else:
            (default_dist, default_params) = (
                ("LogNormal", (2.0,))
                if (name.startswith("sigma") or (name == "delta"))
                else ("Normal", (0.0, 10.0))
            )
            instance = cls(
                dims=dims,
                dist_name=args.get(f"{name}_dist_name", default_dist),
                dist_params=args.get(f"{name}_dist_params", default_params),
            )
        instance.set_name(name)
        return instance

    @classmethod
    def from_dict(cls, dict_: Dict[str, Any]) -> Param:
        """
        Create parameter from dictionary configuration.

        Parameters
        ----------
        dict_ : Dict[str, Any]
            Dictionary containing parameter configuration

        Returns
        -------
        Param
            New parameter instance
        """
        if dict_.get("linear", False):
            slope = cls.from_dict(dict_["slope"])
            intercept = cls.from_dict(dict_["intercept"])
            instance = cls(
                linear=True,
                slope=slope,
                intercept=intercept,
                mapping=dict_.get("mapping", "identity"),
                mapping_params=dict_.get("mapping_params", (0.0, 1.0)),
            )
            instance.set_name(dict_["name"])
            return instance
        elif dict_.get("random", False):
            if dict_.get("centered", False):
                mu = cls.from_dict(dict_["mu"])
                sigma = cls.from_dict(dict_["sigma"])
                instance = cls(
                    dims=dict_["dims"],
                    random=True,
                    centered=True,
                    mu=mu,
                    sigma=sigma,
                )
                instance.set_name(dict_["name"])
                return instance
            else:
                mu = cls.from_dict(dict_["mu"])
                sigma = cls.from_dict(dict_["sigma"])
                instance = cls(
                    dims=dict_["dims"],
                    random=True,
                    centered=False,
                    mu=mu,
                    sigma=sigma,
                )
                instance.set_name(dict_["name"])
                return instance
        else:
            name = dict_["name"]
            if name.startswith("sigma") or name == "delta":
                default_dist = "LogNormal"
                default_params: tuple[float, ...] = (2.0,)
            else:
                default_dist = "Normal"
                default_params = (0.0, 10.0)
            instance = cls(
                dims=dict_["dims"],
                dist_name=default_dist,
                dist_params=default_params,
            )
            instance.set_name(name)
            return instance

    @classmethod
    def default_mu(cls) -> Param:
        """
        Create default mean parameter.

        Returns
        -------
        Param
            Default mu parameter with linear relationship
        """
        slope = cls.default_slope("mu")
        intercept = cls.default_intercept("mu")
        return cls(
            linear=True,
            slope=slope,
            intercept=intercept,
        )

    @classmethod
    def default_sigma(cls) -> Param:
        """
        Create default standard deviation parameter.

        Returns
        -------
        Param
            Default sigma parameter with linear relationship and softplus mapping
        """
        slope = cls.default_slope()
        intercept = cls.default_intercept()
        return cls(
            linear=True,
            slope=slope,
            intercept=intercept,
            mapping="softplus",
            mapping_params=(0.0, 10.0),
        )

    @classmethod
    def default_epsilon(cls) -> Param:
        """
        Create default epsilon (error) parameter.

        Returns
        -------
        Param
            Default epsilon parameter with normal distribution
        """
        return cls(
            linear=False,
            random=False,
            dist_name="Normal",
            dist_params=(
                0.0,
                2.0,
            ),
        )

    @classmethod
    def default_delta(cls) -> Param:
        """
        Create default delta parameter.

        Returns
        -------
        Param
            Default delta parameter with normal distribution and softplus mapping
        """
        return cls(
            linear=False,
            random=False,
            dist_name="Normal",
            dist_params=(
                0.0,
                2.0,
            ),
            mapping="softplus",
            mapping_params=(0.0, 3.0, 0.3),
        )

    @classmethod
    def default_slope(
        cls, dims: Union[Tuple[str, ...], str] = ("covariates",)
    ) -> Param:
        """
        Create default slope parameter.

        Parameters
        ----------
        dims : Union[Tuple[str, ...], str], optional
            Parameter dimensions, by default ("covariates",)

        Returns
        -------
        Param
            Default slope parameter with normal distribution
        """
        return cls(
            linear=False,
            random=False,
            dims=dims,
            dist_name="Normal",
            dist_params=(
                0.0,
                10.0,
            ),
        )

    @classmethod
    def default_intercept(
        cls, dims: Optional[Union[Tuple[str, ...], str]] = None
    ) -> Param:
        """
        Create default intercept parameter.

        Parameters
        ----------
        dims : Optional[Union[Tuple[str, ...], str]], optional
            Parameter dimensions, by default None

        Returns
        -------
        Param
            Default intercept parameter with normal distribution
        """
        return cls(
            linear=False,
            random=False,
            dims=dims,
            dist_name="Normal",
            dist_params=(
                0.0,
                10.0,
            ),
        )

    @classmethod
    def default_sub_mu(
        cls, dims: Optional[Union[Tuple[str, ...], str]] = None
    ) -> Param:
        """
        Create default sub-model mean parameter.

        Parameters
        ----------
        dims : Optional[Union[Tuple[str, ...], str]], optional
            Parameter dimensions, by default None

        Returns
        -------
        Param
            Default sub-model mu parameter with normal distribution
        """
        return cls(
            linear=False,
            random=False,
            dims=dims,
            dist_name="Normal",
            dist_params=(
                0.0,
                10.0,
            ),
        )

    @classmethod
    def default_sub_sigma(
        cls, dims: Optional[Union[Tuple[str, ...], str]] = None
    ) -> Param:
        """
        Create default sub-model standard deviation parameter.

        Parameters
        ----------
        dims : Optional[Union[Tuple[str, ...], str]], optional
            Parameter dimensions, by default None

        Returns
        -------
        Param
            Default sub-model sigma parameter with lognormal distribution
        """
        return cls(
            linear=False,
            random=False,
            dims=dims,
            dist_name="LogNormal",
            dist_params=(2.0,),
        )

    def set_name(self, name: str):
        self.name = name
        if self.linear:
            self.slope.set_name(f"slope_{name}")
            self.intercept.set_name(f"intercept_{name}")
        elif self.random:
            self.mu.set_name(f"mu_{name}")
            self.sigma.set_name(f"sigma_{name}")
