"""
Configuration module for Hierarchical Bayesian Regression (HBR) models.

This module provides the HBRConf class for configuring Hierarchical Bayesian Regression
models in PCNToolkit. It defines parameters for MCMC sampling, model specification,
and prior distributions, ensuring consistent configuration across HBR model instances.

The module implements a comprehensive configuration system that handles:
- MCMC sampling parameters (draws, chains, tuning)
- Prior distribution specifications
- Model likelihood selection
- Parallel computation settings
- Initialization strategies

Classes
-------
HBRConf
    Configuration class for HBR models, inheriting from RegConf. Handles all
    parameters needed to specify and fit an HBR model.

Notes
-----
The configuration system supports multiple likelihood functions:
- Normal: Standard normal likelihood
- SHASHb: Sinh-arcsinh distribution (basic)
- SHASHo: Sinh-arcsinh distribution (original)
- SHASHo2: Sinh-arcsinh distribution (original v2)

The module supports two NUTS sampler implementations:
- pymc: Default PyMC implementation
- nutpie: Alternative NutPie implementation

Example
-------
>>> conf = HBRConf(draws=2000, chains=4, likelihood="Normal", cores=2)
>>> conf.to_dict()
{'draws': 2000, 'chains': 4, 'likelihood': 'Normal', 'cores': 2, ...}

See Also
--------
pcntoolkit.regression_model.reg_conf : Base configuration module
pcntoolkit.regression_model.hbr.param : Prior parameter specifications
pcntoolkit.regression_model.hbr.hbr : HBR model implementation
"""

from dataclasses import dataclass, field
from typing import Any, ClassVar, Dict, List, Optional

from pcntoolkit.regression_model.hbr.likelihood import Likelihood, NormalLikelihood
from pcntoolkit.regression_model.hbr.prior import (
    get_default_mu,
    get_default_sigma,
)
from pcntoolkit.regression_model.reg_conf import RegConf

# Default configuration values
DRAWS = 1000
TUNE = 1000
CHAINS = 2
PYMC_CORES = 1
LIKELIHOOD = "Normal"
NUTS_SAMPLER = "pymc"
INIT = "jitter+adapt_diag_grad"


@dataclass(frozen=True)
class HBRConf(RegConf):
    """
    Configuration class for Hierarchical Bayesian Regression (HBR) models.

    This class defines the configuration parameters for HBR models, including sampling
    settings, model specification, and prior distributions. It inherits from RegConf
    and implements configuration validation specific to HBR models.

    Parameters
    ----------
    draws : int, optional
        Number of posterior samples to draw, by default 1000
    tune : int, optional
        Number of tuning steps for the MCMC sampler, by default 1000
    chains : int, optional
        Number of parallel MCMC chains to run, by default 2
    cores : int, optional
        Number of CPU cores to use for parallel sampling, by default 1
    nuts_sampler : str, optional
        NUTS sampler implementation to use ('pymc' or 'nutpie'), by default 'pymc'
    init : str, optional
        Initialization strategy for MCMC chains, by default 'jitter+adapt_diag'
    likelihood : str, optional
        Likelihood function to use ('Normal', 'SHASHb', 'SHASHo', or 'SHASHo2'),
        by default 'Normal'
    mu : Param, optional
        Prior parameters for the mean (μ), defaults to Param.default_mu()
    sigma : Param, optional
        Prior parameters for the standard deviation (σ), defaults to Param.default_sigma()
    epsilon : Param, optional
        Prior parameters for epsilon (ε), defaults to Param.default_epsilon()
    delta : Param, optional
        Prior parameters for delta (δ), defaults to Param.default_delta()

    Methods
    -------
    detect_configuration_problems()
        Validates the configuration parameters and returns a list of any problems

    Examples
    --------
    >>> conf = HBRConf(draws=2000, chains=4, likelihood="Normal")

    Notes
    -----
    - Uses the dataclass decorator with frozen=True for immutability
    - Implements comprehensive validation of all configuration parameters
    - Supports multiple likelihood functions for different modeling scenarios
    """

    # sampling config
    draws: int = DRAWS
    tune: int = TUNE
    chains: int = CHAINS
    pymc_cores: int = PYMC_CORES

    nuts_sampler: str = NUTS_SAMPLER
    init: str = INIT

    # model config
    likelihood: Likelihood = field(default_factory=lambda: NormalLikelihood(mu=get_default_mu(), sigma=get_default_sigma()))

    # Add class variable for dataclass fields
    __dataclass_fields__: ClassVar[Dict[str, Any]]

    def __post_init__(self) -> None:
        self.detect_configuration_problems()

    def detect_configuration_problems(self) -> List[str]:
        """
        Detects problems in the configuration and returns them as a list of strings.
        """
        configuration_problems: List[str] = []

        def add_problem(problem: str) -> None:
            nonlocal configuration_problems
            configuration_problems.append(f"{problem}")

        # Check if nuts_sampler is valid
        if self.nuts_sampler not in ["pymc", "nutpie"]:
            add_problem(
                f"""Nuts sampler '{self.nuts_sampler}' is not supported. Please specify a valid nuts sampler. Available
                options are 'pymc' and 'nutpie'."""
            )

        return configuration_problems

    @classmethod
    def from_args(cls, args: Dict[str, Any]) -> "HBRConf":
        """
        Creates a configuration from command line arguments parsed by argparse.
        """
        # Filter out the arguments that are not relevant for this configuration
        args_filt = {k: v for k, v in args.items() if k in cls.__dataclass_fields__}
        likelihood = Likelihood.from_dict(args_filt.pop("likelihood"))
        self = cls(**args_filt, likelihood=likelihood)
        return self

    @classmethod
    def from_dict(cls, dct: Dict[str, Any]) -> "HBRConf":
        """
        Creates a configuration from a dictionary.
        """
        # Filter out the arguments that are not relevant for this configuration
        args_filt = {k: v for k, v in dct.items() if k in cls.__dataclass_fields__}
        likelihood = Likelihood.from_dict(args_filt.pop("likelihood"))
        self = cls(**args_filt, likelihood=likelihood)
        return self

    def to_dict(self, path: Optional[str] = None) -> Dict[str, Any]:
        """Converts the configuration to a dictionary.
        Parameters
        ----------
        path : str | None, optional
            Optional file path for configurations that include file references.
            Used to resolve relative paths to absolute paths.

        Returns:
        ----------
            Dict[str, Any]: Dictionary containing the configuration.
        """
        conf_dict = {
            "draws": self.draws,
            "tune": self.tune,
            "pymc_cores": self.pymc_cores,
            "likelihood": self.likelihood.to_dict(),
            "nuts_sampler": self.nuts_sampler,
            "init": self.init,
            "chains": self.chains,
        }
        return conf_dict

    @property
    def has_random_effect(self) -> bool:
        return self.likelihood.has_random_effect
