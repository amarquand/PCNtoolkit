"""
Module: blr_conf

This module defines the configuration class for Bayesian Linear Regression (BLR) models.
The `BLRConf` class is a dataclass that encapsulates configuration parameters for setting
up and running BLR models. It provides a convenient method to initialize its instances
from a dictionary of arguments.

Classes
-------
BLRConf
    A dataclass that holds configuration parameters for a BLR model.

Methods
-------
BLRConf.from_args(args: dict[str, Any]) -> BLRConf
    Class method to create an instance of `BLRConf` from a dictionary of arguments.

Examples
--------
>>> args = {'param1': value1, 'param2': value2}
>>> config = BLRConf.from_args(args)
>>> print(config)
BLRConf(param1=value1, param2=value2)
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Any

from pcntoolkit.regression_model.reg_conf import RegConf

# Default configuration values
N_ITER = 300
TOL = 1e-5
ARD = False
OPTIMIZER = "l-bfgs-b"
L_BFGS_B_L = 0.1
L_BFGS_B_EPSILON = 0.1
L_BFGS_B_NORM = "l2"
INTERCEPT = False
RANDOM_INTERCEPT = False
HETEROSKEDASTIC = False
INTERCEPT_VAR = False
RANDOM_INTERCEPT_VAR = False


@dataclass(frozen=True)
class BLRConf(RegConf):
    """
    Class: BLRConf

    A dataclass for configuring Bayesian Linear Regression (BLR) models. This class encapsulates
    the parameters required to set up and execute a BLR model, providing a structured way to
    manage configuration data.

    Parameters
    ----------
    n_iter: int = N_ITER
        The number of iterations for the optimization algorithm.
    tol: float = TOL
        The tolerance for the optimization algorithm.
    ard: bool = ARD
        Whether to use Automatic Relevance Determination (ARD).
    optimizer: str = OPTIMIZER
        The optimization algorithm to use. Options are: "l-bfgs-b", "cg", "powell", "nelder-mead".
    l_bfgs_b_l: float = L_BFGS_B_L
        The L2 regularization parameter for the "l-bfgs-b" optimizer.
    l_bfgs_b_epsilon: float = L_BFGS_B_EPSILON
        The epsilon parameter for the "l-bfgs-b" optimizer.
    l_bfgs_b_norm: str = L_BFGS_B_NORM
        The norm to use for the "l-bfgs-b" optimizer. Options are: "l1", "l2".
    intercept: bool = INTERCEPT
        Whether to include an intercept in the model.
    random_intercept: bool = RANDOM_INTERCEPT
        Whether to include a random intercept in the model.
    heteroskedastic: bool = HETEROSKEDASTIC
        Whether to model heteroskedasticity in the data.
    intercept_var: bool = INTERCEPT_VAR
        Whether the variance has an intercept (a fixed effect).
    random_intercept_var: bool = RANDOM_INTERCEPT_VAR
        Whether the variance has a random intercept for each group.

    Methods
    -------
    from_args(args: dict[str, Any]) -> BLRConf
        Class method to create an instance of `BLRConf` from a dictionary of arguments.

    Examples
    --------
    >>> args = {'param1': value1, 'param2': value2}
    >>> config = BLRConf.from_args(args)
    >>> print(config)
    BLRConf(param1=value1, param2=value2)
    """
    # some configuration parameters
    n_iter: int = N_ITER
    tol: float = TOL

    # use ard
    ard: bool = ARD

    # optimization parameters
    optimizer: str = OPTIMIZER  # options: "l-bfgs-b", "cg", "powell", " nelder-mead"
    l_bfgs_b_l: float = L_BFGS_B_L
    l_bfgs_b_epsilon: float = L_BFGS_B_EPSILON
    l_bfgs_b_norm: str = L_BFGS_B_NORM

    # Design matrix configuration
    intercept: bool = INTERCEPT
    random_intercept: bool = RANDOM_INTERCEPT
    heteroskedastic: bool = HETEROSKEDASTIC
    intercept_var: bool = INTERCEPT_VAR
    random_intercept_var: bool = RANDOM_INTERCEPT_VAR

    # TODO implement warp
    # warp: WarpBase = None
    # warp_reparam: bool = Falses

    def detect_configuration_problems(self) -> list[str]:
        """
        Detects problems in the configuration and returns them as a list of strings.
        The super class will throw an exception if the configuration is invalid, and show the problems.
        """

        configuration_problems = []

        def add_problem(problem: str) -> None:
            nonlocal configuration_problems
            configuration_problems.append(f"{problem}")

        if self.n_iter < 1:
            add_problem("n_iter must be greater than 0.")

        if self.tol <= 0:
            add_problem("tol must be greater than 0.")

        if self.optimizer not in ["l-bfgs-b", "cg", "powell", "nelder-mead"]:
            add_problem(f"Optimizer {self.optimizer} not recognized.")

        return configuration_problems

    @classmethod
    def from_args(cls, args:dict[str, Any]) -> "BLRConf":
        args_filt:dict[str, Any] = {k: v for k, v in args.items() if k in fields(cls)}

        return cls(
            n_iter=args_filt.get("n_iter", N_ITER),
            tol=args_filt.get("tol", TOL),
            ard=args_filt.get("ard", ARD),
            optimizer=args_filt.get("optimizer", OPTIMIZER),
            l_bfgs_b_l=args_filt.get("l_bfgs_b_l", L_BFGS_B_L),
            l_bfgs_b_epsilon=args_filt.get("l_bfgs_b_epsilon", L_BFGS_B_EPSILON),
            l_bfgs_b_norm=args_filt.get("l_bfgs_b_norm", L_BFGS_B_NORM),
            intercept=args_filt.get("intercept", INTERCEPT),
            random_intercept=args_filt.get("random_intercept", RANDOM_INTERCEPT),
            heteroskedastic=args_filt.get("heteroskedastic", HETEROSKEDASTIC),
            intercept_var=args_filt.get("intercept_var", INTERCEPT_VAR),
            random_intercept_var=args_filt.get(
                "random_intercept_var", RANDOM_INTERCEPT_VAR
            ),
        )

    @classmethod
    def from_dict(cls, dct:dict[str, Any]) -> "BLRConf":
        return cls(
            n_iter=dct["n_iter"],
            tol=dct["tol"],
            ard=dct["ard"],
            optimizer=dct["optimizer"],
            l_bfgs_b_l=dct["l_bfgs_b_l"],
            l_bfgs_b_epsilon=dct["l_bfgs_b_epsilon"],
            l_bfgs_b_norm=dct["l_bfgs_b_norm"],
            intercept=dct["intercept"],
            random_intercept=dct["random_intercept"],
            heteroskedastic=dct["heteroskedastic"],
            intercept_var=dct["intercept_var"],
            random_intercept_var=dct["random_intercept_var"],
        )
    def to_dict(self, path:str|None="") -> dict[str, Any]:
        return {
            "n_iter": self.n_iter,
            "tol": self.tol,
            "ard": self.ard,
            "optimizer": self.optimizer,
            "l_bfgs_b_l": self.l_bfgs_b_l,
            "l_bfgs_b_epsilon": self.l_bfgs_b_epsilon,
            "l_bfgs_b_norm": self.l_bfgs_b_norm,
            "intercept": self.intercept,
            "random_intercept": self.random_intercept,
            "heteroskedastic": self.heteroskedastic,
            "intercept_var": self.intercept_var,
            "random_intercept_var": self.random_intercept_var,
        }

    @property
    def has_random_effect(self) -> bool:
        return self.random_intercept or self.random_intercept_var