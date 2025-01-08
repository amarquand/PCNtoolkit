"""Configuration class for normative modeling.

This module provides the NormConf dataclass that handles configuration settings
for normative modeling pipelines. It manages settings for:
- Model persistence (saving/loading)
- Directory management
- Basis function configuration
- Data scaling
- Cross-validation parameters

The class provides validation of all configuration parameters and handles
directory creation when needed.
"""

import os
from dataclasses import dataclass, field, fields
from typing import Any, Callable, Dict, List, Optional

from pcntoolkit.util.output import Errors, Messages, Output, Warnings
from pcntoolkit.util.utils import get_type_of_object


@dataclass(frozen=True)
class NormConf:
    """Configuration container for normative modeling.

    Parameters
    ----------
    savemodel : bool, optional
        Whether to save the trained model, by default False
    saveresults : bool, optional
        Whether to save prediction results, by default False
    log_dir : str, optional
        Directory for logging output, by default "./logs"
    save_dir : str, optional
        Directory for saving models and results, by default "./saves"
    basis_function : str, optional
        Type of basis function to use ('linear', 'polynomial', 'bspline', 'none'),
        by default "linear"
    basis_function_kwargs : dict, optional
        Keyword arguments for basis function, by default {}
        For polynomial basis: order
        For bspline basis: order, nknots, left_expand, right_expand, knot_method
    inscaler : str, optional
        Input data scaling method ('none', 'standardize', 'minmax'), by default "none"
    outscaler : str, optional
        Output data scaling method ('none', 'standardize', 'minmax'), by default "none"
    perform_cv : bool, optional
        Whether to perform cross-validation, by default False
    cv_folds : int, optional
        Number of cross-validation folds, by default 0
    normative_model_name : Optional[str], optional
        Name identifier for the normative model, by default None
    """

    savemodel: bool = False
    saveresults: bool = False
    save_dir: str = "./saves"
    basis_function: str = "linear"
    basis_function_kwargs: dict = field(default_factory=dict)
    inscaler: str = "none"
    outscaler: str = "none"
    normative_model_name: Optional[str] = None

    def __post_init__(self) -> None:
        """Validate configuration after initialization."""
        configuration_problems = self.detect_configuration_problems()
        if len(configuration_problems) > 0:
            problem_list = "\n".join([f"{i+1}:\t{v}" for i, v in enumerate(configuration_problems)])
            Output.error(Errors.NORMATIVE_MODEL_CONFIGURATION_PROBLEMS, problems=problem_list)
        else:
            Output.print(Messages.NORMATIVE_MODEL_CONFIGURATION_VALID)

    @classmethod
    def from_args(cls, args: Dict[str, Any]) -> "NormConf":
        """Create configuration from command line arguments.

        Parameters
        ----------
        args : Dict[str, Any]
            Dictionary of argument names and values

        Returns
        -------
        NormConf
            New configuration instance
        """
        norm_args: dict[str, Any] = {k: v for k, v in args.items() if k in [f.name for f in fields(cls)]}
        return cls(**norm_args)

    @classmethod
    def from_dict(cls, dct: Dict[str, Any]) -> "NormConf":
        """Create configuration from dictionary.

        Parameters
        ----------
        dct : Dict[str, Any]
            Dictionary of configuration parameters

        Returns
        -------
        NormConf
            New configuration instance
        """
        return cls.from_args(dct)

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary.

        Returns
        -------
        Dict[str, Any]
            Dictionary representation of configuration
        """
        return self.__dict__

    def detect_configuration_problems(self) -> List[str]:
        """Detect any problems in the current configuration.

        Returns
        -------
        List[str]
            List of detected configuration problems
        """
        configuration_problems: List[str] = []

        def add_problem(problem: str) -> None:
            nonlocal configuration_problems
            configuration_problems.append(f"{problem}")

        self.detect_dir_problem(add_problem, "save_dir")
        self.detect_basis_function_problem(add_problem)
        self.detect_scaler_problem(add_problem, "inscaler")
        self.detect_scaler_problem(add_problem, "outscaler")

        return configuration_problems

    def detect_dir_problem(self, add_problem: Callable[[str], None], dir_attr_str: str) -> None:
        """Detect problems with directory configuration.

        Parameters
        ----------
        add_problem : Callable[[str], None]
            Function to add problem description
        dir_attr_str : str
            Name of directory attribute to check
        """
        dir_attr = getattr(self, dir_attr_str)
        if not isinstance(dir_attr, str):
            add_problem(f"{dir_attr_str} is not a string, but {type(dir_attr).__name__}")
        else:
            if os.path.exists(dir_attr):
                if not os.path.isdir(dir_attr):
                    add_problem(f"{dir_attr_str} is not a directory, but {get_type_of_object(dir_attr)}")
            else:
                Output.warning(
                    Warnings.DIR_DOES_NOT_EXIST,
                    dir_attr_str=dir_attr_str,
                    dir_attr=dir_attr,
                )
                os.makedirs(dir_attr)

    def detect_basis_function_problem(self, add_problem: Callable[[str], None]) -> None:
        """Detect problems with basis function configuration.

        Validates that:
        - basis_function is a string and one of: 'linear', 'polynomial', 'bspline', 'none'
        - For polynomial basis: validates order parameters
        - For bspline basis: validates order and nknots parameters

        Parameters
        ----------
        add_problem : Callable[[str], None]
            Function to add problem description to the list of configuration problems
        """
        acceptable_basis_functions = ["linear", "polynomial", "bspline", "none"]
        if not isinstance(self.basis_function, str):
            add_problem(f"basis_function_type is not a string, but {type(self.basis_function).__name__}")
        else:
            if self.basis_function not in acceptable_basis_functions:
                add_problem(f"basis_function_type is not one of the possible values: {acceptable_basis_functions}")

    def detect_scaler_problem(self, add_problem: Callable[[str], None], scaler_attr_str: str) -> None:
        """Detect problems with data scaling configuration.

                Validates that the specified scaler is one of:
                - 'none'
                - 'standardize'
                - 'minmax'
        ??
                Parameters
                ----------
                add_problem : Callable[[str], None]
                    Function to add problem description to the list of configuration problems
                scaler_attr_str : str
                    Name of the scaler attribute to check ('inscaler' or 'outscaler')
        """
        acceptable_scalers = ["none", "standardize", "minmax"]
        scaler_attr = getattr(self, scaler_attr_str)
        if not isinstance(scaler_attr, str):
            add_problem(f"{scaler_attr_str} is not a string, but {type(scaler_attr).__name__}")
        else:
            if scaler_attr not in acceptable_scalers:
                add_problem(f"{scaler_attr_str} is not one of the possible values: {acceptable_scalers}")

    def set_save_dir(self, path: str) -> None:
        """Set the save directory path.

        Since this is a frozen dataclass, uses object.__setattr__ to modify
        the save_dir attribute.

        Parameters
        ----------
        path : str
            New path for saving models and results
        """
        object.__setattr__(self, "save_dir", path)

    def copy(self) -> "NormConf":
        """Create a deep copy of the configuration.

        Returns
        -------
        NormConf
            A new NormConf instance with the same configuration values
        """
        return NormConf.from_args(self.to_dict())
