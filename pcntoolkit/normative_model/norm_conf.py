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
import warnings
from dataclasses import dataclass, fields
from typing import Any, Callable, Dict, List, Optional

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
    basis_column : int, optional
        Column index for basis expansion, by default 0
    order : int, optional
        Order of polynomial or bspline basis functions, by default 3
    nknots : int, optional
        Number of knots for bspline basis functions, by default 5
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
    log_dir: str = "./logs"
    save_dir: str = "./saves"
    basis_function: str = "linear"
    basis_column: int = 0
    order: int = 3
    nknots: int = 5
    inscaler: str = "none"
    outscaler: str = "none"
    normative_model_name: Optional[str] = None

    def __post_init__(self) -> None:
        """Validate configuration after initialization."""
        configuration_problems = self.detect_configuration_problems()
        if len(configuration_problems) > 0:
            problem_list = "\n".join(
                [f"{i+1}:\t{v}" for i, v in enumerate(configuration_problems)]
            )
            raise ValueError(
                f"The following problems have been detected in the normative model configuration:\n{problem_list}"
            )
        else:
            print("Configuration of normative model is valid.")

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
        print(fields(cls))
        norm_args:dict[str, Any] = {k: v for k, v in args.items() if k in [f.name for f in fields(cls)]}
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

        self.detect_dir_problem(add_problem, "log_dir")
        self.detect_dir_problem(add_problem, "save_dir")
        self.detect_cv_problem(add_problem)
        self.detect_basis_function_problem(add_problem)
        self.detect_scaler_problem(add_problem, "inscaler")
        self.detect_scaler_problem(add_problem, "outscaler")

        return configuration_problems

    def detect_dir_problem(
        self, add_problem: Callable[[str], None], dir_attr_str: str
    ) -> None:
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
            add_problem(
                f"{dir_attr_str} is not a string, but {type(dir_attr).__name__}"
            )
        else:
            if os.path.exists(dir_attr):
                if not os.path.isdir(dir_attr):
                    add_problem(
                        f"{dir_attr_str} is not a directory, but {get_type_of_object(dir_attr)}"
                    )
            else:
                warnings.warn(
                    f"{dir_attr_str} ({dir_attr}) does not exist, creating it for you"
                )
                os.makedirs(dir_attr)

    def detect_cv_problem(self, add_problem: Callable[[str], None]) -> None:
        """Detect problems with cross-validation configuration.

        Validates that:
        - perform_cv is a boolean
        - cv_folds is an integer
        - If perform_cv is True, cv_folds must be >= 2

        Parameters
        ----------
        add_problem : Callable[[str], None]
            Function to add problem description to the list of configuration problems
        """
        performisbool: bool = isinstance(self.perform_cv, bool)
        foldsisint: bool = isinstance(self.cv_folds, int)
        if not performisbool:
            add_problem(
                f"perform_cv is not a boolean, but {type(self.perform_cv).__name__}"
            )
        if not foldsisint:
            add_problem(
                f"cv_folds is not an integer, but {type(self.cv_folds).__name__}"
            )
        if performisbool and foldsisint:
            if self.perform_cv and self.cv_folds < 2:
                add_problem(f"cv_folds must be at least 2, but is {self.cv_folds}")

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
            add_problem(
                f"basis_function_type is not a string, but {type(self.basis_function).__name__}"
            )
        else:
            if self.basis_function not in acceptable_basis_functions:
                add_problem(
                    f"basis_function_type is not one of the possible values: {acceptable_basis_functions}"
                )

            if self.basis_function == "polynomial":
                self.detect_polynomial_basis_expansion_problem(add_problem)

            if self.basis_function == "bspline":
                self.detect_bspline_basis_expansion_problem(add_problem)

    def detect_bspline_basis_expansion_problem(
        self, add_problem: Callable[[str], None]
    ) -> None:
        """Detect problems with B-spline basis expansion configuration.

        Validates that:
        - nknots is an integer >= 2
        - order is an integer >= 1
        - order is less than nknots

        Parameters
        ----------
        add_problem : Callable[[str], None]
            Function to add problem description to the list of configuration problems
        """
        nknotsisint = isinstance(self.nknots, int)
        orderisint = isinstance(self.order, int)
        if not nknotsisint:
            add_problem(f"nknots is not an integer, but {type(self.nknots).__name__}")
        else:
            if self.nknots < 2:
                add_problem(f"nknots must be at least 2, but is {self.nknots}")

        if not orderisint:
            add_problem(f"order is not an integer, but {type(self.order).__name__}")

        else:
            if self.order < 1:
                add_problem(f"order must be at least 1, but is {self.order}")
            if nknotsisint:
                if self.order > self.nknots:
                    add_problem(
                        f"order must be smaller than nknots, but order is {self.order} and nknots is {self.nknots}"
                    )

    def detect_polynomial_basis_expansion_problem(
        self, add_problem: Callable[[str], None]
    ) -> None:
        """Detect problems with polynomial basis expansion configuration.

        Validates that:
        - order is an integer >= 1

        Parameters
        ----------
        add_problem : Callable[[str], None]
            Function to add problem description to the list of configuration problems
        """
        orderisint = isinstance(self.order, int)
        if not orderisint:
            add_problem(f"order is not an integer, but {type(self.order).__name__}")
        else:
            if self.order < 1:
                add_problem(f"order must be at least 1, but is {self.order}")

    def detect_scaler_problem(
        self, add_problem: Callable[[str], None], scaler_attr_str: str
    ) -> None:
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
            add_problem(
                f"{scaler_attr_str} is not a string, but {type(scaler_attr).__name__}"
            )
        else:
            if scaler_attr not in acceptable_scalers:
                add_problem(
                    f"{scaler_attr_str} is not one of the possible values: {acceptable_scalers}"
                )

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

    def set_log_dir(self, path: str) -> None:
        """Set the log directory path.

        Since this is a frozen dataclass, uses object.__setattr__ to modify
        the log_dir attribute.

        Parameters
        ----------
        path : str
            New path for logging output
        """
        object.__setattr__(self, "log_dir", path)

    def copy(self) -> "NormConf":
        """Create a deep copy of the configuration.

        Returns
        -------
        NormConf
            A new NormConf instance with the same configuration values
        """
        return NormConf.from_args(self.to_dict())
