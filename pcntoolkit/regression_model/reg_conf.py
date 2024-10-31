"""
Configuration base module for regression models in PCNToolkit.

This module provides the abstract base class RegConf that defines the interface for
all regression model configurations in the PCNToolkit package. It establishes a
standardized way to handle model parameters, validation, and serialization across
different regression implementations.

The module implements a robust configuration management system that ensures:
- Consistent parameter validation across all regression models
- Standardized serialization methods for saving/loading configurations
- Clear error reporting for invalid configurations
- Type safety through static typing
- Extensible design for adding new regression models

Classes
-------
RegConf
    Abstract base class for regression model configurations. Provides the interface
    that all concrete configuration classes must implement.

Notes
-----
When implementing a new regression model configuration:
1. Create a new class that inherits from RegConf
2. Implement all abstract methods and properties
3. Define model-specific parameters in __init__
4. Add validation logic in detect_configuration_problems()
5. Implement serialization in to_dict() and from_dict()

Example
-------
>>> class MyModelConf(RegConf):
...     def __init__(self, learning_rate: float = 0.01):
...         self.learning_rate = learning_rate
...         self.__post_init__()
...
...     @property
...     def has_random_effect(self) -> bool:
...         return False
...
...     def detect_configuration_problems(self) -> List[str]:
...         problems = []
...         if self.learning_rate <= 0:
...             problems.append("learning_rate must be positive")
...         return problems

See Also
--------
pcntoolkit.regression_model.blr.blr_conf : Bayesian Linear Regression configuration
pcntoolkit.regression_model.gpr.gpr_conf : Gaussian Process Regression configuration
pcntoolkit.regression_model.hbr.hbr_conf : Hierarchical Bayesian Regression configuration
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List


class RegConf(ABC):
    """
    Abstract base class for regression model configurations.

    This class defines the interface for storing and validating configuration parameters
    of regression models. It handles only configuration parameters and not learned
    coefficients, which are stored in the model instances themselves.

    The class provides methods for serialization to/from dictionaries and validation
    of configuration parameters.

    Parameters
    ----------
    Specific parameters are defined in concrete subclasses.

    Attributes
    ----------
    has_random_effect : bool
        Indicates whether the regression model includes random effects.

    Methods
    -------
    __post_init__()
        Validates the configuration after initialization.
    detect_configuration_problems()
        Checks for any issues in the configuration parameters.
    from_args(args)
        Creates a configuration instance from command-line arguments.
    from_dict(dct)
        Creates a configuration instance from a dictionary.
    to_dict(path=None)
        Serializes the configuration to a dictionary.

    Raises
    ------
    ValueError
        If any configuration problems are detected during initialization.

    Examples
    --------
    Subclasses should implement this abstract base class like so:

    >>> class MyModelConf(RegConf):
    ...     def __init__(self, param1: float, param2: str):
    ...         self.param1 = param1
    ...         self.param2 = param2
    ...
    ...     @property
    ...     def has_random_effect(self) -> bool:
    ...         return False
    ...
    ...     def detect_configuration_problems(self) -> List[str]:
    ...         problems = []
    ...         if self.param1 < 0:
    ...             problems.append("param1 must be non-negative")
    ...         return problems

    Notes
    -----
    - All configuration parameters should be immutable after initialization
    - Validation is automatically performed via __post_init__
    - Subclasses must implement all abstract methods
    - The class follows the configuration validation pattern
    """

    def __post_init__(self) -> None:
        """
        Validates the configuration after initialization.

        This method is automatically called after object initialization to verify
        that all configuration parameters are valid. It uses detect_configuration_problems()
        to identify any issues with the configuration.

        Returns
        -------
        None

        Raises
        ------
        ValueError
            If any configuration problems are detected, with a detailed list of the problems.

        Notes
        -----
        - Prints a confirmation message if configuration is valid
        - The error message includes numbered list of all detected problems
        """
        configuration_problems = self.detect_configuration_problems()
        if len(configuration_problems) > 0:
            problem_list = "\n".join(
                [f"{i+1}:\t{v}" for i, v in enumerate(configuration_problems)]
            )
            raise ValueError(
                f"The following problems have been detected in the regression model configuration:\n{problem_list}"
            )
        else:
            print("Configuration of regression model is valid.")

    @property
    @abstractmethod
    def has_random_effect(self) -> bool:
        """
        Indicates whether the regression model includes random effects.

        Returns
        -------
        bool
            True if the model includes random effects, False otherwise.

        Notes
        -----
        - This is an abstract property that must be implemented by subclasses
        - Default implementation returns False
        """

    @abstractmethod
    def detect_configuration_problems(self) -> List[str]:
        """
        Checks for any issues in the configuration parameters.

        This method should perform validation checks on all configuration parameters
        and return a list of string descriptions of any problems found.

        Returns
        -------
        List[str]
            A list of strings describing any configuration problems.
            An empty list indicates no problems were found.

        Notes
        -----
        - Subclasses must implement this method
        - Each problem should be described in a human-readable format
        - Return an empty list if no problems are detected
        """

    @classmethod
    @abstractmethod
    def from_args(cls, args: dict) -> RegConf:
        """
        Creates a configuration instance from command-line arguments.

        Parameters
        ----------
        args : dict
            Dictionary of command-line arguments, typically parsed from argparse.

        Returns
        -------
        RegConf
            A new instance of the configuration class.

        Notes
        -----
        - Subclasses must implement this method
        - Should handle type conversion from string arguments
        - Should validate all required arguments are present
        """

    @classmethod
    @abstractmethod
    def from_dict(cls, dct: dict) -> RegConf:
        """
        Creates a configuration instance from a dictionary.

        Parameters
        ----------
        dct : dict
            Dictionary containing configuration parameters.

        Returns
        -------
        RegConf
            A new instance of the configuration class.

        Notes
        -----
        - Subclasses must implement this method
        - Should validate all required keys are present
        - Useful for loading configurations from JSON/YAML files
        """

    @abstractmethod
    def to_dict(self, path: str | None = None) -> Dict[str, Any]:
        """
        Serializes the configuration to a dictionary.

        Parameters
        ----------
        path : str | None, optional
            Optional file path for configurations that include file references.
            Used to resolve relative paths to absolute paths.

        Returns
        -------
        Dict[str, Any]
            Dictionary representation of the configuration.

        Notes
        -----
        - Subclasses must implement this method
        - Should handle conversion of all parameters to JSON-serializable types
        - If path is provided, should resolve any relative paths to absolute paths
        """
