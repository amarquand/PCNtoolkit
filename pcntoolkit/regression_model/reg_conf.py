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
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List

from pcntoolkit.util.output import Errors, Messages, Output


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
    """

    def __post_init__(self) -> None:
        """
        Validates the configuration after initialization.

        This method is automatically called after object initialization to verify
        that all configuration parameters are valid. It uses detect_configuration_problems()
        to identify any issues with the configuration.
        """
        configuration_problems = self.detect_configuration_problems()
        if len(configuration_problems) > 0:
            problem_list = "\n".join(
                [f"{i+1}:\t{v}" for i, v in enumerate(configuration_problems)]
            )
            Output.error(Errors.REGRESSION_MODEL_CONFIGURATION_PROBLEMS, problems=problem_list)
        else:
            Output.print(Messages.REGRESSION_MODEL_CONFIGURATION_VALID)

    @property
    @abstractmethod
    def has_random_effect(self) -> bool:
        """
        Indicates whether the regression model includes random effects.
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
        """

    @abstractmethod
    def to_dict(self, path: str | None = None) -> Dict[str, Any]:
        """
        Serializes the configuration to a dictionary.

        Parameters
        ----------
        path : str | None, optional
            Optional file path for storing large objects. 

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
