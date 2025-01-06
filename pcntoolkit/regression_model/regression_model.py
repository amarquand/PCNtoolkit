"""
Abstract base class for regression models in the PCNToolkit.

This module provides the base class for all regression models, defining the common
interface and shared functionality that all regression implementations must follow.

Notes
-----
All regression model implementations should inherit from this base class and
implement the abstract methods.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from pcntoolkit.regression_model.reg_conf import RegConf


class RegressionModel(ABC):
    """
    Abstract base class for regression models.

    This class defines the interface for all regression models in the toolkit,
    providing common attributes and methods that must be implemented by concrete
    subclasses.

    Parameters
    ----------
    name : str
        Unique identifier for the regression model instance
    reg_conf : RegConf
        Configuration object containing regression model parameters
    is_fitted : bool, optional
        Flag indicating if the model has been fitted to data, by default False
    is_from_dict : bool, optional
        Flag indicating if the model was instantiated from a dictionary, by default False

    Attributes
    ----------
    is_fitted : bool
        Indicates whether the model has been fitted to data
    """

    def __init__(
        self,
        name: str,
        reg_conf: RegConf,
        is_fitted: bool = False,
        is_from_dict: bool = False,
    ):
        self._name: str = name
        self._reg_conf: RegConf = reg_conf
        self.is_fitted: bool = is_fitted
        self._is_from_dict: bool = is_from_dict

    @property
    def name(self) -> str:
        """
        Get the model's name.

        Returns
        -------
        str
            The unique identifier of the model
        """
        return self._name

    @property
    @abstractmethod
    def reg_conf(self) -> RegConf:
        """
        Get the model's configuration.

        Returns
        -------
        RegConf
            The configuration object containing model parameters
        """

    @property
    def is_from_dict(self) -> bool:
        """
        Check if model was instantiated from dictionary.

        Returns
        -------
        bool
            True if model was created from dictionary, False otherwise
        """
        return self._is_from_dict

    def to_dict(self, path: str | None = None) -> dict:
        """
        Convert model instance to dictionary representation.

        Parameters
        ----------
        path : str | None, optional
            Path to save any associated files, by default None

        Returns
        -------
        dict
            Dictionary containing model parameters and configuration
        """
        my_dict: dict[str, str | dict | bool] = {}
        my_dict["name"] = self.name
        my_dict["type"] = self.__class__.__name__
        my_dict["reg_conf"] = self.reg_conf.to_dict(path)
        my_dict["is_fitted"] = self.is_fitted
        my_dict["is_from_dict"] = self.is_from_dict
        return my_dict

    @classmethod
    @abstractmethod
    def from_dict(cls, my_dict: dict, path: str) -> RegressionModel:
        """
        Create model instance from dictionary representation.

        Parameters
        ----------
        dct : dict
            Dictionary containing model parameters and configuration
        path : str
            Path to load any associated files

        Returns
        -------
        RegressionModel
            New instance of the regression model
        
        Raises
        ------
        NotImplementedError
            Must be implemented by concrete subclasses
        """ 

    @classmethod
    @abstractmethod
    def from_args(cls, name: str, args: dict) -> RegressionModel:
        """
        Create model instance from arguments dictionary.

        Parameters
        ----------
        name : str
            Unique identifier for the model instance
        args : dict
            Dictionary of model parameters and configuration

        Returns
        -------
        RegressionModel
            New instance of the regression model
        
        Raises
        ------
        NotImplementedError
            Must be implemented by concrete subclasses
        """

    @property
    def has_random_effect(self) -> bool:
        """
        Check if model includes random effects.

        Returns
        -------
        bool
            True if model includes random effects, False otherwise
        """
        return self.reg_conf.has_random_effect
