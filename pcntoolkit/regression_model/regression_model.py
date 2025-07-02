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

import importlib.metadata
from abc import ABC, abstractmethod

import numpy as np
import xarray as xr

from pcntoolkit.util.output import Messages, Output


class RegressionModel(ABC):
    """
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
        is_fitted: bool = False,
        is_from_dict: bool = False,
    ):
        self._name: str = name
        self.is_fitted: bool = is_fitted
        self.is_from_dict: bool = is_from_dict

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
    
    @name.setter
    def name(self, name: str) -> None:
        """
        Set the model's name.
        """
        self._name = name
    
    @abstractmethod
    def fit(self, X: xr.DataArray, be: xr.DataArray, be_maps: dict[str, dict[str, int]], Y: xr.DataArray) -> None:
        """
        Fit the model to the data.

        Parameters
        ----------
        X: xr.DataArray containing covariates
        be: xr.DataArray containing batch effects
        be_maps: dictionary of dictionaries mapping batch effect to indices
        Y: xr.DataArray containing covariates

        Returns
        -------
        Nothing
        """
        pass

    @abstractmethod
    def forward(self, X: xr.DataArray, be: xr.DataArray, be_maps: dict[str, dict[str, int]], Y: xr.DataArray) -> xr.DataArray:
        """Compute Z-scores for provided Y values 

        Parameters
        ----------
        X: xr.DataArray containing covariates
        be: xr.DataArray containing batch effects
        be_maps: dictionary of dictionaries mapping batch effect to indices
        Y: xr.DataArray containing covariates

        Returns
        -------
        xr.DataArray
            Data with Z-scores derived from Y values
        """
        pass

    @abstractmethod
    def backward(self, X: xr.DataArray, be: xr.DataArray, be_maps: dict[str, dict[str, int]], Z: xr.DataArray) -> xr.DataArray:
        """Compute points in feature space for given z-scores 

        Parameters
        ----------
        X: xr.DataArray containing covariates
        be: xr.DataArray containing batch effects
        be_maps: dictionary of dictionaries mapping batch effect to indices
        Y: xr.DataArray containing covariates

        Returns
        -------
        xr.DataArray
            Data with Y values derived from Z-scores
        """

    @abstractmethod
    def elemwise_logp(self, X: xr.DataArray, be: xr.DataArray, be_maps: dict[str, dict[str, int]], Y: xr.DataArray) -> xr.DataArray:
        """Compute the log-probability of the data under the model.
        """
        pass

    @abstractmethod
    def transfer(self, X: xr.DataArray, be: xr.DataArray, be_maps: dict[str, dict[str, int]], Y: xr.DataArray) -> RegressionModel:
        """Transfer the model to a new dataset.

        Parameters
        ----------
        X: xr.DataArray containing covariates
        be: xr.DataArray containing batch effects
        be_maps: dictionary of dictionaries mapping batch effect to indices
        Y: xr.DataArray containing covariates

        Returns
        -------
        RegressionModel
            New instance of the regression model, transfered to the new dataset
        """
        pass

    def model_specific_evaluation(self, path: str) -> None:
        """
        Save model-specific evaluation metrics.
        """
        pass
    
    @property
    def regmodel_dict(self) -> dict:
        my_dict: dict[str, str | dict | bool] = {}

        my_dict["name"] = self.name
        my_dict["type"] = self.__class__.__name__
        my_dict["is_fitted"] = self.is_fitted
        my_dict["is_from_dict"] = self.is_from_dict
        my_dict["ptk_version"] = importlib.metadata.version("pcntoolkit")
        return my_dict
        
    def compute_yhat(self, data, n_samples, responsevar, X, be, be_maps):
        samples = np.zeros((data.X.shape[0], n_samples))
        Output.print(Messages.COMPUTING_YHAT_MODEL, model_name=responsevar)
        random_samples = np.random.randn(data.X.shape[0], n_samples)
        Z = xr.DataArray(np.random.randn(data.X.shape[0]), dims=("observations",))
        for s in range(n_samples):
            Z.values = random_samples[:,s]
            samples[:, s] = self.backward(X, be, be_maps, Z).values
        return samples.mean(axis=1)
        
    @abstractmethod
    def to_dict(self, path: str | None = None) -> dict:
        """
        Convert model instance to dictionary representation.

        Used for saving models to disk. 

        Parameters
        ----------
        path : str | None, optional
            Path to save any associated files, by default None

        Returns
        -------
        dict
            Dictionary containing model parameters and configuration
        """
        pass

    @classmethod
    @abstractmethod
    def from_dict(cls, my_dict: dict, path: str) -> RegressionModel:
        """
        Create model instance from dictionary representation.

        Used for loading models from disk. 
        
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

        Used for instantiating models from the command line. 

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
    @abstractmethod
    def has_batch_effect(self) -> bool:
        """
        Check if model includes batch effects.

        Returns
        -------
        bool
            True if model includes batch effects, False otherwise
        """
        pass

# class TransferableRegressionModel(RegressionModel):
#     """
#     Abstract base class for transferable regression models.
#     """
    
#     @abstractmethod
#     def transfer(self, X: xr.DataArray, be: xr.DataArray, be_maps: dict[str, dict[str, int]], Y: xr.DataArray) -> TransferableRegressionModel:
#         """
#         Transfer the model to a new dataset.

#         Parameters
#         ----------
#         X : xr.DataArray
#             Covariate data
#         be : xr.DataArray
#             Batch effect data
#         be_maps : dict[str, dict[str, int]]
#             Batch effect maps
#         Y : xr.DataArray
#             Response data

#         Returns
#         -------
#         TransferableRegressionModel
#             New instance of the regression model
#         """
#         pass
