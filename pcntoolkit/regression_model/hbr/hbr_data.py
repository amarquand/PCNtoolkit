"""
Data container and processor for Hierarchical Bayesian Regression (HBR) models.

This module provides the HBRData class which handles data preparation, validation,
and integration with PyMC models for hierarchical Bayesian regression. It manages
covariates, response variables, and batch effects while providing utilities for
data transformation and model integration.

Notes
-----
The module requires PyMC for Bayesian modeling integration and NumPy for array operations.
All data is validated and transformed into appropriate formats for use in HBR models.
"""

import warnings
from collections import OrderedDict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pymc as pm  # type: ignore


class HBRData:
    """
    A data container class for Hierarchical Bayesian Regression models.

    This class handles the preparation and management of data for HBR models,
    including covariates, response variables, and batch effects. It provides
    functionality for data validation, transformation, and integration with
    PyMC models.

    Parameters
    ----------
    X : np.ndarray
        Covariate matrix of shape (n_samples, n_features)
    y : np.ndarray, optional
        Response variable array of shape (n_samples,) or (n_samples, 1)
    batch_effects : np.ndarray, optional
        Batch effects matrix of shape (n_samples, n_batch_effects)
    response_var : str, optional
        Name of the response variable
    covariate_dims : List[str], optional
        Names of the covariate dimensions
    batch_effect_dims : List[str], optional
        Names of the batch effect dimensions
    datapoint_coords : List[Any], optional
        Coordinates for each datapoint

    Attributes
    ----------
    X : np.ndarray
        Processed covariate matrix
    y : np.ndarray
        Processed response variable array
    batch_effects : np.ndarray
        Processed batch effects matrix
    response_var : str
        Name of the response variable
    covariate_dims : List[str]
        Names of covariate dimensions
    batch_effect_dims : List[str]
        Names of batch effect dimensions
    pm_X : pm.Data
        PyMC Data container for covariates
    pm_y : pm.Data
        PyMC Data container for response variable
    pm_batch_effect_indices : Tuple[pm.Data, ...]
        PyMC Data containers for batch effect indices

    Examples
    --------
    >>> X = np.random.randn(100, 5)
    >>> y = np.random.randn(100)
    >>> batch_effects = np.random.randint(0, 3, (100, 2))
    >>> data = HBRData(X, y, batch_effects)
    >>> with pm.Model() as model:
    ...     data.add_to_graph(model)
    """

    def __init__(
        self,
        X: np.ndarray,
        y: Optional[np.ndarray] = None,
        batch_effects: Optional[np.ndarray] = None,
        response_var: Optional[str] = None,
        covariate_dims: Optional[List[str]] = None,
        batch_effect_dims: Optional[List[str]] = None,
        datapoint_coords: Optional[List[Any]] = None,
    ) -> None:
        self.check_and_set_data(X, y, batch_effects)

        # Set the response var
        self.response_var = response_var

        # Set the number of covariates, datapoints and batch effect columns
        self._n_covariates = self.X.shape[1]
        self._n_datapoints = self.X.shape[0]
        self._n_batch_effect_columns = self.batch_effects.shape[1]

        # The coords will be passed to the pymc model
        self._coords = OrderedDict()  # This preserves the order of the keys

        # Create datapoint coordinates
        self._coords["datapoints"] = datapoint_coords or list(
            np.arange(self._n_datapoints)
        )

        # Create covariate dims if they are not provided
        if covariate_dims is None:
            self.covariate_dims = [
                "covariate_" + str(i) for i in range(self._n_covariates)
            ]
        else:
            self.covariate_dims = covariate_dims
        assert (
            len(self.covariate_dims) == self._n_covariates
        ), "The number of covariate dimensions must match the number of covariates"
        self._coords["covariates"] = self.covariate_dims

        # Create batch_effect dims if they are not provided
        self.batch_effect_dims = batch_effect_dims or [
            "batch_effect_" + str(i) for i in range(self._n_batch_effect_columns)
        ]
        assert (
            len(self.batch_effect_dims) == self._n_batch_effect_columns
        ), "The number of batch effect dimensions must match the number of batch effect columns"
        self._coords["batch_effects"] = self.batch_effect_dims

        # This will be used to index the batch effects in the pymc model
        self._batch_effects_maps = {}
        for i, v in enumerate(self.batch_effect_dims):
            be_values = np.unique(self.batch_effects[:, i])
            self._batch_effects_maps[v] = {w: j for j, w in enumerate(be_values)}
            self._coords[v] = be_values

        self.pm_X: pm.Data = None  # type: ignore
        self.pm_y: pm.Data = None  # type:ignore
        self.pm_batch_effect_indices: dict[str, pm.Data] = None  # type: ignore

        self.create_batch_effect_indices()

    def check_and_set_data(
        self,
        X: np.ndarray,
        y: Optional[np.ndarray],
        batch_effects: Optional[np.ndarray],
    ) -> None:
        """
        Validate and set the input data arrays.

        Performs validation checks on the input arrays and sets them as instance
        attributes. Ensures proper dimensionality and compatibility between arrays.

        Parameters
        ----------
        X : np.ndarray
            Covariate matrix of shape (n_samples, n_features)
        y : np.ndarray or None
            Response variable array of shape (n_samples,) or (n_samples, 1).
            If None, creates zero array of appropriate shape.
        batch_effects : np.ndarray or None
            Batch effects matrix of shape (n_samples, n_batch_effects).
            If None, creates zero array of appropriate shape.

        Raises
        ------
        ValueError
            If X is None
        AssertionError
            If array dimensions are incompatible or y has incorrect shape

        Notes
        -----
        - Automatically expands 1D arrays to 2D
        - Squeezes y to 1D if it's a 2D array with one column
        """

        if X is None:
            raise ValueError("X must be provided")
        else:
            self.X = X

        if y is None:
            self.y = np.zeros((X.shape[0], 1))
        else:
            self.y = y

        if batch_effects is None:
            warnings.warn(
                "batch_effects is not provided, setting self.batch_effects to zeros"
            )
            self.batch_effects = np.zeros((X.shape[0], 1))
        else:
            self.batch_effects = batch_effects

        self.X, self.batch_effects = self.expand_all("X", "batch_effects")

        # Check that the dimensions are correct
        assert (
            self.X.shape[0] == self.y.shape[0] == self.batch_effects.shape[0]
        ), "X, y and batch_effects must have the same number of rows"
        if len(self.y.shape) > 1:
            assert (
                self.y.shape[1] == 1
            ), "y can only have one column, or it must be a 1D array"
            self.y = np.squeeze(self.y)

    def add_to_graph(self, model: pm.Model) -> None:
        """
        Add data variables to a PyMC model graph.

        Parameters
        ----------
        model : pm.Model
            PyMC model to add the data variables to

        Notes
        -----
        Creates PyMC Data objects for X, y, and batch effect indices within
        the model context. Also adds custom batch effect dimensions to the model.
        """
        with model:
            self.pm_X = pm.Data("X", self.X, dims=("datapoints", "covariates"))
            self.pm_y = pm.Data("y", self.y, dims=("datapoints",))
            self.pm_batch_effect_indices = {k:
                pm.Data(
                    k + "_data",
                    self.batch_effect_indices[i],
                    dims=("datapoints",),
                )
                for i, k in enumerate(self.batch_effect_dims)
            }
            model.custom_batch_effect_dims = self.batch_effect_dims  # type: ignore

    def set_data_in_existing_model(self, model: pm.Model) -> None:
        """
        Update data values in an existing PyMC model.

        Parameters
        ----------
        model : pm.Model
            Existing PyMC model whose data values need to be updated

        Notes
        -----
        Updates the values of X, y, and batch effect indices in the model
        while preserving the model structure.
        """
        model.set_data(
            "X",
            self.X,
            coords={"datapoints": self._coords["datapoints"]},
        )
        self.pm_X = model["X"]
        model.set_data("y", self.y)
        self.pm_y = model["y"]
        be_acc = {}
        for i in range(self._n_batch_effect_columns):
            model.set_data(
                str(self.batch_effect_dims[i]) + "_data", self.batch_effect_indices[i]
            )
            be_acc[self.batch_effect_dims[i]] = model[self.batch_effect_dims[i] + "_data"]
        self.pm_batch_effect_indices = be_acc

    def expand_all(self, *args: str) -> Tuple[np.ndarray, ...]:
        """
        Expand multiple data attributes to 2D arrays if necessary.

        Parameters
        ----------
        *args : str
            Names of attributes to expand

        Returns
        -------
        Tuple[np.ndarray, ...]
            Tuple of expanded arrays

        See Also
        --------
        expand : Method for expanding individual arrays
        """
        return tuple(self.expand(arg) for arg in args)

    def expand(self, data_attr_str: str) -> np.ndarray:
        """
        Expand a single data attribute to a 2D array if necessary.

        Parameters
        ----------
        data_attr_str : str
            Name of the attribute to expand

        Returns
        -------
        np.ndarray
            Expanded array

        Raises
        ------
        AssertionError
            If the array is not 1D or 2D
        """
        data_attr: np.ndarray = getattr(self, data_attr_str)
        if len(data_attr.shape) == 1:
            data_attr = data_attr.reshape(-1, 1)
        assert len(data_attr.shape) == 2, f"{data_attr_str} must be a 1D or 2D array"
        return data_attr

    def set_batch_effects_maps(
        self, batch_effects_maps: Dict[str, Dict[Any, int]]
    ) -> None:
        """
        Set the mapping between batch effect values and their indices.

        Parameters
        ----------
        batch_effects_maps : Dict[str, Dict[Any, int]]
            Mapping from batch effect names to value-index dictionaries

        Notes
        -----
        Updates the batch effect indices after setting the new maps.
        """
        self._batch_effects_maps = batch_effects_maps
        self.create_batch_effect_indices()

    def create_batch_effect_indices(self) -> None:
        """
        Create numerical indices for batch effects.

        Creates a list of arrays where each array contains the numerical indices
        corresponding to the categorical batch effect values. These indices are
        used for indexing in the PyMC model.

        Notes
        -----
        The indices are created based on the current batch_effects_maps.
        """
        self.batch_effect_indices = []
        for i, v in enumerate(self.batch_effect_dims):
            self.batch_effect_indices.append(
                np.array(
                    [self._batch_effects_maps[v][w] for w in self.batch_effects[:, i]]
                )
            )

    # pylint: disable=C0116

    @property
    def n_covariates(self) -> int:
        return self._n_covariates

    @property
    def n_datapoints(self) -> int:
        return self._n_datapoints

    @property
    def n_batch_effect_columns(self) -> int:
        return self._n_batch_effect_columns

    @property
    def coords(self) -> OrderedDict:
        return self._coords

    @property
    def batch_effects_maps(self) -> Dict[str, Dict[Any, int]]:
        return self._batch_effects_maps
