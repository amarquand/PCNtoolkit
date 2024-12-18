"""
Module for handling data in Bayesian linear regression.

This module provides the BLRData class, which is used to store and manage
data for Bayesian linear regression models. It includes functionality for
validating and expanding data arrays, setting batch effects, and managing
covariate and data point counts.

Classes
-------
BLRData
    An object to store the data used in Bayesian linear regression.
"""

from typing import Any, Optional, Tuple

import numpy as np


class BLRData:
    """An object to store the data used in Bayesian linear regression."""

    def __init__(
        self,
        X: np.ndarray,
        y: Optional[np.ndarray] = None,
        var_X: Optional[np.ndarray] = None,
        batch_effects: Optional[np.ndarray] = None,
        response_var: Optional[str] = None,
    ):
        """
        Initializes the BLRData object.
        Parameters
        ----------
        X : np.ndarray
            The input data matrix.
        y : np.ndarray, optional
            The response variable.
        var_X : np.ndarray, optional
            The variance of the input data.
        batch_effects : np.ndarray, optional
            The batch effects data.
        response_var : str, optional
            The name of the response variable.
        """
        self._batch_effects_maps = {}
        self.check_and_set_data(X, y, var_X, batch_effects)
        self.response_var = response_var
        self._n_covariates = self.X.shape[1]
        self._n_datapoints = self.X.shape[0]
        self._n_batch_effect_columns = self.batch_effects.shape[1]

    def check_and_set_data(
        self, X: np.ndarray, y: Optional[np.ndarray], var_X: Optional[np.ndarray], batch_effects: Optional[np.ndarray]
    ) -> None:
        """
        Checks that the data is valid and sets the data attributes.

        Parameters
        ----------
        X : np.ndarray
            The input data matrix.
        y : np.ndarray, optional
            The response variable.
        var_X : np.ndarray, optional
            The variance of the input data.
        batch_effects : np.ndarray, optional
            The batch effects data.

        Raises
        ------
        ValueError
            If X is not provided.
        """
        if X is None:
            raise ValueError("X must be provided")
        else:
            self.X = X

        if y is None:
            # warnings.warn("y is not provided, setting self.y to zeros")
            self.y = np.zeros((X.shape[0], 1))
        else:
            self.y = y

        if var_X is None:
            # warnings.warn("var_X is not provided, setting self.var_X to zeros")
            self.var_X = np.zeros((X.shape[0], 1))
        else:
            self.var_X = var_X

        if batch_effects is None:
            # warnings.warn(
            #     "batch_effects is not provided, setting self.batch_effects to zeros"
            # )
            self.batch_effects = np.zeros((X.shape[0], 1))
        else:
            self.batch_effects = batch_effects

        self.X, self.var_X, self.batch_effects = self.expand_all(
            "X", "var_X", "batch_effects"
        )

        assert (
            self.X.shape[0]
            == self.y.shape[0]
            == self.var_X.shape[0]
            == self.batch_effects.shape[0]
        ), "X, var_X, y and batch_effects must have the same number of rows"

        if len(self.y.shape) > 1:
            assert (
                self.y.shape[1] == 1
            ), "y can only have one column, or it must be a 1D array"
            self.y = np.squeeze(self.y)

    def expand_all(self, *args: str) -> Tuple[np.ndarray, ...]:
        """
        Expands all data attributes.

        Parameters
        ----------
        *args : str
            The names of the data attributes to expand.

        Returns
        -------
        tuple of np.ndarray
            The expanded data attributes.
        """
        return tuple(self.expand(arg) for arg in args)

    def expand(self, data_attr_str: str) -> np.ndarray:
        """
        Expands a 1D array to a 2D array if necessary.

        Parameters
        ----------
        data_attr_str : str
            The name of the data attribute to expand.

        Returns
        -------
        np.ndarray
            The expanded data attribute.

        Raises
        ------
        AssertionError
            If the array is not 1D or 2D.
        """
        data_attr: np.ndarray = getattr(self, data_attr_str)
        if len(data_attr.shape) == 1:
            data_attr = data_attr.reshape(-1, 1)
        assert len(data_attr.shape) == 2, f"{data_attr_str} must be a 1D or 2D array"
        return data_attr

    def set_batch_effects_maps(self, batch_effects_maps: dict[str, dict[Any, int]]) -> None:
        """
        Sets the batch effects map.

        Parameters
        ----------
        batch_effects_maps : dict of str to dict of Any to int
            The batch effects map.
        """
        self._batch_effects_maps = batch_effects_maps

    @property
    def n_covariates(self) -> int:
        """int: The number of covariates."""
        return self._n_covariates

    @property
    def n_datapoints(self) -> int:
        """int: The number of data points."""
        return self._n_datapoints

    @property
    def n_batch_effect_columns(self) -> int:
        """int: The number of batch effect columns."""
        return self._n_batch_effect_columns

    @property
    def batch_effects_maps(self) -> dict[str, dict[Any, int]]:
        """dict of str to dict of Any to int: The batch effects map."""
        return self._batch_effects_maps
