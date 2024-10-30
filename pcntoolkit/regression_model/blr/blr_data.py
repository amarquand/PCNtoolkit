import warnings
from typing import Any

import numpy as np


class BLRData:
    """An object to store the data used in Bayesian linear regression."""

    def __init__(
        self,
        X: np.ndarray,
        y: np.ndarray = None,
        var_X: np.ndarray = None,
        batch_effects: np.ndarray = None,
        response_var: str = None,
    ):
        self.check_and_set_data(X, y, var_X, batch_effects)

        # Set the response var
        self.response_var = response_var

        # Set the number of covariates, datapoints and batch effect columns
        self._n_covariates = self.X.shape[1]
        self._n_datapoints = self.X.shape[0]
        self._n_batch_effect_columns = self.batch_effects.shape[1]

    def check_and_set_data(self, X, y, var_X, batch_effects):
        """
        Checks that the data is valid and sets the data attributes.
        """

        if X is None:
            raise ValueError("X must be provided")
        else:
            self.X = X

        if y is None:
            warnings.warn("y is not provided, setting self.y to zeros")
            self.y = np.zeros((X.shape[0], 1))
        else:
            self.y = y

        if var_X is None:
            warnings.warn("var_X is not provided, setting self.var_X to zeros")
            self.var_X = np.zeros((X.shape[0], 1))
        else:
            self.var_X = var_X

        if batch_effects is None:
            warnings.warn(
                "batch_effects is not provided, setting self.batch_effects to zeros"
            )
            self.batch_effects = np.zeros((X.shape[0], 1))
        else:
            self.batch_effects = batch_effects

        self.X, self.var_X, self.batch_effects = self.expand_all(
            "X", "var_X", "batch_effects"
        )

        # Check that the dimensions are correct
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

    def expand_all(self, *args):
        """
        Expands all data attributes.
        """
        return tuple(self.expand(arg) for arg in args)

    def expand(self, data_attr_str: str) -> np.ndarray:
        """
        Expands a 1D array to a 2D array if necessary.
        Raises an error if the array is not 1D or 2D.
        """
        data_attr: np.ndarray = getattr(self, data_attr_str)
        if len(data_attr.shape) == 1:
            data_attr = data_attr.reshape(-1, 1)
        assert len(data_attr.shape) == 2, f"{data_attr_str} must be a 1D or 2D array"
        return data_attr

    def set_batch_effects_maps(self, batch_effects_maps: dict[str, dict[Any, int]]):
        """
        Sets the batch effects map.
        """
        self._batch_effects_maps = batch_effects_maps

    @property
    def n_covariates(self):
        return self._n_covariates

    @property
    def n_datapoints(self):
        return self._n_datapoints

    @property
    def n_batch_effect_columns(self):
        return self._n_batch_effect_columns

    @property
    def batch_effects_maps(self):
        return self._batch_effects_maps
