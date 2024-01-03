from typing import List
from collections import OrderedDict
import pymc as pm
import numpy as np


class HBRData:

    def __init__(self, X: np.ndarray, y: np.ndarray, batch_effects: np.ndarray, covariate_dims: List = None, batch_effect_dims: List = None):

        self.X = X
        self.y = y
        self.batch_effects = batch_effects
        self.batch_effect_dims = batch_effect_dims
        self.covariate_dims = covariate_dims

        self.X, self.y, self.batch_effects = self.expand_all(
            'X', 'y', 'batch_effects')

        assert self.y.shape[1] == 1, "y contains more than one response variable, this is not supported"
        assert self.X.shape[0] == y.shape[0] == batch_effects.shape[0], "X, y and batch_effects must have the same number of rows"

        self._n_covariates = self.X.shape[1]
        self._n_response_vars = self.y.shape[1]
        self._n_datapoints = self.X.shape[0]
        self._n_batch_effect_columns = self.batch_effects.shape[1]

        # The coords will be passed to the pymc model
        self._coords = OrderedDict()  # This preserves the order of the keys
        self._coords_mutable = OrderedDict()

        # Create datapoint coordinates
        self._coords_mutable['datapoints'] = np.arange(self._n_datapoints)

        # Create covariate dims if they are not provided
        if covariate_dims is None:
            self.covariate_dims = ['covariate_' + str(i)
                                   for i in range(self._n_covariates)]
        assert len(
            self.covariate_dims) == self._n_covariates, "The number of covariate dimensions must match the number of covariates"
        self._coords['covariates'] = self.covariate_dims

        # Create batch_effect dims if they are not provided
        if self.batch_effect_dims is None:
            self.batch_effect_dims = [
                'batch_effect_' + str(i) for i in range(self._n_batch_effect_columns)]
        assert len(self.batch_effect_dims) == self._n_batch_effect_columns, "The number of batch effect dimensions must match the number of batch effect columns"
        self._coords['batch_effects'] = self.batch_effect_dims

        # This will be used to index the batch effects in the pymc model
        # Need to be pm.mutableData
        self._batch_effect_indices = []

        # Add the batch effect dimensions to the coords
        # Also accumulate the batch effect indices
        for batch_effect_column in range(self._n_batch_effect_columns):
            be_values, be_indices = np.unique(
                batch_effects[:, batch_effect_column], return_inverse=True)
            self._coords[self.batch_effect_dims[batch_effect_column]] = be_values
            self._batch_effect_indices.append(be_indices)

    def add_to_pymc_model(self, model: pm.Model) -> None:
        """
        Adds the data to the pymc model.
        """
        # TODO set_data if applicable
        with model:
            self.pm_X = pm.MutableData(
                "X", self.X, dims=('datapoints', 'covariates'))
            self.pm_y = pm.MutableData(
                "y", self.y, dims=('datapoints', 'response_vars'))
            self.pm_batch_effect_indices = tuple([pm.Data(self.batch_effect_dims[i], self.batch_effect_indices[i], mutable=True, dims=(
                'datapoints',)) for i in range(self._n_batch_effect_columns)])

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
        assert len(
            data_attr.shape) == 2, f"{data_attr_str} must be a 1D or 2D array"
        return data_attr

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
    def coords(self):
        return self._coords

    @property
    def coords_mutable(self):
        return self._coords_mutable

    @property
    def batch_effect_indices(self):
        return self._batch_effect_indices
