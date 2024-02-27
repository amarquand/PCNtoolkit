import warnings
from collections import OrderedDict
from typing import Any, List

import numpy as np
import pymc as pm


class HBRData:
    def __init__(
        self,
        X: np.ndarray,
        y: np.ndarray = None,
        batch_effects: np.ndarray = None,
        response_var_dims: List = None,
        covariate_dims: List = None,
        batch_effect_dims: List = None,
        datapoint_coords: List = None,
    ):
        self.check_and_set_data(X, y, batch_effects)

        # Set the number of covariates, response vars, datapoints and batch effect columns
        self._n_covariates = self.X.shape[1]
        self._n_response_vars = self.y.shape[1]
        self._n_datapoints = self.X.shape[0]
        self._n_batch_effect_columns = self.batch_effects.shape[1]

        # The coords will be passed to the pymc model
        self._coords = OrderedDict()  # This preserves the order of the keys
        self._coords_mutable = OrderedDict()

        # Create datapoint coordinates
        if datapoint_coords is None:
            self._coords_mutable["datapoints"] = [
                f"datapoint_{i}" for i in np.arange(self._n_datapoints)
            ]
        else:
            self._coords_mutable["datapoints"] = datapoint_coords

        # Create covariate dims if they are not provided
        self.covariate_dims = covariate_dims
        if self.covariate_dims is None:
            self.covariate_dims = [
                "covariate_" + str(i) for i in range(self._n_covariates)
            ]
        assert (
            len(self.covariate_dims) == self._n_covariates
        ), "The number of covariate dimensions must match the number of covariates"
        self._coords["covariates"] = self.covariate_dims

        # Create response var dims if they are not provided
        self.response_var_dims = response_var_dims
        if self.response_var_dims is None:
            self.response_var_dims = [
                "response_var_" + str(i) for i in range(self._n_response_vars)
            ]
        elif type(self.response_var_dims) != list:
            self.response_var_dims = [self.response_var_dims]
        assert (
            len(self.response_var_dims) == self._n_response_vars
        ), "The number of response var dimensions must match the number of response vars"
        self._coords["response_vars"] = self.response_var_dims

        # Create batch_effect dims if they are not provided
        self.batch_effect_dims = batch_effect_dims
        if self.batch_effect_dims is None:
            self.batch_effect_dims = [
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

        self.create_batch_effect_indices()

    def check_and_set_data(self, X, y, batch_effects):
        """
        Checks that the data is valid and sets the data attributes.
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

        self.X, self.y, self.batch_effects = self.expand_all("X", "y", "batch_effects")

        # Check that the dimensions are correct
        assert (
            self.X.shape[0] == self.y.shape[0] == self.batch_effects.shape[0]
        ), "X, y and batch_effects must have the same number of rows"

    def add_to_graph(self, model: pm.Model) -> None:
        """
        Add the data to the pymc model graph using the model context.
        """
        with model:
            self.pm_X = pm.MutableData("X", self.X, dims=("datapoints", "covariates"))
            self.pm_y = pm.MutableData(
                "y", self.y, dims=("datapoints", "response_vars")
            )
            self.pm_batch_effect_indices = tuple(
                [
                    pm.Data(
                        str(self.batch_effect_dims[i]),
                        self.batch_effect_indices[i],
                        mutable=True,
                        dims=("datapoints",),
                    )
                    for i in range(self._n_batch_effect_columns)
                ]
            )

            model.custom_batch_effect_dims = self.batch_effect_dims

    def set_data_in_existing_model(self, model: pm.Model) -> None:
        """
        Sets the data in an existing pymc model.
        """
        model.set_data(
            "X",
            self.X,
            coords={"datapoints": self._coords_mutable["datapoints"]},
        )
        self.pm_X = model["X"]
        model.set_data("y", self.y)
        self.pm_y = model["y"]
        be_acc = []
        for i in range(self._n_batch_effect_columns):
            model.set_data(str(self.batch_effect_dims[i]), self.batch_effect_indices[i])
            be_acc.append(model[self.batch_effect_dims[i]])
        self.pm_batch_effect_indices = tuple(be_acc)

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
        self.create_batch_effect_indices()

    def create_batch_effect_indices(self):
        self.batch_effect_indices = []
        for i, v in enumerate(self.batch_effect_dims):
            self.batch_effect_indices.append(
                np.array(
                    [self._batch_effects_maps[v][w] for w in self.batch_effects[:, i]]
                )
            )

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
    def batch_effects_maps(self):
        return self._batch_effects_maps
