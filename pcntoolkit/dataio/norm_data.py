from typing import Tuple, Union

import numpy as np
import xarray as xr
from sklearn.model_selection import StratifiedKFold, train_test_split

from pcntoolkit.dataio.basis_expansions import create_bspline_basis, create_poly_basis
from pcntoolkit.dataio.scaler import scaler


class NormData(xr.Dataset):
    """This class is only here as a placeholder for now. It will be used to store the data for fitting normative models."""

    """Should keep track of the dimensions and coordinates of the data, and provide consistency between splits of the data."""

    __slots__ = (
        "scaled_X",
        "scaled_y",
        "inscaler",
        "outscaler",
        "bspline_basis",
        "Phi",
        "batch_effects_maps",
        "name",
        "covariates",
        "basis_functions",
    )

    def __init__(self, name, data_vars, coords, attrs) -> None:
        super().__init__(data_vars=data_vars, coords=coords, attrs=attrs)
        self.create_batch_effects_maps()
        self.name = name

    @classmethod
    def from_ndarrays(cls, name, X, y, batch_effects, attrs=None):
        if y.ndim == 1:
            y = y[:, None]
        return cls(
            name,
            {
                "X": (["datapoints", "covariates"], X),
                "y": (["datapoints", "response_vars"], y),
                "batch_effects": (["datapoints", "batch_effect_dims"], batch_effects),
            },
            coords={
                "datapoints": [f"datapoint_{i}" for i in np.arange(X.shape[0])],
                "covariates": [f"covariate_{i}" for i in np.arange(X.shape[1])],
                "response_vars": [f"response_var_{i}" for i in np.arange(y.shape[1])],
                "batch_effect_dims": [
                    f"batch_effect_{i}" for i in range(batch_effects.shape[1])
                ],
                "basis_functions": np.arange(X.shape[1]),
            },
            attrs=attrs,
        )

    @classmethod
    def from_fsl(cls, fsl_folder, config_params) -> "NormData":
        """Load a normative dataset from a FSL file."""
        pass

    @classmethod
    def from_nifti(cls, nifti_folder, config_params) -> "NormData":
        """Load a normative dataset from a Nifti file."""
        pass

    @classmethod
    def from_bids(cls, bids_folder, config_params) -> "NormData":
        """Load a normative dataset from a BIDS dataset."""
        pass

    @classmethod
    def from_xarray(cls, xarray_dataset) -> "NormData":
        """Load a normative dataset from an xarray dataset."""
        pass

    @classmethod
    def from_dataframe(
        cls, name, dataframe, covariates, batch_effects, response_vars, attrs=None
    ):
        return cls(
            name,
            {
                "X": (["datapoints", "covariates"], dataframe[covariates].to_numpy()),
                "y": (
                    ["datapoints", "response_vars"],
                    dataframe[response_vars].to_numpy(),
                ),
                "batch_effects": (
                    ["datapoints", "batch_effect_dims"],
                    dataframe[batch_effects].to_numpy(),
                ),
            },
            coords={
                "datapoints": [
                    f"datapoint_{i}"
                    for i in np.arange(dataframe[covariates].to_numpy().shape[0])
                ],
                "response_vars": response_vars,
                "covariates": covariates,
                "batch_effect_dims": batch_effects,
            },
            attrs=attrs,
        )

    def train_test_split(
        self, splits: Tuple[float, ...], split_names: Tuple[str, ...] = None
    ) -> Tuple["NormData", ...]:
        """Split the data into 2 datasets."""

        batch_effects_added_strings = np.core.defchararray.add(
            *[
                self.batch_effects[:, i].astype(str)
                for i in range(self.batch_effects.shape[1])
            ]
        )
        train_idx, test_idx = train_test_split(
            np.arange(self.X.shape[0]),
            test_size=splits[1],
            random_state=42,
            stratify=batch_effects_added_strings,
        )
        split1 = self.isel(datapoints=train_idx)
        split2 = self.isel(datapoints=test_idx)
        split1.name = split_names[0]
        split2.name = split_names[1]
        return split1, split2

    def kfold_split(self, k: int):
        # Returns an iterator of (NormData, NormData) objects, split into k folds
        stratified_kfold_split = StratifiedKFold(
            n_splits=k, shuffle=True, random_state=42
        )
        batch_effects_added_strings = np.core.defchararray.add(
            *[
                self.batch_effects[:, i].astype(str)
                for i in range(self.batch_effects.shape[1])
            ]
        )
        for train_idx, test_idx in stratified_kfold_split.split(
            self.X, batch_effects_added_strings
        ):
            split1 = self.isel(datapoints=train_idx)
            split2 = self.isel(datapoints=test_idx)
            split1.name = "train"
            split2.name = "test"

            yield split1, split2

    def create_batch_effects_maps(self):
        # create a dictionary with for each column in the batch effects, a dict from value to int
        self.batch_effects_maps = {}
        for i, dim in enumerate(self.batch_effect_dims.to_numpy()):
            self.batch_effects_maps[dim] = {
                value: j for j, value in enumerate(np.unique(self.batch_effects[:, i]))
            }

    def is_compatible_with(self, other: "NormData"):
        """Check if the data is compatible with another dataset."""
        same_covariates = np.all(self.covariates == other.covariates)
        same_batch_effect_dims = np.all(
            self.batch_effect_dims == other.batch_effect_dims
        )
        same_batch_effects_maps = self.batch_effects_maps == other.batch_effects_maps
        return same_covariates and same_batch_effect_dims and same_batch_effects_maps

    def responsevar_iter(self):
        # Returns an iterator over NormData objects, each containing only one response variable
        for response_var in self.response_vars:
            yield self.sel(response_vars=response_var)

    def expand_basis(
        self,
        basis_expansion,
        order=3,
        nknots=5,
        basis_column: int = 0,
        source_array: str = "scaled_X",
        intercept: bool = False,
    ):
        # Expand the basis of the source array
        if source_array == "scaled_X":
            source_array = self.scaled_X
        elif source_array == "X":
            source_array = self.X

        all_arrays = [source_array.data]
        all_dims = list(self.covariates.to_numpy())

        # Create a new array with the expanded basis
        if basis_expansion == "polynomial":
            expanded_basis = create_poly_basis(
                source_array.data[:, basis_column], order
            )
            all_arrays.append(expanded_basis)
            all_dims.extend(
                [f"{basis_expansion}_{i}" for i in range(expanded_basis.shape[1])]
            )
        elif basis_expansion == "bspline":
            self.bspline_basis = create_bspline_basis(
                np.min(source_array[:, basis_column]),
                np.max(source_array.data[:, basis_column]),
                order,
                nknots,
            )
            expanded_basis = np.array(
                [self.bspline_basis(c) for c in source_array.data[:, basis_column]]
            )
            all_arrays.append(expanded_basis)
            all_dims.extend(
                [f"{basis_expansion}_{i}" for i in range(expanded_basis.shape[1])]
            )

        if intercept:
            all_dims.append("intercept")
            all_arrays.append(np.ones((expanded_basis.shape[0], 1)))

        Phi = np.concatenate(all_arrays, axis=1)
        print(all_dims)
        self.Phi = xr.DataArray(
            Phi,
            coords={"basis_functions": all_dims},
            dims=["datapoints", "basis_functions"],
        )

    def scale_forward(self, inscalers: dict[str, scaler], outscaler: dict[str, scaler]):
        # Scale X column-wise using the inscalers
        self.scaled_X = xr.DataArray(
            np.zeros(self.X.shape),
            coords=self.X.coords,
            dims=self.X.dims,
            attrs=self.X.attrs,
        )
        for covariate in self.covariates.to_numpy():
            self.scaled_X.loc[:, covariate] = inscalers[covariate].transform(
                self.X.sel(covariates=covariate).data
            )

        # Scale y column-wise using the outscalers
        self.scaled_y = xr.DataArray(
            np.zeros(self.y.shape),
            coords=self.y.coords,
            dims=self.y.dims,
            attrs=self.y.attrs,
        )
        for responsevar in self.response_vars.to_numpy():
            self.scaled_y.loc[:, responsevar] = outscaler[responsevar].transform(
                self.y.sel(response_vars=responsevar).data
            )

    def scale_backward(
        self, inscalers: dict[str, scaler], outscalers: dict[str, scaler]
    ):
        # Scale X column-wise using the inscalers
        self["X"] = xr.DataArray(
            np.zeros(self.scaled_X.shape),
            coords=self.scaled_X.coords,
            dims=self.scaled_X.dims,
            attrs=self.scaled_X.attrs,
        )
        for covariate in self.covariates.to_numpy():
            self.X.loc[:, covariate] = inscalers[covariate].inverse_transform(
                self.scaled_X.sel(covariates=covariate).data
            )

        # Scale y column-wise using the outscalers
        self["y"] = xr.DataArray(
            np.zeros(self.scaled_y.shape),
            coords=self.scaled_y.coords,
            dims=self.scaled_y.dims,
            attrs=self.scaled_y.attrs,
        )
        for responsevar in self.response_vars.to_numpy():
            self.y.loc[:, responsevar] = outscalers[responsevar].inverse_transform(
                self.scaled_y.sel(response_vars=responsevar).data
            )

    def sel(self, *args, **kwargs):
        result = super().sel(*args, **kwargs)
        result.__class__ = NormData
        result.batch_effects_maps = self.batch_effects_maps
        result.name = self.name
        return result

    def isel(self, *args, **kwargs):
        result = super().isel(*args, **kwargs)
        result.__class__ = NormData
        result.batch_effects_maps = self.batch_effects_maps
        result.name = self.name
        return result