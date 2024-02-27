from typing import Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from sklearn.model_selection import StratifiedKFold, train_test_split

from pcntoolkit.dataio.basis_expansions import create_bspline_basis, create_poly_basis
from pcntoolkit.dataio.scaler import scaler


class NormData(xr.Dataset):
    """Should keep track of the dimensions and coordinates of the data, and provide consistency between splits of the data."""

    __slots__ = (
        "X",
        "y",
        "scaled_X",
        "scaled_y",
        "Phi",
        "scaled_centiles",
        "centiles",
        "zscores",
    )

    def __init__(self, name, data_vars, coords, attrs=None) -> None:
        if attrs is None:
            attrs = {}
        attrs["name"] = name
        super().__init__(data_vars=data_vars, coords=coords, attrs=attrs)
        self.create_batch_effects_maps()

    @classmethod
    def from_ndarrays(cls, name, X, y, batch_effects, attrs=None):
        if X.ndim == 1:
            X = X[:, None]
        if y.ndim == 1:
            y = y[:, None]
        if batch_effects.ndim == 1:
            batch_effects = batch_effects[:, None]
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
    def from_xarray(cls, name, xarray_dataset) -> "NormData":
        """Load a normative dataset from an xarray dataset."""
        return cls(
            name,
            xarray_dataset.data_vars,
            xarray_dataset.coords,
            xarray_dataset.attrs,
        )

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

    def create_synthetic_data(
        self,
        n_datapoints: int = 100,
        range_dim: Union[int, str] = 0,
        batch_effects_to_sample: list[list[str]] = None,
    ):
        ## Creates a synthetic dataset with the same dimensions as the original dataset

        # The range_dim specifies for which covariate a range of values will be generated
        if isinstance(range_dim, int):
            range_dim = self.covariates[range_dim].to_numpy().item()

        min_range = np.min(self.X.sel(covariates=range_dim))
        max_range = np.max(self.X.sel(covariates=range_dim))
        X = np.linspace(min_range, max_range, n_datapoints)

        df = pd.DataFrame(X, columns=[range_dim])

        # For all the other covariates:
        for covariate in self.covariates.to_numpy():
            if covariate != range_dim:
                # Use the mean of the original dataset
                df[covariate] = np.mean(self.X.sel(covariates=covariate))

        # The batch effects specifies from which batch_effects can be sampled
        if batch_effects_to_sample is None:
            batch_effects_to_sample = [
                [list(map.keys())[0]] for map in self.batch_effects_maps.values()
            ]
        else:
            # Assert that the batch effects to sample are in the batch effects maps
            for i, bes in enumerate(batch_effects_to_sample):
                for be in bes:
                    assert (
                        be
                        in self.attrs["batch_effects_maps"][
                            self.batch_effect_dims[i].to_numpy().item()
                        ].keys()
                    )

        for i, bes in enumerate(batch_effects_to_sample):
            df[self.batch_effect_dims[i].to_numpy().item()] = np.random.choice(
                bes, n_datapoints
            )

        # For each response variable, sample from a normal distribution with the mean and std of the original dataset
        for response_var in self.response_vars.to_numpy():
            df[response_var] = np.random.normal(
                self.y.sel(response_vars=response_var).mean(),
                self.y.sel(response_vars=response_var).std(),
                n_datapoints,
            )

        to_return = NormData.from_dataframe(
            f"{self.attrs['name']}_synthetic",
            df,
            self.covariates.to_numpy(),
            self.batch_effect_dims.to_numpy(),
            self.response_vars.to_numpy(),
        )
        # set the batch effects maps
        to_return.attrs["batch_effects_maps"] = self.attrs["batch_effects_maps"]

        return to_return

    def train_test_split(
        self, splits: Tuple[float, ...], split_names: Tuple[str, ...] = None
    ) -> Tuple["NormData", ...]:
        """Split the data into 2 datasets."""
        batch_effects_stringified = np.core.defchararray.add(
            *[
                self.batch_effects[:, i].astype(str)
                for i in range(self.batch_effects.shape[1])
            ]
        )
        train_idx, test_idx = train_test_split(
            np.arange(self.X.shape[0]),
            test_size=splits[1],
            random_state=42,
            stratify=batch_effects_stringified,
        )
        split1 = self.isel(datapoints=train_idx)
        split1.attrs = self.attrs.copy()
        split2 = self.isel(datapoints=test_idx)
        split2.attrs = self.attrs.copy()
        if split_names is not None:
            split1.attrs["name"] = split_names[0]
            split2.attrs["name"] = split_names[1]
        else:
            split1.attrs["name"] = f"{self.attrs['name']}_train"
            split2.attrs["name"] = f"{self.attrs['name']}_test"
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
            yield split1, split2

    def create_batch_effects_maps(self):
        # create a dictionary with for each column in the batch effects, a dict from value to int
        batch_effects_maps = {}
        for i, dim in enumerate(self.batch_effect_dims.to_numpy()):
            batch_effects_maps[dim] = {
                value: j for j, value in enumerate(np.unique(self.batch_effects[:, i]))
            }
        self.attrs["batch_effects_maps"] = batch_effects_maps

    def is_compatible_with(self, other: "NormData"):
        """Check if the data is compatible with another dataset."""
        same_covariates = np.all(self.covariates == other.covariates)
        same_batch_effect_dims = np.all(
            self.batch_effect_dims == other.batch_effect_dims
        )
        same_batch_effects_maps = (
            self.attrs["batch_effects_maps"] == other.attrs["batch_effects_maps"]
        )
        return same_covariates and same_batch_effect_dims and same_batch_effects_maps

    # def responsevar_iter(self):
    #     # Returns an iterator over NormData objects, each containing only one response variable
    #     for response_var in self.response_vars:
    #         yield self.sel(response_vars=response_var)

    def scale_forward(self, inscalers: dict[str, scaler], outscaler: dict[str, scaler]):
        # Scale X column-wise using the inscalers
        self["scaled_X"] = xr.DataArray(
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
        self["scaled_y"] = xr.DataArray(
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

        # Unscale the centiles, if they exist
        if "scaled_centiles" in self.data_vars:
            self["centiles"] = xr.DataArray(
                np.zeros(self.scaled_centiles.shape),
                coords=self.scaled_centiles.coords,
                dims=self.scaled_centiles.dims,
                attrs=self.scaled_centiles.attrs,
            )
            for responsevar in self.response_vars.to_numpy():
                self.centiles.loc[{"response_vars": responsevar}] = outscalers[
                    responsevar
                ].inverse_transform(
                    self.scaled_centiles.sel(response_vars=responsevar).data
                )

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
            self.attrs["bspline_basis"] = create_bspline_basis(
                xmin=np.min(source_array[:, basis_column]),
                xmax=np.max(source_array.data[:, basis_column]),
                p=order,
                nknots=nknots,
            )
            expanded_basis = np.array(
                [
                    self.attrs["bspline_basis"](c)
                    for c in source_array.data[:, basis_column]
                ]
            )
            all_arrays.append(expanded_basis)
            all_dims.extend(
                [f"{basis_expansion}_{i}" for i in range(expanded_basis.shape[1])]
            )

        if intercept:
            all_dims.append("intercept")
            all_arrays.append(np.ones((expanded_basis.shape[0], 1)))

        Phi = np.concatenate(all_arrays, axis=1)
        self["Phi"] = xr.DataArray(
            Phi,
            coords={"basis_functions": all_dims},
            dims=["datapoints", "basis_functions"],
        )
        pass

    def plot_centiles(
        self,
        covariate: str = None,
        batch_effects: Union[str, list[str]] = None,
        show_data=False,
        scatter_data: "NormData" = None,
    ):
        """Plot the centiles for all response variables."""
        for response_var in self.coords["response_vars"].to_numpy():
            self._plot_centiles(
                response_var, covariate, batch_effects, show_data, scatter_data
            )

    def _plot_centiles(
        self,
        response_var: str,
        covariate: str = None,
        batch_effects: Union[str, list[str]] = None,
        show_data=False,
        scatter_data: "NormData" = None,
    ):
        """Plot the centiles for a single response variable."""
        # Use the first covariate, if not specified
        if covariate is None:
            covariate = self.covariates[0].to_numpy().item()

        # Set all batch effects to 0 if not specified
        if batch_effects is None:
            batch_effects = [0] * len(self.coords["batch_effect_dims"])

        if show_data and (scatter_data is None):
            scatter_data = self

        # Filter the covariate and responsevar that are to be plotted
        filter_dict = {
            "covariates": covariate,
            "response_vars": response_var,
        }
        filtered = self.sel(filter_dict)

        # Filter out the correct batch effects
        filtered: xr.Dataset = filtered.where(filtered.batch_effects == batch_effects)

        plt.figure()
        for zscore in self.coords["cummulative_densities"]:
            # Make the mean line thicker
            if zscore == 0:
                linewidth = 3
            else:
                linewidth = 1

            # Make the outer centiles dashed
            if zscore <= -2 or zscore >= 2:
                linestyle = "--"
            else:
                linestyle = "-"
            plt.plot(
                filtered.X,
                filtered.centiles.sel(cummulative_densities=zscore),
                color="black",
                linewidth=linewidth,
                linestyle=linestyle,
            )

        if show_data:
            filtered_scatter = scatter_data.sel(filter_dict)
            filtered_scatter: xr.Dataset = filtered_scatter.where(
                filtered_scatter.batch_effects == batch_effects
            )
            plt.scatter(
                filtered_scatter.X, filtered_scatter.y, color="red", label="data"
            )

        plt.title(f"Quantiles for {response_var}")
        plt.xlabel(covariate)
        plt.ylabel(response_var)
        plt.show()

    def plot_qq(self):
        """Create a QQ-plot for all response variables."""
        for response_var in self.coords["response_vars"].to_numpy():
            self._plot_qq(response_var)

    def _plot_qq(
        self,
        response_var: str,
    ):
        """Create a QQ-plot for a single response variable."""

        # Filter the responsevar that is to be plotted
        filter_dict = {
            "response_vars": response_var,
        }

        filt = self.sel(filter_dict)

        ran = np.random.randn(filt.X.shape[0])
        ran.sort()

        z_scores = filt.zscores.data
        z_scores.sort()

        plt.figure()
        plt.scatter(ran, z_scores)
        plt.title(f"QQ-plot for {response_var}")
        plt.xlabel("Theoretical quantiles")
        plt.ylabel("Predicted quantiles")
        plt.show()

    def select_batch_effects(self, batch_effects: dict[str, list[str]]):
        """Select only the batch_effects specified."""
        mask = np.zeros(self.batch_effects.shape[0], dtype=bool)
        for key, values in batch_effects.items():
            this_batch_effect = self.batch_effects.sel(batch_effect_dims=key)
            for value in values:
                mask = np.logical_or(mask, this_batch_effect == value)

        to_return = self.where(mask).dropna(dim="datapoints", how="all")
        if type(to_return) == xr.Dataset:
            to_return = NormData.from_xarray(
                f"{self.attrs['name']}_selected", to_return
            )
        to_return.attrs["batch_effects_maps"] = self.attrs["batch_effects_maps"].copy()
        return to_return
