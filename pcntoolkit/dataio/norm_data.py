from typing import Any, Dict, List, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from sklearn.model_selection import StratifiedKFold, train_test_split

from pcntoolkit.dataio.scaler import scaler


class NormData(xr.Dataset):
    """
    A class for handling normative modeling data, extending xarray.Dataset.

    NormData provides functionality for loading, preprocessing, and managing
    data for normative modeling. It supports various data formats and includes
    methods for data scaling, splitting, and visualization.

    Attributes:
        X (xr.DataArray): Covariate data.
        y (xr.DataArray): Response variable data.
        scaled_X (xr.DataArray): Scaled version of covariate data.
        scaled_y (xr.DataArray): Scaled version of response variable data.
        batch_effects (xr.DataArray): Batch effect data.
        Phi (xr.DataArray): Design matrix.
        scaled_centiles (xr.DataArray): Scaled centile data (if applicable).
        centiles (xr.DataArray): Unscaled centile data (if applicable).
        zscores (xr.DataArray): Z-score data (if applicable).

    Note:
        This class stores both original and scaled versions of X and y data.
        While this approach offers convenience and transparency, it may
        increase memory usage. Consider memory constraints when working with
        large datasets.

    Example:
        >>> data = NormData.from_dataframe("my_data", df, covariates,
        ...                                batch_effects, response_vars)
        >>> data.scale_forward(inscalers, outscalers)
        >>> train_data, test_data = data.train_test_split([0.8, 0.2])
    """

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
        batch_effects_to_sample: Dict[str, List[Any]] = None,
    ):
        """
        Create a synthetic dataset with the same dimensions as the original dataset.

        Inputs:
            n_datapoints: int = 100
                The number of datapoints to create.
            range_dim: Union[int, str] = 0
                The covariate to use for the range of values. np.linspace will be used to generate values between the min and max of this covariate.
            batch_effects_to_sample: list[Any] = None
                The batch effects to sample. For every batch effect, this list should contain the values to sample from.
                If None, the batch effects to sample are the first values in the batch effects maps.
        """
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

        batch_effects_to_sample = batch_effects_to_sample or {}

        # For each batch_effect dimension that is not specified, sample from the first value in the batch effects map
        for dim in self.batch_effect_dims.to_numpy():
            if dim not in batch_effects_to_sample:
                batch_effects_to_sample[dim] = [
                    list(self.attrs["batch_effects_maps"][dim].keys())[0]
                ]

        # # Assert that the batch effects to sample are in the batch effects maps
        for dim, values in batch_effects_to_sample.items():
            assert (
                dim in self.attrs["batch_effects_maps"]
            ), f"{dim} is not a known batch effect dimension"
            assert (
                len(values) > 0
            ), f"No values provided for batch effect dimension {dim}"
            for value in values:
                assert (
                    value in self.attrs["batch_effects_maps"][dim]
                ), f"{value} is not a known value for batch effect dimension {dim}"

        for batch_effect_dim, values_to_sample in batch_effects_to_sample.items():
            df[batch_effect_dim] = np.random.choice(values_to_sample, n_datapoints)

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

    def get_single_batch_effect(self):
        batch_effects_to_sample = [
            [list(map.keys())[0]] for map in self.batch_effects_maps.values()
        ]

        return batch_effects_to_sample

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

    def plot_qq(self, plt_kwargs=None, bound=0):
        """Create a QQ-plot for all response variables."""
        if not plt_kwargs:
            plt_kwargs = {}
        for response_var in self.coords["response_vars"].to_numpy():
            self._plot_qq(response_var, plt_kwargs, bound)

    def _plot_qq(self, response_var: str, plt_kwargs, bound=0):
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
        plt.scatter(ran, z_scores, **plt_kwargs)
        if bound != 0:
            plt.axis([-bound, bound, -bound, bound])
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

    def to_dataframe(self):
        acc = []
        x_columns = [col for col in ["X", "scaled_X"] if hasattr(self, col)]
        y_columns = [col for col in ["y", "zscores", "scaled_y"] if hasattr(self, col)]
        acc.append(
            xr.Dataset.to_dataframe(self[x_columns])
            .reset_index(drop=False)
            .pivot(index="datapoints", columns="covariates", values=x_columns)
        )
        acc.append(
            xr.Dataset.to_dataframe(self[y_columns])
            .reset_index(drop=False)
            .pivot(index="datapoints", columns="response_vars", values=y_columns)
        )
        be = (
            xr.DataArray.to_dataframe(self.batch_effects)
            .reset_index(drop=False)
            .pivot(
                index="datapoints", columns="batch_effect_dims", values="batch_effects"
            )
        )
        be.columns = [("batch_effects", col) for col in be.columns]
        acc.append(be)
        if hasattr(self, "Phi"):
            phi = (
                xr.DataArray.to_dataframe(self.Phi)
                .reset_index(drop=False)
                .pivot(index="datapoints", columns="basis_functions", values="Phi")
            )
            phi.columns = [("Phi", col) for col in phi.columns]
            acc.append(phi)
        if hasattr(self, "centiles"):
            centiles = (
                xr.DataArray.to_dataframe(self.centiles)
                .reset_index(drop=False)
                .pivot(
                    index="datapoints",
                    columns=["response_vars", "cdf"],
                    values="centiles",
                )
            )
            centiles.columns = [("centiles", col) for col in centiles.columns]
            acc.append(centiles)
        return pd.concat(acc, axis=1)
