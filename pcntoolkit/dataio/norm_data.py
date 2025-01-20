"""
norm_data module
===============

This module provides functionalities for normalizing and converting different types of data into a NormData object.

The NormData object is an xarray.Dataset that contains the data, covariates, batch effects, and response variables, and it
is used by all the models in the toolkit.
"""

from __future__ import annotations

from functools import reduce

# pylint: disable=deprecated-class
from typing import (
    Any,
    Dict,
    Generator,
    Hashable,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    Union,
)

# pylint: enable=deprecated-class
import numpy as np
import pandas as pd  # type: ignore
import xarray as xr
from sklearn.model_selection import StratifiedKFold, train_test_split  # type: ignore

# import datavars from xarray
from xarray.core.types import DataVars

from pcntoolkit.util.output import Output, Warnings


class NormData(xr.Dataset):
    """A class for handling normative modeling data, extending xarray.Dataset.

    This class provides functionality for loading, preprocessing, and managing
    data for normative modeling. It supports various data formats and includes
    methods for data scaling, splitting, and visualization.

    Parameters
    ----------
    name : str
        The name of the dataset
    data_vars : DataVars
        Data variables for the dataset
    coords : Mapping[Any, Any]
        Coordinates for the dataset
    attrs : Mapping[Any, Any] | None, optional
        Additional attributes for the dataset, by default None

    Attributes
    ----------
    X : xr.DataArray
        Covariate data
    y : xr.DataArray
        Response variable data
    scaled_X : xr.DataArray
        Scaled version of covariate data
    scaled_y : xr.DataArray
        Scaled version of response variable data
    batch_effects : xr.DataArray
        Batch effect data
    Phi : xr.DataArray
        Design matrix
    scaled_centiles : xr.DataArray
        Scaled centile data (if applicable)
    centiles : xr.DataArray
        Unscaled centile data (if applicable)
    zscores : xr.DataArray
        Z-score data (if applicable)

    Notes
    -----
    This class stores both original and scaled versions of X and y data.
    While this approach offers convenience and transparency, it may
    increase memory usage. Consider memory constraints when working with
    large datasets.

    Examples
    --------
    >>> data = NormData.from_dataframe("my_data", df, covariates, batch_effects, response_vars)
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
        "unique_batch_effects",
        "batch_effects_counts",
    )

    def __init__(
        self,
        name: str,
        data_vars: DataVars,
        coords: Mapping[Any, Any],
        attrs: Mapping[Any, Any] | None = None,
    ) -> None:
        """
        Initialize a NormData object.

        Parameters
        ----------
        name : str
            The name of the dataset.
        data_vars : DataVars
            Data variables for the dataset.
        coords : Mapping[Any, Any]
            Coordinates for the dataset.
        attrs : Mapping[Any, Any] | None, optional
            Additional attributes for the dataset, by default None.
        """
        if attrs is None:
            attrs = {}
        attrs["name"] = name  # type: ignore
        super().__init__(data_vars=data_vars, coords=coords, attrs=attrs)
        self["batch_effects"] = self["batch_effects"].astype(str)
        self.register_unique_batch_effects()

    @classmethod
    def from_ndarrays(
        cls,
        name: str,
        X: np.ndarray,
        y: np.ndarray,
        batch_effects: Optional[np.ndarray] = None,
        attrs: Mapping[str, Any] | None = None,
    ) -> NormData:
        """Create a NormData object from numpy arrays.

        Parameters
        ----------
        name : str
            The name of the dataset
        X : np.ndarray
            Covariate data of shape (n_samples, n_features)
        y : np.ndarray
            Response variable data of shape (n_samples, n_responses)
        batch_effects : np.ndarray
            Batch effect data of shape (n_samples, n_batch_effects)
        attrs : Mapping[str, Any] | None, optional
            Additional attributes for the dataset, by default None

        Returns
        -------
        NormData
            A new NormData instance containing the provided data

        Notes
        -----
        Input arrays are automatically reshaped to 2D if they are 1D
        """
        if X.ndim == 1:
            X = X[:, None]
        if y.ndim == 1:
            y = y[:, None]
        if batch_effects is not None:
            if batch_effects.ndim == 1:
                batch_effects = batch_effects[:, None]
        else:
            batch_effects = np.zeros((X.shape[0], 1))
        return cls(
            name,
            {
                "X": (["datapoints", "covariates"], X),
                "y": (["datapoints", "response_vars"], y),
                "batch_effects": (["datapoints", "batch_effect_dims"], batch_effects),
            },
            coords={
                "datapoints": list(np.arange(X.shape[0])),
                "covariates": [f"covariate_{i}" for i in np.arange(X.shape[1])],
                "response_vars": [f"response_var_{i}" for i in np.arange(y.shape[1])],
                "batch_effect_dims": [f"batch_effect_{i}" for i in range(batch_effects.shape[1])],
            },
            attrs=attrs,
        )

    @classmethod
    def from_fsl(cls, fsl_folder, config_params) -> "NormData":  # type: ignore
        """
        Load a normative dataset from a FSL file.

        Parameters
        ----------
        fsl_folder : str
            Path to the FSL folder.
        config_params : dict
            Configuration parameters for loading the dataset.

        Returns
        -------
        NormData
            An instance of NormData.
        """

    @classmethod
    def from_nifti(cls, nifti_folder, config_params) -> "NormData":  # type: ignore
        """
        Load a normative dataset from a Nifti file.

        Parameters
        ----------
        nifti_folder : str
            Path to the Nifti folder.
        config_params : dict
            Configuration parameters for loading the dataset.

        Returns
        -------
        NormData
            An instance of NormData.
        """

    @classmethod
    def from_bids(cls, bids_folder, config_params) -> "NormData":  # type: ignore
        """
        Load a normative dataset from a BIDS dataset.

        Parameters
        ----------
        bids_folder : str
            Path to the BIDS folder.
        config_params : dict
            Configuration parameters for loading the dataset.

        Returns
        -------
        NormData
            An instance of NormData.
        """

    @classmethod
    def from_xarray(cls, name: str, xarray_dataset: xr.Dataset) -> NormData:
        """
        Load a normative dataset from an xarray dataset.

        Parameters
        ----------
        name : str
            The name of the dataset.
        xarray_dataset : xr.Dataset
            The xarray dataset to load.

        Returns
        -------
        NormData
            An instance of NormData.
        """
        return cls(
            name,
            xarray_dataset.data_vars,
            xarray_dataset.coords,
            xarray_dataset.attrs,
        )

    # pylint: disable=arguments-differ
    @classmethod
    def from_dataframe(  # type:ignore
        cls,
        name: str,
        dataframe: pd.DataFrame,
        covariates: List[str],
        batch_effects: List[str],
        response_vars: List[str],
        attrs: Mapping[str, Any] | None = None,
    ) -> NormData:
        """
        Load a normative dataset from a pandas DataFrame.

        Parameters
        ----------
        name : str
            The name of the dataset.
        dataframe : pd.DataFrame
            The pandas DataFrame to load.
        covariates : List[str]
            The list of column names to be used as covariates in the dataset.
        batch_effects : List[str]
            The list of column names to be used as batch effects in the dataset.
        response_vars : List[str]
            The list of column names to be used as response variables in the dataset.
        attrs : Mapping[str, Any] | None, optional
            Additional attributes for the dataset, by default None.

        Returns
        -------
        NormData
            An instance of NormData.
        """

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
                "datapoints": list(np.arange(dataframe[covariates].to_numpy().shape[0])),
                "response_vars": response_vars,
                "covariates": covariates,
                "batch_effect_dims": batch_effects,
            },
            attrs=attrs,
        )

    def merge(self, other: NormData) -> NormData:
        """
        Merge two NormData objects.
        """
        new_X = np.concatenate([self.X.values, other.X.values], axis=0)
        new_y = np.concatenate([self.y.values, other.y.values], axis=0)
        new_batch_effects = np.concatenate([self.batch_effects.values, other.batch_effects.values], axis=0)

        new_normdata = NormData(
            name=self.attrs["name"],
            data_vars={
                "X": (["datapoints", "covariates"], new_X),
                "y": (["datapoints", "response_vars"], new_y),
                "batch_effects": (
                    ["datapoints", "batch_effect_dims"],
                    new_batch_effects,
                ),
            },
            coords={
                "datapoints": list(np.arange(new_X.shape[0])),
                "response_vars": self.response_vars.to_numpy(),
                "covariates": self.covariates.to_numpy(),
                "batch_effect_dims": self.batch_effect_dims.to_numpy(),
            },
            attrs=self.attrs,
        )
        return new_normdata

    # pylint: enable=arguments-differ

    def create_synthetic_data(
        self,
        n_datapoints: int = 100,
        range_dim: Union[int, str] = 0,
        batch_effects_to_sample: Dict[str, List[Any]] | None = None,  # type: ignore
    ) -> NormData:
        """
        Create a synthetic dataset with the same dimensions as the original dataset.

        Parameters
        ----------
        n_datapoints : int, optional
            The number of datapoints to create, by default 100.
        range_dim : Union[int, str], optional
            The covariate to use for the range of values, by default 0.
        batch_effects_to_sample : Dict[str, List[Any]] | None, optional
            The batch effects to sample, by default None.

        Returns
        -------
        NormData
            A synthetic NormData instance.
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
                batch_effects_to_sample[dim] = [self.unique_batch_effects[dim][0]]

        # # Assert that the batch effects to sample are in the batch effects maps
        for dim, values in batch_effects_to_sample.items():
            assert dim in self.batch_effect_dims, f"{dim} is not a known batch effect dimension"
            assert len(values) > 0, f"No values provided for batch effect dimension {dim}"
            for value in values:
                assert value in self.unique_batch_effects[dim], f"{value} is not a known value for batch effect dimension {dim}"

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
        return to_return

    def get_single_batch_effect(self) -> Dict[str, List[str]]:
        """
        Get a single batch effect for each dimension.

        Returns
        -------
        Dict[str, List[str]]
            A dictionary mapping each batch effect dimension to a list containing a single value.
        """
        return {k: [v[0]] for k, v in self.unique_batch_effects.items()}

    def concatenate_string_arrays(self, *arrays: Any) -> np.ndarray:
        """
        Concatenate arrays of strings.

        Parameters
        ----------
        arrays : List[np.ndarray]
            A list of numpy arrays containing strings.

        Returns
        -------
        np.ndarray
            A single concatenated numpy array of strings.
        """
        return reduce(np.char.add, arrays)

    def chunk(self, n_chunks: int) -> Generator[NormData]:
        """
        Split the data into n_chunks with roughly equal number of response variables

        Parameters
        ----------
        n_chunks : int
            The number of chunks to split the data into.

        Returns
        -------
        Generator[NormData]
            A generator of NormData instances.
        """
        for i in range(n_chunks):
            yield self.isel(response_vars=slice(i, None, n_chunks))

    def train_test_split(
        self,
        splits: Tuple[float, ...] | List[float] | float = 0.8,
        split_names: Tuple[str, ...] | None = None,  # type: ignore
        random_state: int = 42,
    ) -> Tuple[NormData, ...]:
        """
        Split the data into training and testing datasets.

        Parameters
        ----------
        splits : Tuple[float, ...] | List[float] | float
            A tuple specifying the proportion of data for each split. Or a float specifying the proportion of data for the train set.
        split_names : Tuple[str, ...] | None, optional
            Names for the splits, by default None.
        random_state: int , optional
            Random state for splits, by default 42.

        Returns
        -------
        Tuple[NormData, ...]
            A tuple containing the training and testing NormData instances.
        """
        if isinstance(splits, float):
            splits = (splits, 1 - splits)
        elif isinstance(splits, list):
            splits = tuple(splits)
        assert isinstance(splits, tuple)
        batch_effects_stringified = self.concatenate_string_arrays(
            *[self.batch_effects[:, i].astype(str) for i in range(self.batch_effects.shape[1])]
        )
        train_idx, test_idx = train_test_split(
            np.arange(self.X.shape[0]),
            test_size=splits[1],
            random_state=random_state,
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

    def kfold_split(self, k: int) -> Generator[Tuple[NormData, NormData], Any, Any]:
        """
        Perform k-fold splitting of the data.

        Parameters
        ----------
        k : int
            The number of folds.

        Returns
        -------
        Generator[Tuple[NormData, NormData], Any, Any]
            A generator yielding training and testing NormData instances for each fold.
        """
        # Returns an iterator of (NormData, NormData) objects, split into k folds
        stratified_kfold_split = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)
        batch_effects_stringified = self.concatenate_string_arrays(
            *[self.batch_effects[:, i].astype(str) for i in range(self.batch_effects.shape[1])]
        )
        for train_idx, test_idx in stratified_kfold_split.split(self.X, batch_effects_stringified):
            split1 = self.isel(datapoints=train_idx)
            split2 = self.isel(datapoints=test_idx)
            yield split1, split2

    def register_unique_batch_effects(self) -> None:
        """
        Create a mapping of batch effects dims to unique values.
        """
        # create a dictionary with for each column in the batch effects, a dict from value to int
        self.attrs["unique_batch_effects"] = {}
        for i, dim in enumerate(self.batch_effect_dims.to_numpy()):
            self.attrs["unique_batch_effects"][dim] = list(np.unique(self.batch_effects[:, i]))
        self.attrs["batch_effects_counts"] = {}
        for i, dim in enumerate(self.batch_effect_dims.to_numpy()):
            self.attrs["batch_effects_counts"][dim] = {
                k: v for k, v in zip(*np.unique(self.batch_effects.values[:, i], return_counts=True))
            }

    def check_compatibility(self, other: NormData) -> bool:
        """
        Check if the data is compatible with another dataset.

        Parameters
        ----------
        other : NormData
            Another NormData instance to compare with.

        Returns
        -------
        bool
            True if compatible, False otherwise
        """
        missing_covariates = [i for i in other.covariates.values if i not in self.covariates.values]
        if len(missing_covariates) > 0:
            Output.warning(
                Warnings.MISSING_COVARIATES,
                dataset_name=self.name,
                covariates=missing_covariates,
            )

        extra_covariates = [i for i in self.covariates.values if i not in other.covariates.values]
        if len(extra_covariates) > 0:
            Output.warning(
                Warnings.EXTRA_COVARIATES,
                dataset_name=self.name,
                covariates=extra_covariates,
            )

        extra_response_vars = [i for i in self.response_vars.values if i not in other.response_vars.values]
        if len(extra_response_vars) > 0:
            Output.warning(
                Warnings.EXTRA_RESPONSE_VARS,
                dataset_name=self.name,
                response_vars=extra_response_vars,
            )
        if len(missing_covariates) > 0 or len(extra_covariates) > 0 or len(extra_response_vars) > 0:
            return False
        return True

    def scale_forward(self, inscalers: Dict[str, Any], outscaler: Dict[str, Any]) -> None:
        """
        Scale the data forward using provided scalers.

        Parameters
        ----------
        inscalers : Dict[str, Any]
            Scalers for the covariate data.
        outscaler : Dict[str, Any]
            Scalers for the response variable data.
        """
        # Scale X column-wise using the inscalers
        self["scaled_X"] = xr.DataArray(
            np.zeros(self.X.shape),
            coords=self.X.coords,
            dims=self.X.dims,
            attrs=self.X.attrs,
        )
        for covariate in self.covariates.to_numpy():
            self.scaled_X.loc[:, covariate] = inscalers[covariate].transform(self.X.sel(covariates=covariate).data)

        # Scale y column-wise using the outscalers
        self["scaled_y"] = xr.DataArray(
            np.zeros(self.y.shape),
            coords=self.y.coords,
            dims=self.y.dims,
            attrs=self.y.attrs,
        )
        for responsevar in self.response_vars.to_numpy():
            self.scaled_y.loc[:, responsevar] = outscaler[responsevar].transform(self.y.sel(response_vars=responsevar).data)

    def scale_backward(self, inscalers: Dict[str, Any], outscalers: Dict[str, Any]) -> None:
        """
        Scale the data backward using provided scalers.

        Parameters
        ----------
        inscalers : Dict[str, Any]
            Scalers for the covariate data.
        outscalers : Dict[str, Any]
            Scalers for the response variable data.
        """
        # Scale X column-wise using the inscalers
        self["X"] = xr.DataArray(
            np.zeros(self.scaled_X.shape),
            coords=self.scaled_X.coords,
            dims=self.scaled_X.dims,
            attrs=self.scaled_X.attrs,
        )
        for covariate in self.covariates.to_numpy():
            self.X.loc[:, covariate] = inscalers[covariate].inverse_transform(self.scaled_X.sel(covariates=covariate).data)

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
                self.centiles.loc[{"response_vars": responsevar}] = outscalers[responsevar].inverse_transform(
                    self.scaled_centiles.sel(response_vars=responsevar).data
                )

    def split_batch_effects(
        self,
        batch_effects: Dict[str, List[str]],
        names: Tuple[str, str] | None = None,
    ) -> Tuple[NormData, NormData]:
        """
        Split the data into two datasets, one with the specified batch effects and one without.
        """
        A = self.select_batch_effects(batch_effects)
        B = self.select_batch_effects(batch_effects, invert=True)
        if names is not None:
            A.name = names[0]
            B.name = names[1]
        return A, B

    def select_batch_effects(self, batch_effects: Dict[str, List[str]], invert: bool = False) -> NormData:
        """
        Select only the specified batch effects.

        Parameters
        ----------
        batch_effects : Dict[str, List[str]]
            A dictionary specifying which batch effects to select.

        Returns
        -------
        NormData
            A NormData instance with the selected batch effects.
        """
        mask = np.zeros(self.batch_effects.shape[0], dtype=bool)
        for key, values in batch_effects.items():
            this_batch_effect = self.batch_effects.sel(batch_effect_dims=key)
            for value in values:
                mask = np.logical_or(mask, this_batch_effect == value)
        if invert:
            mask = ~mask

        to_return = self.where(mask).dropna(dim="datapoints", how="all")
        if isinstance(to_return, xr.Dataset):
            to_return = NormData.from_xarray(f"{self.attrs['name']}_selected", to_return)
        to_return.register_unique_batch_effects()
        return to_return

    def to_dataframe(self, dim_order: Sequence[Hashable] | None = None) -> pd.DataFrame:
        """
        Convert the NormData instance to a pandas DataFrame.

        Parameters
        ----------
        dim_order : Sequence[Hashable] | None, optional
            The order of dimensions for the DataFrame, by default None.

        Returns
        -------
        pd.DataFrame
            A DataFrame representation of the NormData instance.
        """
        acc = []
        x_columns = [col for col in ["X", "scaled_X"] if hasattr(self, col)]
        y_columns = [col for col in ["y", "zscores", "scaled_y"] if hasattr(self, col)]
        acc.append(
            xr.Dataset.to_dataframe(self[x_columns], dim_order)
            .reset_index(drop=False)
            .pivot(index="datapoints", columns="covariates", values=x_columns)
        )
        acc.append(
            xr.Dataset.to_dataframe(self[y_columns], dim_order)
            .reset_index(drop=False)
            .pivot(index="datapoints", columns="response_vars", values=y_columns)
        )
        be = (
            xr.DataArray.to_dataframe(self.batch_effects, dim_order)
            .reset_index(drop=False)
            .pivot(index="datapoints", columns="batch_effect_dims", values="batch_effects")
        )
        be.columns = [("batch_effects", col) for col in be.columns]
        acc.append(be)
        if hasattr(self, "Phi"):
            phi = (
                xr.DataArray.to_dataframe(self.Phi, dim_order)
                .reset_index(drop=False)
                .pivot(index="datapoints", columns="basis_functions", values="Phi")
            )
            phi.columns = [("Phi", col) for col in phi.columns]
            acc.append(phi)
        if hasattr(self, "centiles"):
            centiles = (
                xr.DataArray.to_dataframe(self.centiles, dim_order)
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

    def create_measures_group(self) -> None:
        """
        Initializes a DataArray for measures with NaN values.

        This method creates a DataArray with dimensions 'response_vars' and 'statistics',
        where 'response_vars' corresponds to the response variables in the dataset,
        and 'statistics' includes measures such as Rho, RMSE, SMSE, ExpV, NLL, and ShapiroW.
        The DataArray is filled with NaN values initially.
        """
        rv = self.response_vars.to_numpy().copy().tolist()

        self["measures"] = xr.DataArray(
            np.nan * np.ones((len(rv), 6)),
            dims=("response_vars", "statistics"),
            coords={
                "response_vars": np.arange(len(rv)),
                "statistics": ["Rho", "RMSE", "SMSE", "ExpV", "NLL", "ShapiroW"],
            },
        )

    @property
    def name(self) -> str:
        """
        Get the name of the dataset.

        Returns
        -------
        str
            The name of the dataset.
        """
        return self.attrs["name"]

    @name.setter
    def name(self, name: str) -> None:
        self.attrs["name"] = name

    @property
    def response_var_list(self) -> xr.DataArray:
        return self.response_vars.to_numpy().copy().tolist()
