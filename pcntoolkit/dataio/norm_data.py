"""
norm_data module
===============

This module provides functionalities for normalizing and converting different types of data into a NormData object.

The NormData object is an xarray.Dataset that contains the data, covariates, batch effects, and response variables, and it
is used by all the models in the toolkit.
"""

from __future__ import annotations

import copy
import os
from functools import reduce

# pylint: disable=deprecated-class
from typing import (
    Any,
    Dict,
    Generator,
    Hashable,
    List,
    LiteralString,
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

from pcntoolkit.util.output import Messages, Output, Warnings


class NormData(xr.Dataset):
    """A class for handling normative modeling data, extending xarray.Dataset.

    This class provides functionality for loading data for normative modeling.
    It supports various data formats.

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
    batch_effects : xr.DataArray
        Batch effect data
    Z: xr.DataArray
        Z-score data
    centiles: xr.DataArray
        Centile data


    Examples
    --------
    >>> data = NormData.from_dataframe("my_data", df, covariates, batch_effects, response_vars)
    >>> train_data, test_data = data.train_test_split([0.8, 0.2])
    """

    __slots__ = (
        "unique_batch_effects",
        "batch_effect_counts",
        "batch_effect_covariate_ranges",
        "covariate_ranges",
        "real_ids",
        "thrive_covariate"  # Whether the ids are real or synthetic
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
        attrs["is_scaled"] = False  # type: ignore
        attrs["name"] = name  # type: ignore
        super().__init__(data_vars=data_vars, coords=coords, attrs=attrs)
        self["batch_effects"] = self["batch_effects"].astype(str)
        attrs["real_ids"] = attrs.get("real_ids", False)
        self.register_batch_effects()
        be_str = (
            "\t" + ("".join([f"\t{be} ({len(self.unique_batch_effects[be])})\n" for be in self.unique_batch_effects])).strip()
        )
        Output.print(
            Messages.DATASET_CREATED,
            name=name,
            n_subjects=len(np.unique(self.subjects)),
            n_observations=len(self.observations),
            n_covariates=len(self.covariates),
            n_response_vars=len(self.response_vars),
            n_batch_effects=len(self.unique_batch_effects),
            batch_effects=be_str,
        )

    @classmethod
    def from_ndarrays(
        cls,
        name: str,
        X: np.ndarray | None = None,
        Y: np.ndarray | None = None,
        batch_effects: np.ndarray | None = None,
        attrs: Mapping[str, Any] | None = None,
        subject_ids: np.ndarray | None = None,
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
        data_vars = {}
        coords = {}
        attrs = attrs or {}
        if subject_ids is not None:
            attrs["real_ids"] = True
            data_vars["subjects"] = (["observations"], subject_ids)
        else:
            attrs["real_ids"] = False
            data_vars["subjects"] = (["observations"], list(np.arange(X.shape[0])))

        coords["observations"] = list(np.arange(X.shape[0]))
        lengths = []
        if X is not None:
            lengths.append(X.shape[0])
            if X.ndim == 1:
                X = X[:, None]
            data_vars["X"] = (["observations", "covariates"], X)
            coords["covariates"] = [f"covariate_{i}" for i in np.arange(X.shape[1])]
        if Y is not None:
            lengths.append(Y.shape[0])
            if Y.ndim == 1:
                Y = Y[:, None]
            data_vars["Y"] = (["observations", "response_vars"], Y)
            coords["response_vars"] = [f"response_var_{i}" for i in np.arange(Y.shape[1])]
        if batch_effects is not None:
            lengths.append(batch_effects.shape[0])
            if batch_effects.ndim == 1:
                batch_effects = batch_effects[:, None]
            data_vars["batch_effects"] = (["observations", "batch_effect_dims"], batch_effects)
            coords["batch_effect_dims"] = [f"batch_effect_{i}" for i in range(batch_effects.shape[1])]
        else:
            data_vars["batch_effects"] = (["observations", "batch_effect_dims"], np.zeros((lengths[0], 1)))
            coords["batch_effect_dims"] = ["dummy_batch_effect"]
        assert len(set(lengths)) == 1, "All arrays must have the same number of observations"
        return cls(name, data_vars, coords, attrs)

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
        covariates: List[str] | None = None,
        batch_effects: List[str] | None = None,
        response_vars: List[str | LiteralString] | None = None,
        subject_ids: str | None = None,
        remove_Nan: bool = False,
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
        subject_ids: str
            The name of the column containing the subject IDs
        attrs : Mapping[str, Any] | None, optional
            Additional attributes for the dataset, by default None.
        remove_Nan: bool
            Wheter or not to remove NAN values from the dataframe before creationg of the class object. By default False

        Returns
        -------
        NormData
            An instance of NormData.
        """

        if remove_Nan:
            cols_to_check = []
            if covariates:
                cols_to_check += covariates
            if response_vars:
                cols_to_check += response_vars
            if batch_effects:
                cols_to_check += batch_effects
            dataframe = dataframe.dropna(subset=cols_to_check)
        else:
            Output.warning(Warnings.REMOVE_NAN_SET_TO_FALSE)

        data_vars = {}
        coords = {}
        attrs = attrs or {}

        if subject_ids is not None:
            attrs["real_ids"] = True
            data_vars["subjects"] = (["observations"], dataframe[subject_ids].to_numpy())
        else:
            attrs["real_ids"] = False
            data_vars["subjects"] = (["observations"], list(np.arange(len(dataframe))))

        coords["observations"] = list(np.arange(len(dataframe)))

        if response_vars is not None and len(response_vars) > 0:
            for respvar in response_vars:
                if respvar not in dataframe.columns:
                    dataframe[respvar] = np.nan
            data_vars["Y"] = (["observations", "response_vars"], dataframe[response_vars].to_numpy())
            coords["response_vars"] = response_vars

        if covariates is not None and len(covariates) > 0:
            data_vars["X"] = (["observations", "covariates"], dataframe[covariates].to_numpy())
            coords["covariates"] = covariates

        if batch_effects is not None and len(batch_effects) > 0:
            data_vars["batch_effects"] = (["observations", "batch_effect_dims"], dataframe[batch_effects].to_numpy())
            coords["batch_effect_dims"] = batch_effects
        else:
            # Initialize batch effects as zeros
            data_vars["batch_effects"] = (["observations", "batch_effect_dims"], np.zeros((len(dataframe), 1)))
            coords["batch_effect_dims"] = ["dummy_batch_effect"]

        return cls(
            name,
            data_vars,
            coords,
            attrs,
        )

    def merge(self, other: NormData) -> NormData:
        """
        Merge two NormData objects.

        Drops all columns that are not present in both datasets.
        """

        new_data_vars = {}
        new_coords = {}

        if self.attrs["real_ids"] or other.attrs["real_ids"]:
            new_data_vars["subjects"] = (
                ["observations"],
                list(np.concatenate([self.subjects.to_numpy(), other.subjects.to_numpy()])),
            )
        else:
            new_data_vars["subjects"] = (["observations"], list(np.arange(self.X.shape[0] + other.X.shape[0])))

        new_coords["observations"] = list(np.arange(self.X.shape[0] + other.X.shape[0]))
        covar_intersection = list(set(self.covariates.to_numpy()) & set(other.covariates.to_numpy()))
        respvar_intersection = list(set(self.response_vars.to_numpy()) & set(other.response_vars.to_numpy()))
        batch_effect_dims_intersection = list(set(self.batch_effect_dims.to_numpy()) & set(other.batch_effect_dims.to_numpy()))
        new_coords["covariates"] = covar_intersection
        new_coords["response_vars"] = respvar_intersection
        new_coords["batch_effect_dims"] = batch_effect_dims_intersection

        if hasattr(self, "X") and hasattr(other, "X"):
            new_X = xr.DataArray(
                np.zeros((self.X.shape[0] + other.X.shape[0], len(covar_intersection))),
                dims=["observations", "covariates"],
                coords={"covariates": covar_intersection, "observations": new_coords["observations"]},
            )
            for covar in covar_intersection:
                new_X.loc[{"covariates": covar}] = np.concatenate(
                    [self.X.sel(covariates=covar).values, other.X.sel(covariates=covar).values], axis=0
                )
            new_data_vars["X"] = (["observations", "covariates"], new_X.data)

        if hasattr(self, "Y") and hasattr(other, "Y"):
            new_Y = xr.DataArray(
                np.zeros((new_X.shape[0], len(respvar_intersection))),
                dims=["observations", "response_vars"],
                coords={"response_vars": respvar_intersection, "observations": new_coords["observations"]},
            )
            for respvar in respvar_intersection:
                new_Y.loc[{"response_vars": respvar}] = np.concatenate(
                    [self.Y.sel(response_vars=respvar).values, other.Y.sel(response_vars=respvar).values], axis=0
                )
            new_data_vars["Y"] = (["observations", "response_vars"], new_Y.data)

        if hasattr(self, "Y_harmonized") and hasattr(other, "Y_harmonized"):
            new_Y_harmonized = xr.DataArray(
                np.zeros((new_X.shape[0], len(respvar_intersection))),
                dims=["observations", "response_vars"],
                coords={"response_vars": respvar_intersection, "observations": new_coords["observations"]},
            )
            for respvar in respvar_intersection:
                new_Y_harmonized.loc[{"response_vars": respvar}] = np.concatenate(
                    [self.Y_harmonized.sel(response_vars=respvar).values, other.Y_harmonized.sel(response_vars=respvar).values],
                    axis=0,
                )
            new_data_vars["Y_harmonized"] = (["observations", "response_vars"], new_Y_harmonized.data)

        if hasattr(self, "Z") and hasattr(other, "Z"):
            new_Z = xr.DataArray(
                np.zeros((new_X.shape[0], len(respvar_intersection))),
                dims=["observations", "response_vars"],
                coords={"response_vars": respvar_intersection, "observations": new_coords["observations"]},
            )
            for respvar in respvar_intersection:
                new_Z.loc[{"response_vars": respvar}] = np.concatenate(
                    [self.Z.sel(response_vars=respvar).values, other.Z.sel(response_vars=respvar).values], axis=0
                )
            new_data_vars["Z"] = (["observations", "response_vars"], new_Z.data)

        if hasattr(self, "centiles") and hasattr(other, "centiles"):
            if self.centile.to_numpy() == other.centile.to_numpy():
                new_centiles = xr.DataArray(
                    np.zeros((new_X.shape[0], len(respvar_intersection), len(self.centile.to_numpy()))),
                    dims=["observations", "response_vars", "centile"],
                    coords={
                        "response_vars": respvar_intersection,
                        "centile": self.centile.to_numpy(),
                        "observations": new_coords["observations"],
                    },
                )
                for respvar in respvar_intersection:
                    for centile in self.centile.to_numpy():
                        new_centiles.loc[{"response_vars": respvar, "centile": centile}] = np.concatenate(
                            [
                                self.centiles.sel(response_vars=respvar, centile=centile).values,
                                other.centiles.sel(response_vars=respvar, centile=centile).values,
                            ],
                            axis=0,
                        )
                new_data_vars["centiles"] = (["observations", "response_vars", "centile"], new_centiles.data)
                new_coords["centile"] = self.centile.to_numpy()

        if hasattr(self, "batch_effects") and hasattr(other, "batch_effects"):
            new_batch_effects = xr.DataArray(
                np.zeros((new_X.shape[0], len(batch_effect_dims_intersection))).astype(str),
                dims=["observations", "batch_effect_dims"],
                coords={"batch_effect_dims": batch_effect_dims_intersection, "observations": new_coords["observations"]},
            )
            for batch_effect_dim in batch_effect_dims_intersection:
                new_batch_effects.loc[{"batch_effect_dims": batch_effect_dim}] = np.concatenate(
                    [
                        self.batch_effects.sel(batch_effect_dims=batch_effect_dim).values,
                        other.batch_effects.sel(batch_effect_dims=batch_effect_dim).values,
                    ],
                    axis=0,
                )
            new_data_vars["batch_effects"] = (["observations", "batch_effect_dims"], new_batch_effects.data)

        new_normdata = NormData(
            name=self.attrs["name"],
            data_vars=new_data_vars,
            coords=new_coords,
            attrs=self.attrs,
        )
        return new_normdata

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
            A tuple (train_size, test_size), specifying the proportion of data for each split. Or a float specifying the proportion of data for the train set.
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
        assert all([isinstance(i, float) for i in splits]), "Splits must be a list of floats"
        assert sum(list(splits)) == 1, "Splits must sum to 1"
        assert len(splits) > 1, "Splits must contain at least two elements"
        batch_effects_stringified = self.concatenate_string_arrays(
            *[self.batch_effects[:, i].astype(str) for i in range(self.batch_effects.shape[1])]
        )
        train_idx, test_idx = train_test_split(
            np.arange(self.X.shape[0]),
            test_size=splits[1],
            random_state=random_state,
            stratify=batch_effects_stringified,
        )
        split1 = self.isel(observations=train_idx)
        split1.attrs = copy.deepcopy(self.attrs)
        split2 = self.isel(observations=test_idx)
        split2.attrs = copy.deepcopy(self.attrs)
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
            split1 = copy.deepcopy(self.isel(observations=train_idx))
            split2 = copy.deepcopy(self.isel(observations=test_idx))
            yield split1, split2

    def register_batch_effects(self) -> None:
        """
        Create a mapping of batch effects to unique values.
        """
        my_be: xr.DataArray = self.batch_effects
        # create a dictionary with for each column in the batch effects, a dict from value to int
        self.attrs["unique_batch_effects"] = {}
        self.attrs["batch_effect_counts"] = {}
        self.attrs["batch_effect_covariate_ranges"] = {}
        for dim in self.batch_effect_dims.to_numpy():
            dim_subset = my_be.sel(batch_effect_dims=dim)
            uniques, counts = np.unique(dim_subset, return_counts=True)
            self.attrs["unique_batch_effects"][dim] = list(uniques)
            self.attrs["batch_effect_counts"][dim] = {k: v for k, v in zip(uniques, counts)}
            self.attrs["batch_effect_covariate_ranges"][dim] = {}
            if self.X is not None:
                for u in uniques:
                    self.attrs["batch_effect_covariate_ranges"][dim][u] = {}
                    for c in self.covariates.to_numpy():
                        u_mask = dim_subset.values == u
                        my_c = self.X.sel(covariates=c).values[u_mask]
                        my_min = my_c.min()
                        my_max = my_c.max()
                        self.attrs["batch_effect_covariate_ranges"][dim][u][c] = {"min": my_min, "max": my_max}
        self.attrs["covariate_ranges"] = {}
        for c in self.covariates.to_numpy():
            my_c = self.X.sel(covariates=c).values
            my_min = my_c.min()
            my_max = my_c.max()
            self.attrs["covariate_ranges"][c] = {"min": my_min, "max": my_max}

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

    def scale_forward(self, inscalers: Dict[str, Any], outscalers: Dict[str, Any]) -> None:
        """
        Scale the data forward in-place using provided scalers.

        Parameters
        ----------
        inscalers : Dict[str, Any]
            Scalers for the covariate data.
        outscaler : Dict[str, Any]
            Scalers for the response variable data.
        """
        if not self.attrs["is_scaled"]:
            # Scale X column-wise using the inscalers
            if "X" in self.data_vars:
                scaled_X = np.zeros(self.X.shape)
                for i, covariate in enumerate(self.covariates.to_numpy()):
                    scaled_X[:, i] = inscalers[covariate].transform(self.X.sel(covariates=covariate).data)
                self["X"] = xr.DataArray(
                    scaled_X,
                    coords=self.X.coords,
                    dims=self.X.dims,
                    attrs=self.X.attrs,
                )
            if "thrive_X" in self.data_vars:
                self["thrive_X"] = xr.DataArray(
                    inscalers[self.attrs['thrive_covariate']].transform(self.thrive_X.data),
                    coords=self.thrive_X.coords,
                    dims=self.thrive_X.dims,
                    attrs=self.thrive_X.attrs,
                )    
            # Scale y column-wise using the outscalers
            if "Y" in self.data_vars:
                scaled_y = np.zeros(self.Y.shape)
                for i, responsevar in enumerate(self.response_vars.to_numpy()):
                    scaled_y[:, i] = outscalers[responsevar].transform(self.Y.sel(response_vars=responsevar).data)
                self["Y"] = xr.DataArray(
                    scaled_y,
                    coords=self.Y.coords,
                    dims=self.Y.dims,
                    attrs=self.Y.attrs,
                )
            if "Y_harmonized" in self.data_vars:
                scaled_Y_harmonized = np.zeros(self.Y_harmonized.shape)
                for i, responsevar in enumerate(self.response_vars.to_numpy()):
                    scaled_Y_harmonized[:, i] = outscalers[responsevar].transform(
                        self.Y_harmonized.sel(response_vars=responsevar).data
                    )
                self["Y_harmonized"] = xr.DataArray(
                    scaled_Y_harmonized,
                    coords=self.Y_harmonized.coords,
                    dims=self.Y_harmonized.dims,
                    attrs=self.Y_harmonized.attrs,
                )
            if "centiles" in self.data_vars:
                scaled_centiles = np.zeros(self.centiles.shape)
                for i, responsevar in enumerate(self.response_vars.to_numpy()):
                    scaled_centiles[:, :, i] = outscalers[responsevar].transform(self.centiles.sel(response_vars=responsevar).data)
                self["centiles"] = xr.DataArray(
                    scaled_centiles,
                    coords=self.centiles.coords,
                    dims=self.centiles.dims,
                    attrs=self.centiles.attrs,
                )
            if "thrive_Y" in self.data_vars:
                scaled_thrive_Y = np.zeros(self.thrive_Y.shape)
                for i, responsevar in enumerate(self.response_vars.to_numpy()):
                    scaled_thrive_Y[:, i,:] = outscalers[responsevar].transform(
                        self.thrive_Y.sel(response_vars=responsevar).data
                    )
                self["thrive_Y"] = xr.DataArray(
                    scaled_thrive_Y,
                    coords=self.thrive_Y.coords,
                    dims=self.thrive_Y.dims,
                    attrs=self.thrive_Y.attrs,
                )
            self.attrs["is_scaled"] = True

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
        if self.attrs["is_scaled"]:
            if "X" in self.data_vars:
                unscaled_X = np.zeros(self.X.shape)
                for i, covariate in enumerate(self.covariates.to_numpy()):
                    unscaled_X[:, i] = inscalers[covariate].inverse_transform(self.X.sel(covariates=covariate).data)
                self["X"] = xr.DataArray(
                    unscaled_X,
                    coords=self.X.coords,
                    dims=self.X.dims,
                    attrs=self.X.attrs,
                )
            if "thrive_X" in self.data_vars:
                self["thrive_X"] = xr.DataArray(
                    inscalers[self.attrs['thrive_covariate']].inverse_transform(self.thrive_X.data),
                    coords=self.thrive_X.coords,
                    dims=self.thrive_X.dims,
                    attrs=self.thrive_X.attrs,
                )    
            if "Y" in self.data_vars:
                unscaled_y = np.zeros(self.Y.shape)
                for i, responsevar in enumerate(self.response_vars.to_numpy()):
                    unscaled_y[:, i] = outscalers[responsevar].inverse_transform(self.Y.sel(response_vars=responsevar).data)
                self["Y"] = xr.DataArray(
                    unscaled_y,
                    coords=self.Y.coords,
                    dims=self.Y.dims,
                    attrs=self.Y.attrs,
                )

            if "Y_harmonized" in self.data_vars:
                unscaled_Y_harmonized = np.zeros(self.Y_harmonized.shape)
                for i, responsevar in enumerate(self.response_vars.to_numpy()):
                    unscaled_Y_harmonized[:, i] = outscalers[responsevar].inverse_transform(
                        self.Y_harmonized.sel(response_vars=responsevar).data
                    )
                self["Y_harmonized"] = xr.DataArray(
                    unscaled_Y_harmonized,
                    coords=self.Y_harmonized.coords,
                    dims=self.Y_harmonized.dims,
                    attrs=self.Y_harmonized.attrs,
                )

            if "centiles" in self.data_vars:
                unscaled_centiles = np.zeros(self.centiles.shape)
                for i, responsevar in enumerate(self.response_vars.to_numpy()):
                    unscaled_centiles[:, :, i] = outscalers[responsevar].inverse_transform(
                        self.centiles.sel(response_vars=responsevar).data
                    )
                self["centiles"] = xr.DataArray(
                    unscaled_centiles,
                    coords=self.centiles.coords,
                    dims=self.centiles.dims,
                    attrs=self.centiles.attrs,
                )

            if "thrive_Y" in self.data_vars:
                unscaled_thrive_Y = np.zeros(self.thrive_Y.shape)
                for i, responsevar in enumerate(self.response_vars.to_numpy()):
                    unscaled_thrive_Y[:, i,:] = outscalers[responsevar].inverse_transform(
                        self.thrive_Y.sel(response_vars=responsevar).data
                    )
                self["thrive_Y"] = xr.DataArray(
                    unscaled_thrive_Y,
                    coords=self.thrive_Y.coords,
                    dims=self.thrive_Y.dims,
                    attrs=self.thrive_Y.attrs,
                )

            self.attrs["is_scaled"] = False

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

        to_return = self.where(mask).dropna(dim="observations", how="all")
        if isinstance(to_return, xr.Dataset):
            if invert:
                to_return = NormData.from_xarray(f"{self.attrs['name']}_not_selected", to_return)
            else:
                to_return = NormData.from_xarray(f"{self.attrs['name']}_selected", to_return)
        to_return.register_batch_effects()
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
        x_columns = [col for col in ["X"] if hasattr(self, col)]
        y_columns = [col for col in ["Y", "Y_harmonized", "Z"] if hasattr(self, col)]
        acc.append(
            xr.Dataset.to_dataframe(self[x_columns], dim_order)
            .reset_index(drop=False)
            .pivot(index="observations", columns="covariates", values=x_columns)
        )
        acc.append(
            xr.Dataset.to_dataframe(self[y_columns], dim_order)
            .reset_index(drop=False)
            .pivot(index="observations", columns="response_vars", values=y_columns)
        )
        be = (
            xr.DataArray.to_dataframe(self.batch_effects, dim_order)
            .reset_index(drop=False)
            .pivot(index="observations", columns="batch_effect_dims", values="batch_effects")
        )
        be.columns = [("batch_effects", col) for col in be.columns]

        acc.append(be)

        subjects = xr.DataArray.to_dataframe(self.subjects, dim_order)[["subjects"]]
        subjects.columns = [
            ("subjects", "subjects"),
        ]
        acc.append(subjects)
        if hasattr(self, "centiles"):
            centiles = (
                xr.DataArray.to_dataframe(self.centiles, dim_order)
                .reset_index(drop=False)
                .pivot(
                    index="observations",
                    columns=["response_vars", "centile"],
                    values="centiles",
                )
            )
            centiles.columns = [("centiles", col) for col in centiles.columns]
            acc.append(centiles)
        pandas_df = pd.concat(acc, axis=1)
        pandas_df.index = self.observations.values.astype(str)
        return pandas_df

    def create_statistics_group(self) -> None:
        """
        Initializes a DataArray for statistics with NaN values.

        This method creates a DataArray with dimensions 'response_vars' and 'statistics',
        where 'response_vars' corresponds to the response variables in the dataset,
        and 'statistics' includes statistics such as Rho, RMSE, SMSE, ExpV, NLL, and ShapiroW.
        The DataArray is filled with NaN values initially.
        """
        rv = self.response_vars.to_numpy().copy().tolist()

        self["statistics"] = xr.DataArray(
            np.nan * np.ones((len(rv), 6)),
            dims=("response_vars", "statistics"),
            coords={
                "response_vars": np.arange(len(rv)),
                "statistics": ["Rho", "RMSE", "SMSE", "ExpV", "NLL", "ShapiroW"],
            },
        )

    def get_statistics_df(self) -> pd.DataFrame:
        """
        Get the statistics as a pandas DataFrame.
        """
        assert "statistics" in self.data_vars, "Measures DataArray not found"
        statistics_df: pd.DataFrame = (
            self.statistics.to_dataframe()
            .reset_index()
            .pivot(index=["response_vars"], columns=["statistic"], values="statistics")
            .round(2)
        )
        return statistics_df

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




def load_fcon1000():
    resource_dir = "pcntoolkit_resources"
    os.makedirs(os.path.join(resource_dir, "data"), exist_ok=True)
    if not os.path.exists(os.path.join(resource_dir, "data/fcon1000.csv")):
        pd.read_csv(
            "https://raw.githubusercontent.com/predictive-clinical-neuroscience/PCNtoolkit-demo/refs/heads/main/data/fcon1000.csv"
        ).to_csv(os.path.join(resource_dir, "data/fcon1000.csv"), index=False)

    data = pd.read_csv(os.path.join(resource_dir, "data/fcon1000.csv"))
    subject_ids = "sub_id"
    covariates = ["age"]
    batch_effects = ["sex", "site"]
    response_vars = [
        "lh_G&S_frontomargin_thickness",
        "lh_G&S_occipital_inf_thickness",
        "lh_G&S_paracentral_thickness",
        "lh_G&S_subcentral_thickness",
        "lh_G&S_transv_frontopol_thickness",
        "lh_G&S_cingul-Ant_thickness",
        "lh_G&S_cingul-Mid-Ant_thickness",
        "lh_G&S_cingul-Mid-Post_thickness",
        "lh_G_cingul-Post-dorsal_thickness",
        "lh_G_cingul-Post-ventral_thickness",
        "lh_G_cuneus_thickness",
        "lh_G_front_inf-Opercular_thickness",
        "lh_G_front_inf-Orbital_thickness",
        "lh_G_front_inf-Triangul_thickness",
        "lh_G_front_middle_thickness",
        "lh_G_front_sup_thickness",
        "lh_G_Ins_lg&S_cent_ins_thickness",
        "lh_G_insular_short_thickness",
        "lh_G_occipital_middle_thickness",
        "lh_G_occipital_sup_thickness",
        "lh_G_oc-temp_lat-fusifor_thickness",
        "lh_G_oc-temp_med-Lingual_thickness",
        "lh_G_oc-temp_med-Parahip_thickness",
        "lh_G_orbital_thickness",
        "lh_G_pariet_inf-Angular_thickness",
        "lh_G_pariet_inf-Supramar_thickness",
        "lh_G_parietal_sup_thickness",
        "lh_G_postcentral_thickness",
        "lh_G_precentral_thickness",
        "lh_G_precuneus_thickness",
        "lh_G_rectus_thickness",
        "lh_G_subcallosal_thickness",
        "lh_G_temp_sup-G_T_transv_thickness",
        "lh_G_temp_sup-Lateral_thickness",
        "lh_G_temp_sup-Plan_polar_thickness",
        "lh_G_temp_sup-Plan_tempo_thickness",
        "lh_G_temporal_inf_thickness",
        "lh_G_temporal_middle_thickness",
        "lh_Lat_Fis-ant-Horizont_thickness",
        "lh_Lat_Fis-ant-Vertical_thickness",
        "lh_Lat_Fis-post_thickness",
        "lh_Pole_occipital_thickness",
        "lh_Pole_temporal_thickness",
        "lh_S_calcarine_thickness",
        "lh_S_central_thickness",
        "lh_S_cingul-Marginalis_thickness",
        "lh_S_circular_insula_ant_thickness",
        "lh_S_circular_insula_inf_thickness",
        "lh_S_circular_insula_sup_thickness",
        "lh_S_collat_transv_ant_thickness",
        "lh_S_collat_transv_post_thickness",
        "lh_S_front_inf_thickness",
        "lh_S_front_middle_thickness",
        "lh_S_front_sup_thickness",
        "lh_S_interm_prim-Jensen_thickness",
        "lh_S_intrapariet&P_trans_thickness",
        "lh_S_oc_middle&Lunatus_thickness",
        "lh_S_oc_sup&transversal_thickness",
        "lh_S_occipital_ant_thickness",
        "lh_S_oc-temp_lat_thickness",
        "lh_S_oc-temp_med&Lingual_thickness",
        "lh_S_orbital_lateral_thickness",
        "lh_S_orbital_med-olfact_thickness",
        "lh_S_orbital-H_Shaped_thickness",
        "lh_S_parieto_occipital_thickness",
        "lh_S_pericallosal_thickness",
        "lh_S_postcentral_thickness",
        "lh_S_precentral-inf-part_thickness",
        "lh_S_precentral-sup-part_thickness",
        "lh_S_suborbital_thickness",
        "lh_S_subparietal_thickness",
        "lh_S_temporal_inf_thickness",
        "lh_S_temporal_sup_thickness",
        "lh_S_temporal_transverse_thickness",
        "lh_MeanThickness_thickness",
        "BrainSegVolNotVent",
        "eTIV",
        "rh_G&S_frontomargin_thickness",
        "rh_G&S_occipital_inf_thickness",
        "rh_G&S_paracentral_thickness",
        "rh_G&S_subcentral_thickness",
        "rh_G&S_transv_frontopol_thickness",
        "rh_G&S_cingul-Ant_thickness",
        "rh_G&S_cingul-Mid-Ant_thickness",
        "rh_G&S_cingul-Mid-Post_thickness",
        "rh_G_cingul-Post-dorsal_thickness",
        "rh_G_cingul-Post-ventral_thickness",
        "rh_G_cuneus_thickness",
        "rh_G_front_inf-Opercular_thickness",
        "rh_G_front_inf-Orbital_thickness",
        "rh_G_front_inf-Triangul_thickness",
        "rh_G_front_middle_thickness",
        "rh_G_front_sup_thickness",
        "rh_G_Ins_lg&S_cent_ins_thickness",
        "rh_G_insular_short_thickness",
        "rh_G_occipital_middle_thickness",
        "rh_G_occipital_sup_thickness",
        "rh_G_oc-temp_lat-fusifor_thickness",
        "rh_G_oc-temp_med-Lingual_thickness",
        "rh_G_oc-temp_med-Parahip_thickness",
        "rh_G_orbital_thickness",
        "rh_G_pariet_inf-Angular_thickness",
        "rh_G_pariet_inf-Supramar_thickness",
        "rh_G_parietal_sup_thickness",
        "rh_G_postcentral_thickness",
        "rh_G_precentral_thickness",
        "rh_G_precuneus_thickness",
        "rh_G_rectus_thickness",
        "rh_G_subcallosal_thickness",
        "rh_G_temp_sup-G_T_transv_thickness",
        "rh_G_temp_sup-Lateral_thickness",
        "rh_G_temp_sup-Plan_polar_thickness",
        "rh_G_temp_sup-Plan_tempo_thickness",
        "rh_G_temporal_inf_thickness",
        "rh_G_temporal_middle_thickness",
        "rh_Lat_Fis-ant-Horizont_thickness",
        "rh_Lat_Fis-ant-Vertical_thickness",
        "rh_Lat_Fis-post_thickness",
        "rh_Pole_occipital_thickness",
        "rh_Pole_temporal_thickness",
        "rh_S_calcarine_thickness",
        "rh_S_central_thickness",
        "rh_S_cingul-Marginalis_thickness",
        "rh_S_circular_insula_ant_thickness",
        "rh_S_circular_insula_inf_thickness",
        "rh_S_circular_insula_sup_thickness",
        "rh_S_collat_transv_ant_thickness",
        "rh_S_collat_transv_post_thickness",
        "rh_S_front_inf_thickness",
        "rh_S_front_middle_thickness",
        "rh_S_front_sup_thickness",
        "rh_S_interm_prim-Jensen_thickness",
        "rh_S_intrapariet&P_trans_thickness",
        "rh_S_oc_middle&Lunatus_thickness",
        "rh_S_oc_sup&transversal_thickness",
        "rh_S_occipital_ant_thickness",
        "rh_S_oc-temp_lat_thickness",
        "rh_S_oc-temp_med&Lingual_thickness",
        "rh_S_orbital_lateral_thickness",
        "rh_S_orbital_med-olfact_thickness",
        "rh_S_orbital-H_Shaped_thickness",
        "rh_S_parieto_occipital_thickness",
        "rh_S_pericallosal_thickness",
        "rh_S_postcentral_thickness",
        "rh_S_precentral-inf-part_thickness",
        "rh_S_precentral-sup-part_thickness",
        "rh_S_suborbital_thickness",
        "rh_S_subparietal_thickness",
        "rh_S_temporal_inf_thickness",
        "rh_S_temporal_sup_thickness",
        "rh_S_temporal_transverse_thickness",
        "rh_MeanThickness_thickness",
        "Left-Lateral-Ventricle",
        "Left-Inf-Lat-Vent",
        "Left-Cerebellum-White-Matter",
        "Left-Cerebellum-Cortex",
        "Left-Thalamus-Proper",
        "Left-Caudate",
        "Left-Putamen",
        "Left-Pallidum",
        "3rd-Ventricle",
        "4th-Ventricle",
        "Brain-Stem",
        "Left-Hippocampus",
        "Left-Amygdala",
        "CSF",
        "Left-Accumbens-area",
        "Left-VentralDC",
        "Left-vessel",
        "Left-choroid-plexus",
        "Right-Lateral-Ventricle",
        "Right-Inf-Lat-Vent",
        "Right-Cerebellum-White-Matter",
        "Right-Cerebellum-Cortex",
        "Right-Thalamus-Proper",
        "Right-Caudate",
        "Right-Putamen",
        "Right-Pallidum",
        "Right-Hippocampus",
        "Right-Amygdala",
        "Right-Accumbens-area",
        "Right-VentralDC",
        "Right-vessel",
        "Right-choroid-plexus",
        "5th-Ventricle",
        "WM-hypointensities",
        "Left-WM-hypointensities",
        "Right-WM-hypointensities",
        "non-WM-hypointensities",
        "Left-non-WM-hypointensities",
        "Right-non-WM-hypointensities",
        "Optic-Chiasm",
        "CC_Posterior",
        "CC_Mid_Posterior",
        "CC_Central",
        "CC_Mid_Anterior",
        "CC_Anterior",
        "BrainSegVol",
        "BrainSegVolNotVentSurf",
        "lhCortexVol",
        "rhCortexVol",
        "CortexVol",
        "lhCerebralWhiteMatterVol",
        "rhCerebralWhiteMatterVol",
        "CerebralWhiteMatterVol",
        "SubCortGrayVol",
        "TotalGrayVol",
        "SupraTentorialVol",
        "SupraTentorialVolNotVent",
        "SupraTentorialVolNotVentVox",
        "MaskVol",
        "BrainSegVol-to-eTIV",
        "MaskVol-to-eTIV",
        "lhSurfaceHoles",
        "rhSurfaceHoles",
        "SurfaceHoles",
        "EstimatedTotalIntraCranialVol",
    ]
    norm_data = NormData.from_dataframe(
        name="fcon1000",
        dataframe=data,
        covariates=covariates,
        batch_effects=batch_effects,
        response_vars=response_vars,
        subject_ids=subject_ids,
    )
    return norm_data
