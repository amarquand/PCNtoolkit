"""
norm_data module
===============

This module provides functionalities for normalizing and converting different types of data into a NormData object.

The NormData object is an xarray.Dataset that contains the data, covariates, batch effects, and response variables, and it
is used by all the models in the toolkit.
"""

from __future__ import annotations

import copy
import fcntl
import os
from collections import defaultdict
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
)

# pylint: enable=deprecated-class
import numpy as np
import pandas as pd  # type: ignore
import xarray as xr
from nibabel.loadsave import load
from sklearn.model_selection import StratifiedKFold, train_test_split  # type: ignore

# import datavars from xarray
from xarray.core.types import DataVars

from pcntoolkit.dataio.fileio import load
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
        "thrive_covariate",  # Whether the ids are real or synthetic
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
        X: np.ndarray,
        Y: np.ndarray,
        batch_effects: np.ndarray | None = None,
        subject_ids: np.ndarray | None = None,
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
        data_vars = {}
        coords = {}
        attrs = attrs or {}
        if subject_ids is not None:
            attrs["real_ids"] = True  # type:ignore
            data_vars["subjects"] = (["observations"], subject_ids)
        else:
            attrs["real_ids"] = False  # type: ignore
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
    def from_paths(cls, name: str, covariates_path: str, responses_path: str, batch_effects_path: str, **kwargs) -> "NormData":  # type: ignore
        """
        Load a normative dataset from a dictionary of paths.
        """
        covs = load(covariates_path, **kwargs)
        resp = load(responses_path, **kwargs)
        batch_effects = load(batch_effects_path, **kwargs)
        return cls.from_ndarrays(name, covs, resp, batch_effects, **kwargs)

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
        img = load(fsl_folder)
        dat = img.get_fdata()

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
            The name you want to give to the dataset. Will be used to name saved results.
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
            attrs["real_ids"] = True  # type: ignore
            data_vars["subjects"] = (["observations"], dataframe[subject_ids].to_numpy())
        else:
            attrs["real_ids"] = False  # type: ignore
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

    def batch_effects_split(
        self,
        batch_effects: Dict[str, List[str]],
        names: Optional[Tuple[str, str]],
    ) -> Tuple[NormData, NormData]:
        """
        Split the data into two datasets, one with the specified batch effects and one without.
        """
        if names is None:
            names = ["selected", "not_selected"]  # type:ignore
        assert names is not None, "names can not be None"
        A = self.select_batch_effects(names[0], batch_effects)
        B = self.select_batch_effects(names[1], batch_effects, invert=True)
        return A, B

    def register_batch_effects(self) -> None:
        """
        Create a mapping of batch effects to unique values.
        """
        my_be: xr.DataArray = self.batch_effects
        # create a dictionary with for each column in the batch effects, a dict from value to int
        self.attrs["unique_batch_effects"] = {}
        self.attrs["batch_effect_counts"] = defaultdict(lambda : 0)
        self.attrs["covariate_ranges"] = {}
        self.attrs["batch_effect_covariate_ranges"] = {}
        for dim in self.batch_effect_dims.to_numpy():
            dim_subset = my_be.sel(batch_effect_dims=dim)
            uniques, counts = np.unique(dim_subset, return_counts=True)

            self.attrs["unique_batch_effects"][dim] = list(uniques)
            self.attrs["batch_effect_counts"][dim] = {k: int(v) for k, v in zip(uniques, counts)}
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
        return self.unique_batch_effects == other.unique_batch_effects
    
    def make_compatible(self: NormData, other: NormData):
        """Ensures datasets are compatible by merging the batch effects maps
        """
        myu = self.unique_batch_effects
        otu = other.unique_batch_effects
        all_unique_batch_effects = {dim: list(set(val).union(set(otu[dim]))) for dim, val in myu.items()}

        mycr = self.covariate_ranges
        otcr = other.covariate_ranges
        ncr = {
            cov: {"min": min(mycr[cov]["min"], otcr[cov]["min"]), "max": max(mycr[cov]["max"], otcr[cov]["max"])}
            for cov in self.covariates.to_numpy()
        }

        mybecr = self.batch_effect_covariate_ranges
        otbecr = other.batch_effect_covariate_ranges
        nbecr = {}
        for dim, uniques in myu.items():
            nbecr[dim] = {}
            for u in uniques:
                nbecr[dim][u] = {}
                for c in self.covariates.to_numpy():
                    match (u in mybecr[dim], u in otbecr[dim]):
                        case True, True:
                            nbecr[dim][u][c] = {
                                "min": min(mybecr[dim][u][c]["min"], otbecr[dim][u][c]["min"]),
                                "max": max(mybecr[dim][u][c]["max"], otbecr[dim][u][c]["max"]),
                            }
                        case True, False:
                            nbecr[dim][u][c] = mybecr[dim][u][c]
                        case False, True:
                            nbecr[dim][u][c] = otbecr[dim][u][c]
                        case False, False:
                            raise ValueError("This should never happen")

        self.unique_batch_effects = copy.deepcopy(all_unique_batch_effects)
        other.unique_batch_effects = copy.deepcopy(all_unique_batch_effects)
        self.covariate_ranges = copy.deepcopy(ncr)
        other.covariate_ranges = copy.deepcopy(ncr)
        self.batch_effect_covariate_ranges = copy.deepcopy(nbecr)
        other.covariate_ranges = copy.deepcopy(nbecr)

    def scale_forward(self, inscalers: Dict[str, Any], outscalers: Dict[str, Any]) -> None:
        """
        Scale the data forward in-place using provided scalers.

        Parameters
        ----------
        inscalers : Dict[str, Any]
            Scalers for the covariate data.
        outscalers : Dict[str, Any]
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
                    inscalers[self.attrs["thrive_covariate"]].transform(self.thrive_X.data),
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

            if "Yhat" in self.data_vars:
                scaled_Yhat = np.zeros(self.Yhat.shape)
                for i, responsevar in enumerate(self.response_vars.to_numpy()):
                    scaled_Yhat[:, i] = outscalers[responsevar].transform(self.Yhat.sel(response_vars=responsevar).data)
                self["Yhat"] = xr.DataArray(
                    scaled_Yhat,
                    coords=self.Yhat.coords,
                    dims=self.Yhat.dims,
                    attrs=self.Yhat.attrs,
                )

            if "centiles" in self.data_vars:
                scaled_centiles = np.zeros(self.centiles.shape)
                for i, responsevar in enumerate(self.response_vars.to_numpy()):
                    scaled_centiles[:, :, i] = outscalers[responsevar].transform(
                        self.centiles.sel(response_vars=responsevar).data
                    )
                self["centiles"] = xr.DataArray(
                    scaled_centiles,
                    coords=self.centiles.coords,
                    dims=self.centiles.dims,
                    attrs=self.centiles.attrs,
                )
            if "thrive_Y" in self.data_vars:
                scaled_thrive_Y = np.zeros(self.thrive_Y.shape)
                for i, responsevar in enumerate(self.response_vars.to_numpy()):
                    scaled_thrive_Y[:, i, :] = outscalers[responsevar].transform(
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
                    inscalers[self.attrs["thrive_covariate"]].inverse_transform(self.thrive_X.data),
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

            if "Yhat" in self.data_vars:
                unscaled_Yhat = np.zeros(self.Yhat.shape)
                for i, responsevar in enumerate(self.response_vars.to_numpy()):
                    unscaled_Yhat[:, i] = outscalers[responsevar].inverse_transform(self.Yhat.sel(response_vars=responsevar).data)
                self["Yhat"] = xr.DataArray(
                    unscaled_Yhat,
                    coords=self.Yhat.coords,
                    dims=self.Yhat.dims,
                    attrs=self.Yhat.attrs,
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
                    unscaled_thrive_Y[:, i, :] = outscalers[responsevar].inverse_transform(
                        self.thrive_Y.sel(response_vars=responsevar).data
                    )
                self["thrive_Y"] = xr.DataArray(
                    unscaled_thrive_Y,
                    coords=self.thrive_Y.coords,
                    dims=self.thrive_Y.dims,
                    attrs=self.thrive_Y.attrs,
                )

            self.attrs["is_scaled"] = False

    def select_batch_effects(self, name, batch_effects: Dict[str, List[str]], invert: bool = False) -> NormData:
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
            to_return = NormData.from_xarray(name, to_return)
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

    def save_zscores(self, save_dir: str) -> None:
        zdf = self.Z.to_dataframe().unstack(level="response_vars")
        zdf.columns = zdf.columns.droplevel(0)
        zdf.index = zdf.index.astype(str)
        res_path = os.path.join(save_dir, f"Z_{self.name}.csv")
        with open(res_path, mode="a+" if os.path.exists(res_path) else "w", encoding="utf-8") as f:
            try:
                fcntl.flock(f.fileno(), fcntl.LOCK_EX)
                f.seek(0)
                old_results = pd.read_csv(f) if os.path.getsize(res_path) > 0 else None
                if old_results is not None:
                    old_results["observations"] = old_results["observations"].astype(str)
                    old_results.set_index(["observations"], inplace=True)
                    # Merge on observations, keeping right (new) values for overlapping columns
                    new_results = old_results.merge(zdf, on="observations", how="outer", suffixes=("_old", ""))
                    # Drop columns ending with '_old' as they're the duplicates from old_results
                    new_results = new_results.loc[:, ~new_results.columns.str.endswith("_old")]
                else:
                    new_results = zdf
                f.seek(0)
                f.truncate()
                new_results.to_csv(f)
            finally:
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)

    def load_zscores(self, save_dir) -> None:
        Z_path = os.path.join(save_dir, f"Z_{self.name}.csv")
        if os.path.isfile(Z_path):
            df = pd.read_csv(Z_path)
            non_index_columns = [i for i in list(df.columns) if not i == "observations"]
            self["Z"] = xr.DataArray(
                df[non_index_columns],
                dims=("observations", "response_vars"),
                coords={"observations": df["observations"], "response_vars": non_index_columns},
            )

    def save_centiles(self, save_dir: str) -> None:
        centiles = self.centiles.to_dataframe().unstack(level="response_vars")
        centiles.columns = centiles.columns.droplevel(0)
        centiles.index = centiles.index.set_levels(centiles.index.levels[1].astype(str), level=1)
        res_path = os.path.join(save_dir, f"centiles_{self.name}.csv")
        with open(res_path, mode="a+" if os.path.exists(res_path) else "w", encoding="utf-8") as f:
            try:
                fcntl.flock(f.fileno(), fcntl.LOCK_EX)
                f.seek(0)
                old_results = pd.read_csv(f) if os.path.getsize(res_path) > 0 else None
                if old_results is not None:
                    old_results["observations"] = old_results["observations"].astype(str)
                    old_results.set_index(["observations", "centile"], inplace=True)
                    # Merge on observations, keeping right (new) values for overlapping columns
                    new_results = old_results.merge(centiles, on=["observations", "centile"], how="outer", suffixes=("_old", ""))
                    # Drop columns ending with '_old' as they're the duplicates from old_results
                    new_results = new_results.loc[:, ~new_results.columns.str.endswith("_old")]
                else:
                    new_results = centiles
                f.seek(0)
                f.truncate()
                new_results.to_csv(f)
            finally:
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)

    def load_centiles(self, save_dir) -> None:
        C_path = os.path.join(save_dir, f"centiles_{self.name}.csv")
        if os.path.isfile(C_path):
            df = pd.read_csv(C_path)
            response_vars = [i for i in list(df.columns) if not (i == "observations" or i == "centile")]
            centiles = np.unique(df["centile"])
            obs = np.unique(df["observations"])
            obs.sort()
            A = np.zeros((len(centiles), len(obs), len(response_vars)))
            for i, c in enumerate(centiles):
                sub = df[df["centile"] == c]
                sub.sort_values(by="observations")
                for j, rv in enumerate(response_vars):
                    A[i, :, j] = sub[rv]

            self["centiles"] = xr.DataArray(
                A,
                dims=("centile", "observations", "response_vars"),
                coords={"centile": centiles, "observations": obs, "response_vars": response_vars},
            )

    def save_logp(self, save_dir: str) -> None:
        logp = self.logp.to_dataframe().unstack(level="response_vars")
        logp.columns = logp.columns.droplevel(0)
        logp.index = logp.index.astype(str)
        res_path = os.path.join(save_dir, f"logp_{self.name}.csv")
        with open(res_path, mode="a+" if os.path.exists(res_path) else "w", encoding="utf-8") as f:
            try:
                fcntl.flock(f.fileno(), fcntl.LOCK_EX)
                f.seek(0)
                old_results = pd.read_csv(f) if os.path.getsize(res_path) > 0 else None
                if old_results is not None:
                    old_results["observations"] = old_results["observations"].astype(str)
                    old_results.set_index(["observations"], inplace=True)
                    # Merge on observations, keeping right (new) values for overlapping columns
                    new_results = old_results.merge(logp, on="observations", how="outer", suffixes=("_old", ""))
                    # Drop columns ending with '_old' as they're the duplicates from old_results
                    new_results = new_results.loc[:, ~new_results.columns.str.endswith("_old")]
                else:
                    new_results = logp
                f.seek(0)
                f.truncate()
                new_results.to_csv(f)
            finally:
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)

    def load_logp(self, save_dir) -> None:
        logp_path = os.path.join(save_dir, f"logp_{self.name}.csv")
        if os.path.isfile(logp_path):
            df = pd.read_csv(logp_path)
            non_index_columns = [i for i in list(df.columns) if not i == "observations"]
            self["logp"] = xr.DataArray(
                df[non_index_columns],
                dims=("observations", "response_vars"),
                coords={"observations": df["observations"], "response_vars": non_index_columns},
            )

    def save_statistics(self, save_dir: str) -> None:
        mdf = self.statistics.to_dataframe().unstack(level="response_vars")
        mdf.columns = mdf.columns.droplevel(0)
        res_path = os.path.join(save_dir, f"statistics_{self.name}.csv")
        with open(res_path, mode="a+" if os.path.exists(res_path) else "w", encoding="utf-8") as f:
            try:
                fcntl.flock(f.fileno(), fcntl.LOCK_EX)
                f.seek(0)
                old_results = pd.read_csv(f, index_col=0) if os.path.getsize(res_path) > 0 else None
                if old_results is not None:
                    # Merge on observations, keeping right (new) values for overlapping columns
                    new_results = old_results.merge(mdf, on="statistic", how="outer", suffixes=("_old", ""))
                    # Drop columns ending with '_old' as they're the duplicates from old_results
                    new_results = new_results.loc[:, ~new_results.columns.str.endswith("_old")]
                else:
                    new_results = mdf
                f.seek(0)
                f.truncate()
                new_results.to_csv(f)
            finally:
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)

    def load_statistics(self, save_dir) -> None:
        logp_path = os.path.join(save_dir, f"statistics_{self.name}.csv")
        if os.path.isfile(logp_path):
            df = pd.read_csv(logp_path)
            df = df.set_index("statistic")
            statistics = list(df.index)
            self["statistics"] = xr.DataArray(
                df.T,
                dims=("response_vars", "statistic"),
                coords={"statistic": statistics},
            )

    def save_results(self, save_dir: str) -> None:
        """Saves the results (zscores, centiles, logp, statistics) to disk

        Args:
            save_dir (str): Where the results are saved. I.e.: {save_dir}/Z_fit_test.csv
        """
        self.save_zscores(save_dir)
        self.save_centiles(save_dir)
        self.save_logp(save_dir)
        self.save_statistics(save_dir)

    def load_results(self, save_dir: str) -> None:
        """Loads the results (zscores, centiles, logp, statistics) back into the data

        Args:
            save_dir (str): Where the results are saved. I.e.: {save_dir}/Z_fit_test.csv
        """
        self.load_zscores(save_dir)
        self.load_centiles(save_dir)
        self.load_logp(save_dir)
        self.load_statistics(save_dir)

    def create_statistics_group(self) -> None:
        """
        Initializes a DataArray for statistics with NaN values.

        This method creates a DataArray with dimensions 'response_vars' and 'statistics',
        where 'response_vars' corresponds to the response variables in the dataset,
        and 'statistics' includes statistics such as Rho, RMSE, SMSE, EXPV, NLL, and ShapiroW.
        The DataArray is filled with NaN values initially.
        """
        rv = self.response_vars.to_numpy().copy().tolist()

        self["statistics"] = xr.DataArray(
            np.nan * np.ones((len(rv), 6)),
            dims=("response_vars", "statistics"),
            coords={
                "response_vars": np.arange(len(rv)),
                "statistics": ["Rho", "Rho_p", "R2", "RMSE", "SMSE", "MSLL", "NLL", "ShapiroW", "MACE", "MAPE", "EXPV"],
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
