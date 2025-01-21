from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional, Union

import numpy as np
import xarray as xr
from scipy.interpolate import BSpline

from pcntoolkit.dataio.norm_data import NormData
from pcntoolkit.util.output import Errors, Output


def create_basis_function_from_dict(basis_function_dict: dict) -> BasisFunction:
    basis_function_type = basis_function_dict["basis_function"]
    basis_function = create_basis_function(basis_function_type, **basis_function_dict)
    return basis_function


def create_basis_function(
    basis_type: str | dict,
    source_array_name: str = "scaled_X",
    basis_column: Optional[Union[int, list[int]]] = None,
    **kwargs,
) -> BasisFunction:
    if isinstance(basis_type, dict):
        return create_basis_function_from_dict(basis_type)
    elif basis_type in ["polynomial", "PolynomialBasisFunction"]:
        return PolynomialBasisFunction(source_array_name, basis_column, **kwargs)
    elif basis_type in ["bspline", "BsplineBasisFunction"]:
        return BsplineBasisFunction(source_array_name, basis_column, **kwargs)
    else:
        return LinearBasisFunction(source_array_name, basis_column)


class BasisFunction(ABC):
    def __init__(
        self,
        source_array_name: str = "scaled_X",
        basis_column: Optional[int | list[int]] = None,
        **kwargs,
    ):
        self.source_array_name = source_array_name
        self.is_fitted: bool = kwargs.get("is_fitted", False)
        self.basis_name: str = kwargs.get("basis_name", "basis")
        self.min: dict[int, float] = kwargs.get("min", {})
        self.max: dict[int, float] = kwargs.get("max", {})
        self.compute_min: bool = self.min == {}
        self.compute_max: bool = self.max == {}
        if isinstance(basis_column, int):
            self.basis_column: list[int] = [basis_column]
        elif isinstance(basis_column, list):
            self.basis_column: list[int] = basis_column
        else:
            self.basis_column: list[int] = [-1]

    def fit(self, data: NormData) -> None:
        if self.basis_column == [-1]:
            self.basis_column = [i for i in range(data[self.source_array_name].data.shape[1])]
        if self.source_array_name not in data.data_vars:
            raise Output.error(Errors.ERROR_SOURCE_ARRAY_NOT_FOUND, source_array_name=self.source_array_name)
        source_array = data[self.source_array_name]
        for i in self.basis_column:
            self.fit_column(source_array, i)

        self.is_fitted = True

    def transform(self, data: NormData) -> None:
        if self.source_array_name not in data.data_vars:
            raise Output.error(Errors.ERROR_SOURCE_ARRAY_NOT_FOUND, source_array_name=self.source_array_name)
        if not self.is_fitted:
            raise Output.error(Errors.ERROR_BASIS_FUNCTION_NOT_FITTED)
        source_array = data[self.source_array_name]
        all_arrays = []
        for i in range(source_array.data.shape[1]):
            if i in self.basis_column:
                expanded_arrays = self.transform_column(source_array, i)
                all_arrays.append(expanded_arrays)
            else:
                copied_array = self.copy_column(source_array, i)
                all_arrays.append(copied_array)

        data["Phi"] = xr.concat(all_arrays, dim="basis_functions")

    def copy_column(self, source_array: xr.DataArray, i: int):
        copied_array = xr.DataArray(
            source_array.isel(covariates=i).data[:, None],
            dims=["datapoints", "basis_functions"],
            coords={
                "datapoints": source_array.coords["datapoints"],
                "basis_functions": [f"{source_array.coords['covariates'][i].data.item()}"],
            },
        )
        return copied_array

    def transform_column(self, data: xr.DataArray, i: int) -> xr.DataArray:
        array = data.isel(covariates=i).to_numpy()
        squeezed = np.squeeze(array)
        if squeezed.ndim > 1:
            raise Output.error(Errors.ERROR_DATA_MUST_BE_1D, data=data)
        transformed_array = self._transform(array, i)
        if transformed_array.ndim == 1:
            transformed_array = transformed_array.reshape(-1, 1)
        return xr.DataArray(
            transformed_array,
            coords={
                "basis_functions": [
                    f"{data.coords['covariates'][i].data.item()}_{self.basis_name}_{j}" for j in range(transformed_array.shape[1])
                ]
            },
            dims=["datapoints", "basis_functions"],
        )

    def fit_column(self, data: xr.DataArray, i: int) -> None:
        array = data.isel(covariates=i).to_numpy()
        if self.compute_min:
            self.min[i] = np.min(array)
        if self.compute_max:
            self.max[i] = np.max(array)
        self._fit(array, i)

    @abstractmethod
    def _fit(self, data: np.ndarray, i: int) -> None:
        pass

    @abstractmethod
    def _transform(self, data: np.ndarray, i: int) -> np.ndarray:
        pass

    def to_dict(self) -> dict:
        mydict = self.__dict__
        mydict["basis_function"] = self.__class__.__name__
        return mydict


class PolynomialBasisFunction(BasisFunction):
    def __init__(
        self,
        source_array_name: str = "scaled_X",
        basis_column: Optional[Union[int, list[int]]] = None,
        **kwargs,
    ):
        super().__init__(source_array_name, basis_column, **kwargs)
        self.degree = kwargs.get("degree", 3)
        self.basis_name = "poly"

    def _fit(self, data: np.ndarray, i: int) -> None:
        pass

    def _transform(self, data: np.ndarray, i: int) -> np.ndarray:
        transformed_array = np.power.outer(data, np.arange(1, self.degree + 1))
        return transformed_array


class BsplineBasisFunction(BasisFunction):
    def __init__(
        self,
        source_array_name: str = "scaled_X",
        basis_column: Optional[Union[int, list[int]]] = None,
        **kwargs,
    ):
        super().__init__(source_array_name, basis_column, **kwargs)
        self.degree = kwargs.get("degree", 3)
        self.nknots = kwargs.get("nknots", 3)
        self.left_expand = kwargs.get("left_expand", 0.05)
        self.right_expand = kwargs.get("right_expand", 0.05)
        self.knot_method = kwargs.get("knot_method", "uniform")
        self.knots = kwargs.get("knots", {})
        self.basis_name = "bspline"

    def _fit(self, data: np.ndarray, i: int) -> None:
        mymin = self.min[i]
        mymax = self.max[i]
        delta = mymax - mymin
        aug_min = mymin - delta * self.left_expand
        aug_max = mymax + delta * self.right_expand
        if self.knot_method == "uniform":
            knots = np.linspace(aug_min, aug_max, self.nknots)
        elif self.knot_method == "quantile":
            knots = np.percentile(data, np.linspace(0, 100, self.nknots))
        knots = np.concatenate([[aug_min] * self.degree, knots, [aug_max] * self.degree])
        self.knots[i] = knots

    def _transform(self, data: np.ndarray, i: int) -> np.ndarray:
        return BSpline.design_matrix(data, self.knots[i], self.degree, extrapolate=True).toarray()


class LinearBasisFunction(BasisFunction):
    def __init__(self, source_array_name: str = "scaled_X", basis_column: Optional[Union[int, list[int]]] = None):
        super().__init__(source_array_name, basis_column)
        self.basis_name = "linear"

    def _fit(self, data: np.ndarray, i: int) -> None:
        pass

    def _transform(self, data: np.ndarray, i: int) -> np.ndarray:
        return data
