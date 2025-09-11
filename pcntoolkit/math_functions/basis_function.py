from __future__ import annotations

import copy
from abc import ABC, abstractmethod
from typing import Optional, Union

import numpy as np
from scipy.interpolate import BSpline

from pcntoolkit.util.output import Errors, Output


def create_basis_function(
    basis_type: str | dict | None,
    basis_column: Optional[Union[int, list[int]]] = None,
    **kwargs,
) -> BasisFunction:
    if isinstance(basis_type, dict):
        return BasisFunction.from_dict(basis_type)
    elif basis_type in ["polynomial", "PolynomialBasisFunction"]:
        return PolynomialBasisFunction(basis_column, **kwargs)
    elif basis_type in ["bspline", "BsplineBasisFunction"]:
        new_knots = {int(k): v for k, v in kwargs.pop("knots", {}).items()}
        for k, v in new_knots.items():
            if isinstance(v, list):
                new_knots[k] = np.array(v)
        return BsplineBasisFunction(basis_column, **kwargs, knots=new_knots)
    else:
        return LinearBasisFunction(basis_column)


class BasisFunction(ABC):
    def __init__(
        self,
        basis_column: Optional[int | list[int]] = None,
        **kwargs,
    ):
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

    @classmethod
    def from_dict(cls, my_dict: dict) -> BasisFunction:
        basis_function_type = my_dict["basis_function"]
        basis_function = create_basis_function(basis_function_type, **my_dict)
        return basis_function

    @classmethod
    def from_args(cls, name: str, args: dict) -> BasisFunction:
        basis_function_type = args.pop(name, "linear")
        if basis_function_type == "bspline":
            nknots = args.pop(f"{name}_nknots", 3)
            degree = args.pop(f"{name}_degree", 3)
            left_expand = args.pop(f"{name}_left_expand", 0.05)
            right_expand = args.pop(f"{name}_right_expand", 0.05)
            knot_method = args.pop(f"{name}_knot_method", "uniform")
            basis_column = args.pop(f"{name}_basis_column", None)
            return BsplineBasisFunction(
                basis_column=basis_column,
                degree=degree,
                nknots=nknots,
                left_expand=left_expand,
                right_expand=right_expand,
                knot_method=knot_method,
            )
        elif basis_function_type == "polynomial":
            degree = args.pop(f"{name}_degree", 3)
            basis_column = args.pop(f"{name}_basis_column", None)
            return PolynomialBasisFunction(basis_column=basis_column, degree=degree)
        else:
            basis_column = args.pop(f"{name}_basis_column", None)
            return LinearBasisFunction(basis_column=basis_column)

    def fit(self, X: np.ndarray) -> None:
        if self.basis_column == [-1]:
            self.basis_column = [i for i in range(X.shape[1])]
        for i in self.basis_column:
            self.fit_column(X, i)
        self.is_fitted = True

    def transform(self, X: np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            raise ValueError(Output.error(Errors.ERROR_BASIS_FUNCTION_NOT_FITTED))
        all_arrays = []
        for i in range(X.shape[1]):
            if i in self.basis_column:
                expanded_arrays = self.transform_column(X, i)
                all_arrays.append(expanded_arrays)
            else:
                copied_array = copy.deepcopy(X[:, i])
                all_arrays.append(copied_array[:, None])

        return np.concatenate(all_arrays, axis=1)

    def copy_column(self, X: np.ndarray, i: int):
        copied_array = X[:, i]
        return copied_array

    def transform_column(self, X: np.ndarray, i: int) -> np.ndarray:
        array = X[:, i]
        squeezed = np.squeeze(array)
        if squeezed.ndim > 1:
            raise ValueError(Output.error(Errors.ERROR_DATA_MUST_BE_1D))
        transformed_array = self._transform(array, i)
        if transformed_array.ndim == 1:
            transformed_array = transformed_array.reshape(-1, 1)
        return transformed_array

    def fit_column(self, X: np.ndarray, i: int) -> None:
        array = X[:, i]
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
        mydict = copy.deepcopy(self.__dict__)
        mydict["basis_function"] = self.__class__.__name__
        return mydict


class PolynomialBasisFunction(BasisFunction):
    def __init__(
        self,
        basis_column: Optional[Union[int, list[int]]] = None,
        degree: int = 3,
        **kwargs,
    ):
        super().__init__(basis_column, **kwargs)
        self.degree = degree
        self.basis_name = "poly"

    def _fit(self, data: np.ndarray, i: int) -> None:
        pass

    def _transform(self, data: np.ndarray, i: int) -> np.ndarray:
        transformed_array = np.power.outer(data, np.arange(1, self.degree + 1))
        return transformed_array


class BsplineBasisFunction(BasisFunction):
    def __init__(
        self,
        basis_column: Optional[Union[int, list[int]]] = None,
        degree: int = 3,
        nknots: int = 5,
        left_expand: float = 0.05,
        right_expand: float = 0.05,
        knot_method: str = "uniform",
        knots: dict[int, np.ndarray] = None,  # type: ignore
        **kwargs,
    ):
        super().__init__(basis_column, **kwargs)
        self.degree = degree
        self.nknots = nknots
        self.left_expand = left_expand
        self.right_expand = right_expand
        self.knot_method = knot_method
        self.knots = knots or {}
        for k, v in self.knots.items():
            if isinstance(v, list):
                self.knots[k] = np.array(v)
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
        if isinstance(knots, np.ndarray):
            self.knots[i] = knots
        else:
            self.knots[i] = np.array(knots)

    def _transform(self, data: np.ndarray, i: int) -> np.ndarray:
        spline = BSpline.design_matrix(data, self.knots[i], self.degree, extrapolate=True).toarray()
        return np.concatenate((data.reshape(-1, 1), spline), axis=1)

    def to_dict(self) -> dict:
        mydict = super().to_dict()
        mydict["degree"] = self.degree
        mydict["nknots"] = self.nknots
        mydict["left_expand"] = self.left_expand
        mydict["right_expand"] = self.right_expand
        mydict["knot_method"] = self.knot_method
        mydict["knots"] = {}
        for k, v in self.knots.items():
            if isinstance(v, np.ndarray):
                mydict["knots"][k] = v.tolist()
            elif isinstance(v, list):
                mydict["knots"][k] = v
            else:
                mydict["knots"][k] = list(v)
        return mydict


class LinearBasisFunction(BasisFunction):
    def __init__(self, basis_column: Optional[Union[int, list[int]]] = None, **kwargs):
        super().__init__(basis_column, **kwargs)
        self.basis_name = "linear"

    def _fit(self, data: np.ndarray, i: int) -> None:
        pass

    def _transform(self, data: np.ndarray, i: int) -> np.ndarray:
        return data
