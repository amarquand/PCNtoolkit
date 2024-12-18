import pytest

from pcntoolkit.util.basis_function import create_basis_function
from pytest_tests.fixtures.norm_data_fixtures import *


@pytest.mark.parametrize("basis_function_name, basis_function_class", [("polynomial", "PolynomialBasisFunction"), ("bspline", "BsplineBasisFunction"), ("linear", "LinearBasisFunction")])
def test_to_and_from_dict(basis_function_name, basis_function_class):
    basis_function = create_basis_function(basis_function_name, source_array_name="X", degree=3)
    basis_function_dict = basis_function.to_dict()
    basis_function_from_dict = create_basis_function(basis_function_dict)
    assert basis_function_dict == basis_function_from_dict.__dict__ | {"basis_function":basis_function_class}


@pytest.mark.parametrize("basis_column", [None, 0, [0], [0,1]]) # None is for all columns
def test_poly_fit_and_transform(norm_data_from_arrays, basis_column):
    degree = 3
    basis_function = create_basis_function("polynomial", source_array_name="X", degree=degree, basis_column=basis_column)
    basis_function.fit(norm_data_from_arrays)
    basis_function.transform(norm_data_from_arrays)
    assert basis_function.is_fitted
    assert basis_function.basis_name == "poly"
    if basis_column is None:
        assert norm_data_from_arrays.Phi.shape == (norm_data_from_arrays.X.data.shape[0], len(norm_data_from_arrays.X.coords["covariates"])*degree)
    else:
        if isinstance(basis_column, int):
            assert norm_data_from_arrays.Phi.shape == (norm_data_from_arrays.X.data.shape[0], degree + 1)
        else:
            assert norm_data_from_arrays.Phi.shape == (norm_data_from_arrays.X.data.shape[0], len(basis_column) * degree + (2-len(basis_column)))

@pytest.mark.parametrize("nknots, degree", [(8, 4), (10, 4), (10, 3)])
def test_bspline_fit_and_transform(norm_data_from_arrays, nknots, degree):
    basis_function = create_basis_function("bspline", source_array_name="X", degree=degree, nknots=nknots)
    basis_function.fit(norm_data_from_arrays)
    basis_function.transform(norm_data_from_arrays)
    assert basis_function.is_fitted
    assert basis_function.basis_name == "bspline"
    assert norm_data_from_arrays.Phi.shape == (norm_data_from_arrays.X.data.shape[0], len(norm_data_from_arrays.X.coords["covariates"]) * (nknots + degree - 1))

