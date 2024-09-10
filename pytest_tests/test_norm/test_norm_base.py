import numpy as np
import pandas as pd
import pytest

from pcntoolkit.dataio.norm_data import NormData
from pcntoolkit.normative_model.norm_base import NormBase
from pcntoolkit.normative_model.norm_hbr import NormHBR
from pytest_tests.fixtures.data_fixtures import *
from pytest_tests.fixtures.model_fixtures import *
from pytest_tests.fixtures.path_fixtures import *
from pytest_tests.fixtures.norm_data_fixtures import *


"""
This file contains tests for the NormBase class in the PCNtoolkit.

The tests cover the following aspects:
1. Fitting the model with valid data
2. Handling invalid data for fitting
3. Scaling methods for data
4. Polynomial basis expansion
5. B-spline basis expansion
"""


# Test the fit method of the NormBase class
def test_norm_base_fit(new_norm_hbr_model: NormBase, norm_data_from_arrays: NormData):
    new_norm_hbr_model.fit(norm_data_from_arrays)

    assert new_norm_hbr_model.response_vars is not None
    assert len(new_norm_hbr_model.regression_models) == len(
        new_norm_hbr_model.response_vars
    )

    for responsevar in new_norm_hbr_model.response_vars:
        assert responsevar in new_norm_hbr_model.regression_models
        assert new_norm_hbr_model.regression_models[responsevar].is_fitted


# Test the fit method of the NormBase class with invalid data
def test_norm_base_fit_invalid_data(new_norm_hbr_model: NormBase):
    with pytest.raises(AttributeError):
        new_norm_hbr_model.fit(None)

    with pytest.raises(AttributeError):
        new_norm_hbr_model.fit("invalid_data")


# Scaling tests
@pytest.mark.parametrize("scaler", ["standardize", "minmax"])
def test_scaling(
    norm_data_from_arrays,
    new_norm_hbr_model: NormBase,
    n_covariates,
    n_response_vars,
    scaler,
):
    object.__setattr__(new_norm_hbr_model._norm_conf, "inscaler", scaler)
    object.__setattr__(new_norm_hbr_model._norm_conf, "outscaler", scaler)
    X_bak = norm_data_from_arrays.X.data.copy()
    y_bak = norm_data_from_arrays.y.data.copy()
    new_norm_hbr_model.scale_forward(norm_data_from_arrays)
    new_norm_hbr_model.scale_backward(norm_data_from_arrays)

    if scaler == "standardize":
        assert_standardized(norm_data_from_arrays, n_covariates, n_response_vars)
    elif scaler == "minmax":
        assert_minmax_scaled(norm_data_from_arrays)

    assert np.allclose(norm_data_from_arrays.X.data, X_bak)
    assert np.allclose(norm_data_from_arrays.y.data, y_bak)


def assert_standardized(data, n_covariates, n_response_vars):
    assert data.scaled_X.data.mean(axis=0) == pytest.approx(n_covariates * [0])
    assert data.scaled_X.data.std(axis=0) == pytest.approx(n_covariates * [1])
    assert data.scaled_y.data.mean(axis=0) == pytest.approx(n_response_vars * [0])
    assert data.scaled_y.data.std(axis=0) == pytest.approx(n_response_vars * [1])


def assert_minmax_scaled(data):
    assert np.allclose(data.scaled_X.min(axis=0), 0)
    assert np.allclose(data.scaled_X.max(axis=0), 1)
    assert np.allclose(data.scaled_y.min(axis=0), 0)
    assert np.allclose(data.scaled_y.max(axis=0), 1)


@pytest.mark.parametrize(
    "degree,intercept", [(2, False), (3, False), (2, True), (3, True)]
)
def test_polynomial(
    train_dataframe,
    new_norm_hbr_model,
    n_train_datapoints,
    n_response_vars,
    n_covariates,
    degree,
    intercept,
):
    norm_data = NormData.from_dataframe(
        "train_norm_data",
        train_dataframe,
        covariates=["covariate_0", "covariate_1"],
        batch_effects=["batch_effect_0", "batch_effect_1"],
        response_vars=[f"response_var_{i}" for i in range(n_response_vars)],
    )
    object.__setattr__(new_norm_hbr_model._norm_conf, "inscaler", "standardize")
    object.__setattr__(new_norm_hbr_model._norm_conf, "outscaler", "standardize")
    object.__setattr__(new_norm_hbr_model._norm_conf, "basis_function", "polynomial")
    object.__setattr__(new_norm_hbr_model._norm_conf, "order", degree)

    new_norm_hbr_model.scale_forward(norm_data)
    new_norm_hbr_model.expand_basis_new(norm_data, "scaled_X", intercept=intercept)
    # norm_data.expand_basis("polynomial", order=degree, intercept=intercept)
    assert norm_data.Phi.shape == (
        n_train_datapoints,
        n_covariates + degree + 1 * intercept,
    )


@pytest.mark.parametrize(
    "nknots,order,intercept",
    [(5, 3, False), (5, 3, True), (5, 2, False), (5, 2, True)],
)
def test_bspline(
    train_dataframe,
    n_response_vars,
    new_norm_hbr_model,
    n_train_datapoints,
    n_covariates,
    nknots,
    order,
    intercept,
):
    norm_data = NormData.from_dataframe(
        "train_norm_data",
        train_dataframe,
        covariates=["covariate_0", "covariate_1"],
        batch_effects=["batch_effect_0", "batch_effect_1"],
        response_vars=[f"response_var_{i}" for i in range(n_response_vars)],
    )
    object.__setattr__(new_norm_hbr_model._norm_conf, "inscaler", "standardize")
    object.__setattr__(new_norm_hbr_model._norm_conf, "outscaler", "standardize")
    object.__setattr__(new_norm_hbr_model._norm_conf, "basis_function", "bspline")
    object.__setattr__(new_norm_hbr_model._norm_conf, "nknots", nknots)
    object.__setattr__(new_norm_hbr_model._norm_conf, "order", order)

    new_norm_hbr_model.scale_forward(norm_data)
    new_norm_hbr_model.expand_basis_new(norm_data, "scaled_X", intercept=intercept)
    assert norm_data.Phi.shape == (
        n_train_datapoints,
        n_covariates + nknots + order - 1 + 1 * intercept,
    )
