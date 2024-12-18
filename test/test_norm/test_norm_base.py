import numpy as np
import pytest

from pcntoolkit.dataio.norm_data import NormData
from pcntoolkit.normative_model.norm_base import NormBase
from test.fixtures.data_fixtures import *
from test.fixtures.hbr_model_fixtures import *
from test.fixtures.norm_data_fixtures import *
from test.fixtures.path_fixtures import *

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



