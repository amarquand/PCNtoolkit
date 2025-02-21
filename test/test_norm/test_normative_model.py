import copy

import numpy as np
import pytest

from pcntoolkit.dataio.norm_data import NormData
from pcntoolkit.normative_model import NormativeModel
from test.fixtures.data_fixtures import *
from test.fixtures.norm_data_fixtures import *
from test.fixtures.path_fixtures import *
from test.fixtures.test_model_fixtures import *

"""
This file contains tests for the NormBase class in the PCNtoolkit.

The tests cover the following aspects:
1. Fitting the model with valid data
2. Handling invalid data for fitting
3. Scaling methods for data
4. Polynomial basis expansion
5. B-spline basis expansion
"""

def test_fit(new_norm_test_model: NormativeModel, norm_data_from_arrays: NormData):
    new_norm_test_model.fit(norm_data_from_arrays)
    assert new_norm_test_model.is_fitted

def test_fit_predict(new_norm_test_model: NormativeModel, norm_data_from_arrays: NormData, test_norm_data_from_arrays: NormData):
    new_norm_test_model.fit_predict(norm_data_from_arrays, test_norm_data_from_arrays)
    assert new_norm_test_model.is_fitted

def test_predict(fitted_norm_test_model: NormativeModel, test_norm_data_from_arrays: NormData):
    fitted_norm_test_model.predict(test_norm_data_from_arrays)
    assert fitted_norm_test_model.is_fitted

def test_harmonize(fitted_norm_test_model: NormativeModel, norm_data_from_arrays: NormData):
    fitted_norm_test_model.harmonize(norm_data_from_arrays)
    assert fitted_norm_test_model.is_fitted

def test_compute_zscores(fitted_norm_test_model: NormativeModel, norm_data_from_arrays: NormData):
    fitted_norm_test_model.compute_zscores(norm_data_from_arrays)
    assert fitted_norm_test_model.is_fitted

def test_compute_centiles(fitted_norm_test_model: NormativeModel, norm_data_from_arrays: NormData):
    fitted_norm_test_model.compute_centiles(norm_data_from_arrays)
    assert fitted_norm_test_model.is_fitted

# Test the fit method of the NormBase class with invalid data
def test_norm_base_fit_invalid_data(new_norm_test_model: NormativeModel):
    with pytest.raises(AttributeError):
        new_norm_test_model.fit(None)

    with pytest.raises(AttributeError):
        new_norm_test_model.fit("invalid_data")




# Scaling tests
@pytest.mark.parametrize("scaler", ["standardize", "minmax"])
def test_scaling(
    norm_data_from_arrays,
    new_norm_test_model: NormativeModel,
    n_covariates,
    n_response_vars,
    scaler,
):
    object.__setattr__(new_norm_test_model, "inscaler", scaler)
    object.__setattr__(new_norm_test_model, "outscaler", scaler)
    copydata = copy.deepcopy(norm_data_from_arrays)
    copydata.attrs['is_scaled']=False
    for s in new_norm_test_model.inscalers:
        s.is_fitted=False
    for s in new_norm_test_model.outscalers:
        s.is_fitted=False
    X_bak = norm_data_from_arrays.X.data.copy()
    y_bak = norm_data_from_arrays.Y.data.copy()
    new_norm_test_model.scale_forward(copydata)

    if scaler == "standardize":
        assert_standardized(copydata, n_covariates, n_response_vars)
    elif scaler == "minmax":
        assert_minmax_scaled(copydata)

    new_norm_test_model.scale_backward(copydata)
    assert np.allclose(copydata.X.data, X_bak)
    assert np.allclose(copydata.Y.data, y_bak)


def assert_standardized(data, n_covariates, n_response_vars):
    assert data.X.data.mean(axis=0) == pytest.approx(n_covariates * [0])
    assert data.X.data.std(axis=0) == pytest.approx(n_covariates * [1])
    assert data.Y.data.mean(axis=0) == pytest.approx(n_response_vars * [0])
    assert data.Y.data.std(axis=0) == pytest.approx(n_response_vars * [1])


def assert_minmax_scaled(data):
    assert np.allclose(data.X.data.min(axis=0), 0)
    assert np.allclose(data.X.data.max(axis=0), 1)
    assert np.allclose(data.Y.data.min(axis=0), 0)
    assert np.allclose(data.Y.data.max(axis=0), 1)
