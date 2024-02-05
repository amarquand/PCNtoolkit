import numpy as np
import pandas as pd
import pytest

from pcntoolkit.dataio.norm_data import NormData
from pcntoolkit.normative_model.norm_base import NormBase
from pcntoolkit.normative_model.norm_hbr import NormHBR
from pytest_tests.fixtures.data import *
from pytest_tests.fixtures.model import *
from pytest_tests.fixtures.paths import *


@pytest.fixture
def sample_args():
    return {"draws": 10, "tune": 10, "cores": 1}


@pytest.fixture
def log_dir():
    return "pytest_tests/resources/log_test"


@pytest.fixture
def save_dir():
    return "pytest_tests/resources/save_load_test"


@pytest.fixture
def norm_args(log_dir, save_dir):
    return {"log_dir": log_dir, "save_dir": save_dir}


@pytest.fixture
def fit_data():
    X = np.random.randn(10000, 2)
    y = np.random.randn(10000)
    batch_effects = np.random.choice([0, 1], (1000, 2))
    return NormData.from_ndarrays("fit", X, y, batch_effects)


@pytest.fixture
def predict_data():
    X = np.random.randn(100, 2)
    y = np.random.randn(100)
    batch_effects = np.random.choice([0, 1], (100, 2))
    return NormData.from_ndarrays("predict", X, y, batch_effects)


@pytest.fixture
def save_dir():
    return "pytest_tests/resources/hbr/save_load_test"


def test_standardize(
    train_norm_data, new_norm_hbr_model: NormBase, n_covariates, n_response_vars
):
    object.__setattr__(new_norm_hbr_model._norm_conf, "inscaler", "standardize")
    object.__setattr__(new_norm_hbr_model._norm_conf, "outscaler", "standardize")
    X_bak = train_norm_data.X.data.copy()
    y_bak = train_norm_data.y.data.copy()
    new_norm_hbr_model.scale_forward(train_norm_data)
    new_norm_hbr_model.scale_backward(train_norm_data)

    assert train_norm_data.scaled_X.data.mean(axis=0) == pytest.approx(
        n_covariates * [0]
    )
    assert train_norm_data.scaled_X.data.std(axis=0) == pytest.approx(
        n_covariates * [1]
    )
    assert train_norm_data.scaled_y.data.mean(axis=0) == pytest.approx(
        n_response_vars * [0]
    )
    assert train_norm_data.scaled_y.data.std(axis=0) == pytest.approx(
        n_response_vars * [1]
    )
    assert np.allclose(train_norm_data.X.data, X_bak)
    assert np.allclose(train_norm_data.y.data, y_bak)


def test_minmax(
    train_dataframe, new_norm_hbr_model: NormBase, n_covariates, n_response_vars
):

    norm_data = NormData.from_dataframe(
        "train_norm_data",
        train_dataframe,
        covariates=["X1", "X2"],
        batch_effects=["batch1", "batch2"],
        response_vars=[f"Y{i+1}" for i in range(n_response_vars)],
    )
    object.__setattr__(new_norm_hbr_model._norm_conf, "inscaler", "minmax")
    object.__setattr__(new_norm_hbr_model._norm_conf, "outscaler", "minmax")
    X_bak = norm_data.X.data.copy()
    y_bak = norm_data.y.data.copy()
    new_norm_hbr_model.scale_forward(norm_data)
    new_norm_hbr_model.scale_backward(norm_data)

    assert np.allclose(norm_data.scaled_X.min(axis=0), 0)
    assert np.allclose(norm_data.scaled_X.max(axis=0), 1)
    assert np.allclose(norm_data.scaled_y.min(axis=0), 0)
    assert np.allclose(norm_data.scaled_y.max(axis=0), 1)
    assert np.allclose(norm_data.X.data, X_bak)
    assert np.allclose(norm_data.y.data, y_bak)


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
        covariates=["X1", "X2"],
        batch_effects=["batch1", "batch2"],
        response_vars=[f"Y{i+1}" for i in range(n_response_vars)],
    )
    object.__setattr__(new_norm_hbr_model._norm_conf, "inscaler", "standardize")
    object.__setattr__(new_norm_hbr_model._norm_conf, "outscaler", "standardize")
    new_norm_hbr_model.scale_forward(norm_data)
    norm_data.expand_basis("polynomial", order=degree, intercept=intercept)
    assert norm_data.Phi.shape == (
        n_train_datapoints,
        n_covariates + degree + 1 * intercept,
    )


@pytest.mark.parametrize(
    "nknots,order, intercept",
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
        covariates=["X1", "X2"],
        batch_effects=["batch1", "batch2"],
        response_vars=[f"Y{i+1}" for i in range(n_response_vars)],
    )
    object.__setattr__(new_norm_hbr_model._norm_conf, "inscaler", "standardize")
    object.__setattr__(new_norm_hbr_model._norm_conf, "outscaler", "standardize")
    new_norm_hbr_model.scale_forward(norm_data)
    norm_data.expand_basis("bspline", nknots=nknots, order=order, intercept=intercept)
    assert norm_data.Phi.shape == (
        n_train_datapoints,
        n_covariates + nknots + order - 1 + 1 * intercept,
    )
