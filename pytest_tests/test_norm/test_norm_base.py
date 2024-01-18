import numpy as np
import pandas as pd
import pytest

from pcntoolkit.dataio.norm_data import NormData
from pcntoolkit.normative_model.norm_base import NormBase
from pcntoolkit.normative_model.norm_hbr import NormHBR


@pytest.fixture
def n_datapoints():
    return 1000


@pytest.fixture
def n_covariates():
    return 2


@pytest.fixture
def n_response_vars():
    return 2


@pytest.fixture
def np_arrays(n_datapoints, n_covariates, n_response_vars):
    X = np.random.randn(n_datapoints, n_covariates)
    y = np.random.randn(n_datapoints, n_response_vars)
    batch_effects = np.concatenate(
        [
            np.random.choice([0, 1], (n_datapoints, 1)),
            np.random.choice([0, 2, 1], (n_datapoints, 1)),
        ],
        axis=1,
    )
    return X, y, batch_effects


@pytest.fixture
def dataframe(np_arrays):
    X, y, batch_effects = np_arrays
    if len(y.shape) == 1:
        y = y[:, None]
    return pd.DataFrame(
        np.concatenate([X, y, batch_effects], axis=1),
        columns=["X1", "X2"]
        + [f"resp_{i}" for i in range(y.shape[1])]
        + ["batch1", "batch2"],
    )


@pytest.fixture
def norm_data(dataframe, n_response_vars):
    norm_data = NormData.from_dataframe(
        "test",
        dataframe,
        covariates=["X1", "X2"],
        batch_effects=["batch1", "batch2"],
        response_vars=[f"resp_{i}" for i in range(n_response_vars)],
    )
    return norm_data


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
    X = np.random.randn(1000, 2)
    y = np.random.randn(1000)
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
    return "pytest_tests/resources/save_load_test"


@pytest.fixture
def norm_hbr(save_dir) -> NormHBR:
    return NormHBR.load(save_dir)


def test_standardize(norm_data, norm_hbr: NormBase, n_covariates, n_response_vars):
    object.__setattr__(norm_hbr._norm_conf, "inscaler", "standardize")
    object.__setattr__(norm_hbr._norm_conf, "outscaler", "standardize")
    X_bak = norm_data.X.data.copy()
    y_bak = norm_data.y.data.copy()
    norm_hbr.scale_forward(norm_data)
    norm_hbr.scale_backward(norm_data)

    assert norm_data.scaled_X.data.mean(axis=0) == pytest.approx(n_covariates * [0])
    assert norm_data.scaled_X.data.std(axis=0) == pytest.approx(n_covariates * [1])
    assert norm_data.scaled_y.data.mean(axis=0) == pytest.approx(n_response_vars * [0])
    assert norm_data.scaled_y.data.std(axis=0) == pytest.approx(n_response_vars * [1])
    assert np.allclose(norm_data.X.data, X_bak)
    assert np.allclose(norm_data.y.data, y_bak)


def test_minmax(norm_data, norm_hbr: NormBase, n_covariates, n_response_vars):
    object.__setattr__(norm_hbr._norm_conf, "inscaler", "minmax")
    object.__setattr__(norm_hbr._norm_conf, "outscaler", "minmax")
    X_bak = norm_data.X.data.copy()
    y_bak = norm_data.y.data.copy()
    norm_hbr.scale_forward(norm_data)
    norm_hbr.scale_backward(norm_data)

    assert np.allclose(norm_data.scaled_X.min(axis=0), 0)
    assert np.allclose(norm_data.scaled_X.max(axis=0), 1)
    assert np.allclose(norm_data.scaled_y.min(axis=0), 0)
    assert np.allclose(norm_data.scaled_y.max(axis=0), 1)
    assert np.allclose(norm_data.X.data, X_bak)
    assert np.allclose(norm_data.y.data, y_bak)


@pytest.mark.parametrize(
    "degree,intercept", [(2, False), (3, False), (2, True), (3, True)]
)
def test_polynomial(norm_data, norm_hbr, n_datapoints, n_covariates, degree, intercept):
    object.__setattr__(norm_hbr._norm_conf, "inscaler", "standardize")
    object.__setattr__(norm_hbr._norm_conf, "outscaler", "standardize")
    norm_hbr.scale_forward(norm_data)
    norm_data.expand_basis("polynomial", order=degree, intercept=intercept)
    assert norm_data.Phi.shape == (n_datapoints, n_covariates + degree + 1 * intercept)


@pytest.mark.parametrize(
    "nknots,order, intercept",
    [(5, 3, False), (5, 3, True), (5, 2, False), (5, 2, True)],
)
def test_bspline(
    norm_data, norm_hbr, n_datapoints, n_covariates, nknots, order, intercept
):
    object.__setattr__(norm_hbr._norm_conf, "inscaler", "standardize")
    object.__setattr__(norm_hbr._norm_conf, "outscaler", "standardize")
    norm_hbr.scale_forward(norm_data)
    norm_data.expand_basis("bspline", nknots=nknots, order=order, intercept=intercept)
    assert norm_data.Phi.shape == (
        n_datapoints,
        n_covariates + nknots + order - 1 + 1 * intercept,
    )
