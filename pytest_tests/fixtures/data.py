import numpy as np
import pandas as pd
import pytest

from pcntoolkit.dataio.norm_data import NormData


@pytest.fixture
def n_train_datapoints():
    return 1500


@pytest.fixture
def n_test_datapoints():
    return 1000


@pytest.fixture
def n_transfer_datapoints():
    return 500


@pytest.fixture
def n_covariates():
    return 2


@pytest.fixture
def n_response_vars():
    return 2


def np_arrays(n_datapoints, n_covariates, n_response_vars):
    X = np.random.randn(n_datapoints, n_covariates)
    y = np.random.randn(n_datapoints, n_response_vars)
    batch_effects = []
    batch_effects.append(np.random.choice([0, 1], (n_datapoints, 1)))
    batch_effects.append(np.random.choice([0, 1, 2], (n_datapoints, 1)))
    batch_effects = np.concatenate(batch_effects, axis=1)
    return X, y, batch_effects


def dataframe(n_datapoints, n_covariates, n_response_vars):
    X, y, batch_effects = np_arrays(n_datapoints, n_covariates, n_response_vars)
    X_columns = [f"X{i+1}" for i in range(X.shape[1])]
    y_columns = [f"Y{i+1}" for i in range(y.shape[1])]
    batch_effect_columns = [f"batch{i+1}" for i in range(batch_effects.shape[1])]
    all_columns = X_columns + y_columns + batch_effect_columns
    if len(y.shape) == 1:
        y = y[:, None]
    return pd.DataFrame(
        np.concatenate([X, y, batch_effects], axis=1), columns=all_columns
    )


def norm_data(name, n_datapoints, n_covariates, n_response_vars):
    dataframe_train = dataframe(n_datapoints, n_covariates, n_response_vars)
    norm_data = NormData.from_dataframe(
        name,
        dataframe_train,
        covariates=["X1", "X2"],
        batch_effects=["batch1", "batch2"],
        response_vars=[f"Y{i+1}" for i in range(n_response_vars)],
    )
    return norm_data


@pytest.fixture
def train_arrays(n_train_datapoints, n_covariates, n_response_vars):
    X_train, y_train, batch_effects_train = np_arrays(
        n_train_datapoints, n_covariates, n_response_vars
    )
    return X_train, y_train, batch_effects_train


@pytest.fixture
def test_arrays(n_test_datapoints, n_covariates, n_response_vars):
    X_test, y_test, batch_effects_test = np_arrays(
        n_test_datapoints, n_covariates, n_response_vars
    )
    return X_test, y_test, batch_effects_test


@pytest.fixture
def train_dataframe(n_train_datapoints, n_covariates, n_response_vars):
    dataframe_train = dataframe(n_train_datapoints, n_covariates, n_response_vars)
    return dataframe_train


@pytest.fixture
def test_dataframe(n_test_datapoints, n_covariates, n_response_vars):
    dataframe_test = dataframe(n_test_datapoints, n_covariates, n_response_vars)
    return dataframe_test


@pytest.fixture(scope="function")
def train_norm_data(n_train_datapoints, n_covariates, n_response_vars):
    norm_data_train = norm_data(
        "train", n_train_datapoints, n_covariates, n_response_vars
    )
    return norm_data_train


@pytest.fixture
def test_norm_data(n_test_datapoints, n_covariates, n_response_vars):
    norm_data_test = norm_data("test", n_test_datapoints, n_covariates, n_response_vars)
    return norm_data_test


@pytest.fixture
def transfer_norm_data(n_transfer_datapoints, n_covariates, n_response_vars):
    df = dataframe(n_transfer_datapoints, n_covariates, n_response_vars)
    df["batch1"] += 3

    return NormData.from_dataframe(
        "transfer",
        df,
        covariates=["X1", "X2"],
        batch_effects=["batch1", "batch2"],
        response_vars=[f"Y{i+1}" for i in range(n_response_vars)],
    )
