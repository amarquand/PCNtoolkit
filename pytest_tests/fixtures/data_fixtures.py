import numpy as np
import pandas as pd
import pytest

"""
This file contains pytest fixtures for generating and using synthetic data in the PCNtoolkit.

The fixtures defined here include:
1. Number of training, testing, and transfer data points
2. Number of covariates and response variables
3. Functions to generate numpy arrays and pandas DataFrames

These fixtures are used to create consistent and controlled datasets for testing
"""


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
    X_columns = [f"covariate_{i}" for i in range(X.shape[1])]
    y_columns = [f"response_var_{i}" for i in range(y.shape[1])]
    batch_effect_columns = [f"batch_effect_{i}" for i in range(batch_effects.shape[1])]
    all_columns = X_columns + y_columns + batch_effect_columns
    if len(y.shape) == 1:
        y = y[:, None]
    return pd.DataFrame(
        np.concatenate([X, y, batch_effects], axis=1), columns=all_columns
    )


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
