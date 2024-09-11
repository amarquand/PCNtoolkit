import os
from turtle import pd
import numpy as np
import pytest
from tempfile import gettempdir
from pytest_tests.fixtures.data_fixtures import *

"""This file contains pytest fixtures for file paths in the PCNtoolkit.

The fixtures defined here include:
1. Log directory
2. Save directory
3. Response file
4. Mask file
5. Covariate file
6. Test covariate file
7. Test response file
"""


@pytest.fixture
def log_dir():
    return os.path.join(gettempdir(), "pcntoolkit_tests", "log_test")


@pytest.fixture
def save_dir():
    return os.path.join(gettempdir(), "pcntoolkit_tests", "save_load_test")


@pytest.fixture
def responsefile(n_train_datapoints, n_response_vars):
    file_path = os.path.join(
        gettempdir(), "pcntoolkit_tests", "resources", "data", "responses.csv"
    )
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    np.savetxt(file_path, generate_response_vars(n_train_datapoints, n_response_vars))
    yield file_path
    os.remove(file_path)


@pytest.fixture
def responsefile_test(n_test_datapoints, n_response_vars):
    file_path = os.path.join(
        gettempdir(), "pcntoolkit_tests", "resources", "data", "responses_test.csv"
    )
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    np.savetxt(file_path, generate_response_vars(n_test_datapoints, n_response_vars))
    yield file_path
    os.remove(file_path)


@pytest.fixture
def maskfile():
    return None


@pytest.fixture
def covfile(n_train_datapoints, n_covariates):
    file_path = os.path.join(
        gettempdir(), "pcntoolkit_tests", "resources", "data", "covariates.csv"
    )
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    np.savetxt(file_path, generate_covariates(n_train_datapoints, n_covariates))
    yield file_path
    os.remove(file_path)


@pytest.fixture
def testcov(n_test_datapoints, n_covariates):
    file_path = os.path.join(
        gettempdir(), "pcntoolkit_tests", "resources", "data", "covariates_test.csv"
    )
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    np.savetxt(file_path, generate_covariates(n_test_datapoints, n_covariates))
    yield file_path
    os.remove(file_path)


@pytest.fixture
def testresp(n_test_datapoints, n_response_vars):
    file_path = os.path.join(
        gettempdir(), "pcntoolkit_tests", "resources", "data", "responses_test.csv"
    )
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    np.savetxt(file_path, generate_response_vars(n_test_datapoints, n_response_vars))
    yield file_path
    os.remove(file_path)


@pytest.fixture
def trbefile(n_train_datapoints, batch_effect_values):
    file_path = os.path.join(
        gettempdir(), "pcntoolkit_tests", "resources", "data", "batch_effects.csv"
    )
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    np.savetxt(
        file_path, generate_batch_effects(n_train_datapoints, batch_effect_values)
    )
    yield file_path
    os.remove(file_path)


@pytest.fixture
def tsbefile(n_test_datapoints, batch_effect_values):
    file_path = os.path.join(
        gettempdir(), "pcntoolkit_tests", "resources", "data", "batch_effects_test.csv"
    )
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    np.savetxt(
        file_path, generate_batch_effects(n_test_datapoints, batch_effect_values)
    )
    yield file_path
    os.remove(file_path)
