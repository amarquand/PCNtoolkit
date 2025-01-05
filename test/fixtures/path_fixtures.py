import os
from tempfile import gettempdir

import numpy as np
import pytest

from test.fixtures.data_fixtures import (
    generate_batch_effects,
    generate_covariates,
    generate_response_vars,
)

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
def fit_files(n_train_datapoints, n_covariates, n_response_vars, batch_effect_values):
    source_data_dir = os.path.join(gettempdir(), "pcntoolkit_tests", "resources", "data")
    os.makedirs(source_data_dir, exist_ok=True)
    cov_path = os.path.join(gettempdir(), "pcntoolkit_tests", "resources", "data", "covariates.csv")
    resp_path = os.path.join(gettempdir(), "pcntoolkit_tests", "resources", "data", "responses.csv")
    trbefile_path = os.path.join(gettempdir(), "pcntoolkit_tests", "resources", "data", "batch_effects.csv")
    covariates = generate_covariates(n_train_datapoints, n_covariates)
    responses = generate_response_vars(n_train_datapoints, n_response_vars, covariates)
    batch_effects = generate_batch_effects(n_train_datapoints, batch_effect_values)
    np.savetxt(cov_path, covariates)
    np.savetxt(resp_path, responses)
    np.savetxt(trbefile_path,   batch_effects)
    yield (cov_path, resp_path, trbefile_path)
    os.remove(cov_path)
    os.remove(resp_path)
    os.remove(trbefile_path)


@pytest.fixture
def test_files(n_test_datapoints, n_covariates, n_response_vars, batch_effect_values):
    source_data_dir = os.path.join(gettempdir(), "pcntoolkit_tests", "resources", "data")
    os.makedirs(source_data_dir, exist_ok=True)
    test_cov_path = os.path.join(gettempdir(), "pcntoolkit_tests", "resources", "data", "covariates_test.csv")
    test_resp_path = os.path.join(gettempdir(), "pcntoolkit_tests", "resources", "data", "responses_test.csv")
    test_trbefile_path = os.path.join(gettempdir(), "pcntoolkit_tests", "resources", "data", "batch_effects_test.csv")
    test_covariates = generate_covariates(n_test_datapoints, n_covariates)
    test_responses = generate_response_vars(n_test_datapoints, n_response_vars, test_covariates)
    test_batch_effects = generate_batch_effects(n_test_datapoints, batch_effect_values)
    np.savetxt(test_cov_path, test_covariates)
    np.savetxt(test_resp_path, test_responses)
    np.savetxt(test_trbefile_path, test_batch_effects)
    yield (test_cov_path, test_resp_path, test_trbefile_path)
    os.remove(test_cov_path)
    os.remove(test_resp_path)
    os.remove(test_trbefile_path)


@pytest.fixture
def save_dir():
    return os.path.join(gettempdir(), "pcntoolkit_tests", "save_load_test")

@pytest.fixture
def maskfile():
    return None


# @pytest.fixture
# def covfile(n_train_datapoints, n_covariates):
#     file_path = os.path.join(
#         gettempdir(), "pcntoolkit_tests", "resources", "data", "covariates.csv"
#     )
#     os.makedirs(os.path.dirname(file_path), exist_ok=True)
#     np.savetxt(file_path, generate_covariates(n_train_datapoints, n_covariates))
#     yield file_path
#     os.remove(file_path)


# @pytest.fixture
# def testcov(n_test_datapoints, n_covariates):
#     file_path = os.path.join(
#         gettempdir(), "pcntoolkit_tests", "resources", "data", "covariates_test.csv"
#     )
#     os.makedirs(os.path.dirname(file_path), exist_ok=True)
#     np.savetxt(file_path, generate_covariates(n_test_datapoints, n_covariates))
#     yield file_path
#     os.remove(file_path)


# @pytest.fixture
# def testresp(n_test_datapoints, n_response_vars):
#     file_path = os.path.join(
#         gettempdir(), "pcntoolkit_tests", "resources", "data", "responses_test.csv"
#     )
#     os.makedirs(os.path.dirname(file_path), exist_ok=True)
#     np.savetxt(file_path, generate_response_vars(n_test_datapoints, n_response_vars))
#     yield file_path
#     os.remove(file_path)


# @pytest.fixture
# def trbefile(n_train_datapoints, batch_effect_values):
#     file_path = os.path.join(
#         gettempdir(), "pcntoolkit_tests", "resources", "data", "batch_effects.csv"
#     )
#     os.makedirs(os.path.dirname(file_path), exist_ok=True)
#     np.savetxt(
#         file_path, generate_batch_effects(n_train_datapoints, batch_effect_values)
#     )
#     yield file_path
#     os.remove(file_path)


# @pytest.fixture
# def tsbefile(n_test_datapoints, batch_effect_values):
#     file_path = os.path.join(
#         gettempdir(), "pcntoolkit_tests", "resources", "data", "batch_effects_test.csv"
#     )
#     os.makedirs(os.path.dirname(file_path), exist_ok=True)
#     np.savetxt(
#         file_path, generate_batch_effects(n_test_datapoints, batch_effect_values)
#     )
#     yield file_path
#     os.remove(file_path)
