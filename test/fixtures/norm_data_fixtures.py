import pytest

from pcntoolkit.dataio.norm_data import NormData
from test.fixtures.data_fixtures import *

"""
This file contains pytest fixtures for generating NormData objects in the PCNtoolkit.

The fixtures defined here include:
1. NormData objects from numpy arrays
2. NormData objects from pandas DataFrames

These fixtures are used to create consistent and controlled datasets for testing
"""


@pytest.fixture
def norm_data_from_arrays(train_arrays):
    X, y, batch_effects = train_arrays
    return NormData.from_ndarrays("from_arrays", X, y, batch_effects)


@pytest.fixture
def test_norm_data_from_arrays(test_arrays):
    X, y, batch_effects = test_arrays
    return NormData.from_ndarrays("from_arrays", X, y, batch_effects)


@pytest.fixture
def transfer_norm_data_from_arrays(transfer_arrays):
    X, y, batch_effects = transfer_arrays
    return NormData.from_ndarrays("from_arrays", X, y, batch_effects)


@pytest.fixture
def norm_data_from_dataframe(train_dataframe, n_response_vars):
    return NormData.from_dataframe(
        "from_dataframe",
        train_dataframe,
        covariates=["covariate_0", "covariate_1"],
        batch_effects=["batch_effect_0", "batch_effect_1"],
        response_vars=[f"response_var_{i}" for i in range(n_response_vars)],
    )
