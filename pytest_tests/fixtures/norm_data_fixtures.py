import pytest

from pytest_tests.fixtures.data_fixtures import *
from pcntoolkit.dataio.norm_data import NormData


@pytest.fixture
def norm_data_from_arrays(train_arrays):
    X, y, batch_effects = train_arrays
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
