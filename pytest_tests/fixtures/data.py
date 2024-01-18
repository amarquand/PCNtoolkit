import numpy as np
import pandas as pd
import pytest

from pcntoolkit.dataio.norm_data import NormData


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
    batch_effects = []
    batch_effects.append(np.random.choice([0, 1], (n_datapoints, 1)))
    batch_effects.append(np.random.choice([0, 1, 2], (n_datapoints, 1)))
    batch_effects = np.concatenate(batch_effects, axis=1)
    return X, y, batch_effects


@pytest.fixture
def dataframe(np_arrays):
    X, y, batch_effects = np_arrays
    X_columns = [f"X{i+1}" for i in range(X.shape[1])]
    y_columns = [f"Y{i+1}" for i in range(y.shape[1])]
    batch_effect_columns = [f"batch{i+1}" for i in range(batch_effects.shape[1])]
    all_columns = X_columns + y_columns + batch_effect_columns
    if len(y.shape) == 1:
        y = y[:, None]
    return pd.DataFrame(
        np.concatenate([X, y, batch_effects], axis=1), columns=all_columns
    )


@pytest.fixture
def norm_data(dataframe, n_response_vars):
    norm_data = NormData.from_dataframe(
        "test",
        dataframe,
        covariates=["X1", "X2"],
        batch_effects=["batch1", "batch2"],
        response_vars=[f"Y{i+1}" for i in range(n_response_vars)],
    )
    return norm_data
