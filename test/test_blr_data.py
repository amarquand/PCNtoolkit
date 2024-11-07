import numpy as np

from pcntoolkit.regression_model.blr.blr_data import BLRData


def test_initialization():
    X = np.random.rand(10, 5)
    y = np.random.rand(10, 1)
    var_X = np.random.rand(10, 1)
    batch_effects = np.random.rand(10, 1)
    data = BLRData(X, y, var_X, batch_effects)
    assert data.n_covariates == 5
    assert data.n_datapoints == 10
    assert data.n_batch_effect_columns == 1

def test_missing_y():
    X = np.random.rand(10, 5)
    data = BLRData(X)
    assert np.array_equal(data.y, np.zeros((10, 1)))

def test_expand():
    X = np.random.rand(10)
    data = BLRData(X)
    assert data.X.shape == (10, 1)

def test_set_batch_effects_maps():
    X = np.random.rand(10, 5)
    data = BLRData(X)
    batch_effects_maps = {"batch1": {"A": 0, "B": 1}}
    data.set_batch_effects_maps(batch_effects_maps)
    assert data.batch_effects_maps == batch_effects_maps 