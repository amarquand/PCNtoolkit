from __future__ import annotations
import numpy as np
import pytest
import pymc as pm
from pcntoolkit.regression_model.hbr.hbr import HBR
from pcntoolkit.regression_model.hbr.hbr_conf import HBRConf
from pcntoolkit.regression_model.hbr.hbr_data import HBRData

from pcntoolkit.regression_model.hbr.prior import Prior
# from ...pcntoolkit.regression_model.hbr.prior import Prior


@pytest.fixture
def coords():
    return {"random1": ["random1.1", "random1.2", 'random1.3'], "random2": ['random2.1', 'random2.2'], 'covariates': ['covariate1', 'covariate2']}


@pytest.fixture
def model(coords):
    return pm.Model(coords=coords)

# A fixture with X, y, and batch_effect data


@pytest.fixture
def data():
    n_samples = 100
    X = np.random.rand(n_samples, 2)
    y = np.random.rand(n_samples, 1)
    batch_effect1 = np.random.choice(
        ['random1.1', 'random1.2', 'random1.3'], n_samples)
    batch_effect2 = np.random.choice(
        ['random2.1', 'random2.2'], n_samples)

    data = HBRData(X, y, batch_effects=np.stack(
        [batch_effect1, batch_effect2], axis=1), batch_effect_dims = ['random1', 'random2'])
    return data

@pytest.fixture
def hbrconf():
    return HBRConf(random_mu=True)

@pytest.fixture
def hbr(hbrconf:HBRConf, data:HBRData):
    model =  HBR(hbrconf)
    model.model = pm.Model(coords = data.coords, coords_mutable=data.coords_mutable)
    return model

def test_shape_1(hbr:HBR, data:HBRData):
    # Case 1: has covariates = False, has_random_effect = False
    prior = Prior(hbr, "sigma")[data.batch_effect_indices]
    assert tuple(prior.shape.eval()) == (100, 1)

def test_shape_2(hbr:HBR, data:HBRData):
    # Case 2: has covariates = True, has_random_effect = False
    prior = Prior(hbr, "sigma", dims=('covariates',))[*data.batch_effect_indices]
    assert tuple(prior.shape.eval()) == (100, 2)

def test_shape_3(hbr:HBR, data:HBRData):
    # Case 3: has covariates = False,  has_random_effect = True
    prior = Prior(hbr, "mu", dims=('random1', 'random2'))[data.batch_effect_indices]
    assert tuple(prior.shape.eval()) == (100, 1)


def test_shape_4(hbr:HBR, data:HBRData):
    # Case 4: has covariates = True, has_random_effect = True
    prior = Prior(hbr, "mu", dims=('random1', 'random2', 'covariates'))[data.batch_effect_indices]
    assert tuple(prior.shape.eval()) == (100, 2)
