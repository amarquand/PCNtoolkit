import pytest
import os

import pymc as pm
import numpy as np
from pcntoolkit.regression_model.hbr.hbr import HBR
from pcntoolkit.regression_model.hbr.hbr_conf import HBRConf

from pcntoolkit.regression_model.hbr.hbr_data import HBRData


@pytest.fixture
def n_datapoints():
    return 100

@pytest.fixture
def n_covariates():
    return 2


@pytest.fixture
def coords():
    return {"random1": ["random1.1", "random1.2", 'random1.3'], "random2": ['random2.1', 'random2.2'], 'covariates': ['covariate1', 'covariate2']}


@pytest.fixture
def model(coords):
    return pm.Model(coords=coords)

# A fixture with X, y, and batch_effect data
@pytest.fixture
def data(n_datapoints, n_covariates):
    n_datapoints = 100
    X = np.random.rand(n_datapoints, n_covariates)
    y = np.random.rand(n_datapoints, 1)
    batch_effect1 = np.random.choice(
        ['random1.1', 'random1.2', 'random1.3'], n_datapoints)
    batch_effect2 = np.random.choice(
        ['random2.1', 'random2.2'], n_datapoints)

    data = HBRData(X, y, batch_effects=np.stack(
        [batch_effect1, batch_effect2], axis=1), batch_effect_dims = ['random1', 'random2'])
    return data

@pytest.fixture
def hbrconf():
    return HBRConf(random_mu=True)

@pytest.fixture
def hbrconf2():
    return HBRConf(linear_mu=True)

@pytest.fixture
def hbr(hbrconf:HBRConf, data:HBRData):
    hbr_model=HBR(hbrconf)
        # Create model
    hbr_model.model = pm.Model(coords=data.coords,
                            coords_mutable=data.coords_mutable)

    # Add data to pymc model
    data.add_to_pymc_model(hbr_model.model)
    return hbr_model

@pytest.fixture
def hbr2(hbrconf2:HBRConf, data:HBRData):
    hbr_model  =HBR(hbrconf2)
        # Create model
    hbr_model.model = pm.Model(coords=data.coords,
                            coords_mutable=data.coords_mutable)

    # Add data to pymc model
    data.add_to_pymc_model(hbr_model.model)
    return hbr_model


def test_fixed_param(hbr:HBR, data:HBRData, n_datapoints:int):
    prior = hbr.fixed_parameter("sigma", data, dims =())
    assert tuple(prior.shape.eval()) == (1,)
    samples = prior[data.batch_effect_indices]
    assert tuple(samples.shape.eval()) == (n_datapoints, 1)
    assert tuple(prior.get_samples(data).shape.eval()) == (n_datapoints, 1)

def test_fixed_param_with_covariate_dims(hbr:HBR, data:HBRData, n_datapoints:int, n_covariates:int):
    prior = hbr.fixed_parameter("sigma", data, dims = ('covariates',))
    assert tuple(prior.dist.shape.eval()) == (n_covariates,)
    samples = prior[data.batch_effect_indices]
    assert tuple(samples.shape.eval()) == (n_datapoints, n_covariates)
    assert tuple(prior.get_samples(data).shape.eval()) == (n_datapoints, n_covariates)

def test_fixed_param_with_random_dims(hbr:HBR, data:HBRData, n_datapoints:int):
    prior = hbr.fixed_parameter("mu", data, dims =('random1','random2'))
    assert tuple(prior.dist.shape.eval()) == (3,2)
    samples = prior[data.batch_effect_indices]
    assert tuple(samples.shape.eval()) == (n_datapoints,1)
    assert tuple(prior.get_samples(data).shape.eval()) == (n_datapoints, 1)

def test_fixed_param_with_random_and_covariate_dims(hbr:HBR, data:HBRData, n_datapoints:int, n_covariates:int):
    prior = hbr.fixed_parameter("mu", data, dims =('random1','random2', 'covariates'))
    assert tuple(prior.dist.shape.eval()) == (3,2, n_covariates)
    samples = prior[data.batch_effect_indices]
    assert tuple(samples.shape.eval()) == (n_datapoints,n_covariates)
    assert tuple(prior.get_samples(data).shape.eval()) == (n_datapoints, n_covariates)

def test_noncentered_random_parameter(hbr:HBR, data:HBRData, n_datapoints:int):
    prior = hbr.non_centered_random_parameter("mu", data, dims = ())
    assert tuple(prior.dist.shape.eval()) == (3,2)
    samples = prior[*data.batch_effect_indices]
    assert tuple(samples.shape.eval()) == (n_datapoints, 1)
    assert tuple(prior.get_samples(data).shape.eval()) == (n_datapoints, 1)

def test_noncentered_random_parameter_with_covariate_dims(hbr:HBR, data:HBRData, n_datapoints:int, n_covariates:int):
    prior = hbr.non_centered_random_parameter("mu", data, dims = ('covariates',))
    assert tuple(prior.dist.shape.eval()) == (3,2, n_covariates)
    samples = prior[*data.batch_effect_indices]
    assert tuple(samples.shape.eval()) == (n_datapoints, n_covariates)
    assert tuple(prior.get_samples(data).shape.eval()) == (n_datapoints, n_covariates)

def test_centered_random_parameter(hbr:HBR, data:HBRData, n_datapoints:int):
    prior = hbr.centered_random_parameter("mu", data, dims = ())
    assert tuple(prior.dist.shape.eval()) == (3,2)
    samples = prior[data.batch_effect_indices]
    assert tuple(samples.shape.eval()) == (n_datapoints, 1)
    assert tuple(prior.get_samples(data).shape.eval()) == (n_datapoints, 1)

def test_centered_random_parameter_with_covariate_dims(hbr:HBR, data:HBRData, n_datapoints:int, n_covariates:int):
    prior = hbr.centered_random_parameter("mu", data, dims = ('covariates',))
    assert tuple(prior.dist.shape.eval()) == (3,2, n_covariates)
    samples = prior[data.batch_effect_indices]
    assert tuple(samples.shape.eval()) == (n_datapoints, n_covariates)
    assert tuple(prior.get_samples(data).shape.eval()) == (n_datapoints, n_covariates)

def test_linear_parameter(hbr2:HBR, data:HBRData, n_datapoints:int):
    prior = hbr2.linear_parameter("sigma", data, dims=())
    with pytest.raises(NotImplementedError):
        prior.shape.eval()
    with pytest.raises(NotImplementedError):
        prior[data.batch_effect_indices]
    assert tuple(prior.get_samples(data).shape.eval()) == (n_datapoints, 1)

def test_linear_parameter_random(hbr2:HBR, data:HBRData, n_datapoints:int):
    prior = hbr2.linear_parameter("mu", data, dims = ('random1','random2'))
    with pytest.raises(NotImplementedError):
        prior.shape.eval()
    with pytest.raises(NotImplementedError):
        prior[*data.batch_effect_indices]
    assert tuple(prior.get_samples(data).shape.eval()) == (n_datapoints, 1)