import pytest
import os
import pymc as pm
import numpy as np
import xarray
import pickle

from pcntoolkit.dataio.norm_data import NormData
from pcntoolkit.normative_model.norm_hbr import NormHBR


@pytest.fixture
def n_fit_datapoints():
    return 1000


@pytest.fixture
def n_predict_datapoints():
    return 100


@pytest.fixture
def n_covariates():
    return 2


@pytest.fixture
def sample_args():
    return {'draws': 10, 'tune': 10, 'cores': 1}


@pytest.fixture
def log_dir():
    return 'pytest_tests/resources/log_test'


@pytest.fixture
def save_dir():
    return 'pytest_tests/resources/save_load_test'


@pytest.fixture
def norm_args(log_dir, save_dir):
    return {'log_dir': log_dir, 'save_dir': save_dir}


@pytest.fixture
def fit_data():
    X = np.random.randn(1000, 2)
    y = np.random.randn(1000)
    batch_effects = np.random.choice([0, 1], (1000, 2))
    return NormData.from_ndarrays('fit', X, y, batch_effects)


@pytest.fixture
def predict_data():
    X = np.random.randn(100, 2)
    y = np.random.randn(100)
    batch_effects = np.random.choice([0, 1], (100, 2))
    return NormData.from_ndarrays('predict', X, y, batch_effects)


@pytest.fixture
def transfer_data():
    X = np.random.randn(33, 2)
    y = np.random.randn(33)
    batch_effects = []
    batch_effects.append(np.random.choice([0, 1], (33, 1)))
    batch_effects.append(np.random.choice([2, 3, 4], (33, 1)))
    batch_effects = np.concatenate(batch_effects, axis=1)
    return NormData.from_ndarrays('transfer', X, y, batch_effects)


@pytest.mark.parametrize("args", [
    {'likelihood': 'Normal', 'linear_mu': False, 'random_mu': False},
    {'likelihood': 'Normal',
     'linear_mu': False, 'random_mu': True},
    {'likelihood': 'Normal', 'linear_mu': False,
        'random_mu': True, 'centered_mu': True},
    {'likelihood': 'Normal', 'linear_mu': True,
        'random_slope_mu': False, 'random_intercept_mu': False},
    {'likelihood': 'Normal', 'linear_mu': True,
        'random_slope_mu': True, 'random_intercept_mu': False},
    {'likelihood': 'Normal', 'linear_mu': True, 'random_slope_mu': True,
        'centered_slope_mu': True, 'random_intercept_mu': False},
    {'likelihood': 'Normal', 'linear_mu': True,
        'random_slope_mu': True, 'random_intercept_mu': True},
    {'likelihood': 'Normal', 'linear_mu': True, 'random_slope_mu': True,
        'centered_slope_mu': True, 'random_intercept_mu': True, 'centered_intercept_mu': True},
])
def test_normhbr_from_args(norm_args: dict[str, str], sample_args: dict[str, int], args: dict[str, str | bool]):
    hbr = NormHBR.from_args(norm_args | sample_args | args)
    assert hbr.model.conf.draws == 10
    assert hbr.model.conf.tune == 10
    assert hbr.model.conf.cores == 1
    assert hbr.model.conf.likelihood == 'Normal'
    assert hbr.model.conf.mu.linear == args.get('linear_mu', False)
    if args.get('linear_mu', False):
        assert hbr.model.conf.mu.slope.random == args.get(
            'random_slope_mu', False)
        assert hbr.model.conf.mu.intercept.random == args.get(
            'random_intercept_mu', False)
        assert hbr.model.conf.mu.slope.centered == args.get(
            'centered_slope_mu', False)
        assert hbr.model.conf.mu.intercept.centered == args.get(
            'centered_intercept_mu', False)
    assert not hbr.model.conf.sigma.linear


def test_normdata_to_hbrdata(fit_data: NormData):
    hbrdata = NormHBR.normdata_to_hbrdata(fit_data)
    assert hbrdata.X.shape == (1000, 2)
    assert hbrdata.y.shape == (1000, 1)
    assert hbrdata.batch_effects.shape == (1000, 2)
    assert tuple(hbrdata.covariate_dims) == (0, 1)
    assert tuple(hbrdata.batch_effect_dims) == (
        'batch_effect_0', 'batch_effect_1')
    assert hbrdata.batch_effects_maps == {
        'batch_effect_0': {0: 0, 1: 1}, 'batch_effect_1': {0: 0, 1: 1}}


def test_fit(norm_args: dict[str, str], fit_data: NormData, sample_args: dict[str, int]):
    hbr = NormHBR.from_args(norm_args | sample_args)
    hbr.fit(fit_data)

    assert hbr.model.is_fitted
    assert hbr.model.idata.posterior.mu.shape[:2] == (2, 10)
    assert hbr.model.idata.posterior.sigma.shape[:2] == (2, 10)


def test_predict(norm_args: dict[str, str], fit_data: NormData, predict_data: NormData, sample_args: dict[str, int]):
    hbr = NormHBR.from_args(norm_args | sample_args)
    hbr.fit(fit_data)
    hbr.predict(predict_data)
    assert hbr.model.is_fitted
    assert hbr.model.idata.posterior_predictive.y_pred.datapoints.shape == (
        100,)


def test_fit_predict(norm_args: dict[str, str], fit_data: NormData,  predict_data: NormData, sample_args: dict[str, int]):
    hbr = NormHBR.from_args(norm_args | sample_args)
    hbr.fit_predict(fit_data, predict_data)
    assert hbr.model.is_fitted
    assert hbr.model.idata.observed_data.y_pred.shape == (100, 1)


def test_transfer(norm_args: dict[str, str], fit_data: NormData, transfer_data: NormData, sample_args: dict[str, int]):
    hbr = NormHBR.from_args(norm_args | sample_args | {'random_mu': True})
    hbr.fit(fit_data)
    assert hbr.model.model.coords['batch_effect_1'] == (0, 1)
    hbr_transfered = hbr.transfer(transfer_data)
    assert hbr_transfered.model.model.coords['batch_effect_1'] == (2, 3, 4)
    assert hbr_transfered.model.is_fitted
    assert hbr_transfered.model.idata.posterior.mu.shape[:2] == (2, 10)


def test_save(norm_args: dict[str, str], fit_data: NormData, sample_args: dict[str, int]):
    hbr = NormHBR.from_args(norm_args | sample_args)
    hbr.fit(fit_data)
    os.makedirs(hbr.norm_conf.save_dir, exist_ok=True)
    hbr.save()
    assert os.path.exists(os.path.join(
        hbr.norm_conf.save_dir, 'normative_model_dict.json'))
    assert os.path.exists(os.path.join(hbr.norm_conf.save_dir, 'idata.nc'))


def test_load(save_dir):
    hbr = NormHBR.load(save_dir)
    if hbr.model.is_fitted:
        assert hbr.model.idata.posterior.mu.shape[:2] == (2, 10)
        assert hbr.model.idata.posterior.sigma.shape[:2] == (2, 10)
        assert hasattr(hbr.model.idata, 'posterior')
    assert hbr.norm_conf.save_dir == save_dir
