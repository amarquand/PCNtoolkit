import pytest
import os
import pymc as pm
import numpy as np
from pcntoolkit.regression_model.hbr.hbr import HBR
from pcntoolkit.regression_model.hbr.hbr_conf import HBRConf
from pcntoolkit.regression_model.hbr.hbr_data import HBRData
import matplotlib.pyplot as plt


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
    return {'n_samples': 10,'n_tune': 10,'n_cores': 1}


@pytest.fixture
def fit_data(n_fit_datapoints, n_covariates):
    X = np.random.randn(n_fit_datapoints, n_covariates)
    y = np.random.randn(n_fit_datapoints)
    batch_effects = np.random.choice([0, 1], size=(n_fit_datapoints, n_covariates))
    return HBRData(X, y, batch_effects)

@pytest.fixture
def predict_data(n_predict_datapoints, n_covariates):
    X = np.random.randn(n_predict_datapoints, n_covariates)
    batch_effects = np.random.choice([0, 1], size=(n_predict_datapoints, n_covariates))
    return HBRData(X, None, batch_effects)



@pytest.mark.parametrize("args", [
                                {'likelihood': 'Normal', 'linear_mu': False,'random_mu':False},
                                {'likelihood': 'Normal', 'linear_mu': False,'random_mu':True},
                                {'likelihood': 'Normal', 'linear_mu': False,'random_mu':True,'centered_mu':True},
                                {'likelihood': 'Normal', 'linear_mu': True, 'random_slope_mu':False, 'random_intercept_mu':False },
                                {'likelihood': 'Normal', 'linear_mu': True, 'random_slope_mu':True, 'random_intercept_mu':False },
                                {'likelihood': 'Normal', 'linear_mu': True, 'random_slope_mu':True, 'centered_slope_mu':True,'random_intercept_mu':False },
                                {'likelihood': 'Normal', 'linear_mu': True, 'random_slope_mu':True, 'random_intercept_mu':True },
                                {'likelihood': 'Normal', 'linear_mu': True, 'random_slope_mu':True, 'centered_slope_mu':True,'random_intercept_mu':True,'centered_intercept_mu':True },
                                ]) 
def test_hbr_fit_from_args(sample_args, fit_data, args):
    hbr = HBR.from_args(sample_args | args)
    assert hbr.conf.n_samples == 10
    assert hbr.conf.n_tune == 10
    assert hbr.conf.n_cores == 1
    assert hbr.conf.likelihood == 'Normal'
    assert hbr.conf.mu.linear == args.get('linear_mu', False)
    if args.get('linear_mu', False):
        assert hbr.conf.mu.slope.random == args.get('random_slope_mu', False)
        assert hbr.conf.mu.intercept.random == args.get('random_intercept_mu', False)
        assert hbr.conf.mu.slope.centered == args.get('centered_slope_mu', False)
        assert hbr.conf.mu.intercept.centered == args.get('centered_intercept_mu', False)
    assert hbr.is_from_args
    assert not hbr.conf.sigma.linear

    hbr.fit(fit_data)

    assert hbr.is_fitted
    assert hbr.idata is not None
    assert hbr.model is not None


@pytest.mark.parametrize("args", [
                                {'likelihood': 'Normal', 'linear_mu': False,'random_mu':False},
                                {'likelihood': 'Normal', 'linear_mu': False,'random_mu':True},
                                {'likelihood': 'Normal', 'linear_mu': False,'random_mu':True,'centered_mu':True},
                                {'likelihood': 'Normal', 'linear_mu': True, 'random_slope_mu':False, 'random_intercept_mu':False },
                                {'likelihood': 'Normal', 'linear_mu': True, 'random_slope_mu':True, 'random_intercept_mu':False },
                                {'likelihood': 'Normal', 'linear_mu': True, 'random_slope_mu':True, 'centered_slope_mu':True,'random_intercept_mu':False },
                                {'likelihood': 'Normal', 'linear_mu': True, 'random_slope_mu':True, 'random_intercept_mu':True },
                                {'likelihood': 'Normal', 'linear_mu': True, 'random_slope_mu':True, 'centered_slope_mu':True,'random_intercept_mu':True,'centered_intercept_mu':True },
                                ]) 
def test_hbr_fit_predict_from_args(sample_args, fit_data, predict_data, args):
    hbr = HBR.from_args(sample_args | args)
    assert hbr.conf.n_samples == 10
    assert hbr.conf.n_tune == 10
    assert hbr.conf.n_cores == 1
    assert hbr.conf.likelihood == 'Normal'
    assert hbr.conf.mu.linear == args.get('linear_mu', False)
    if args.get('linear_mu', False):
        assert hbr.conf.mu.slope.random == args.get('random_slope_mu', False)
        assert hbr.conf.mu.intercept.random == args.get('random_intercept_mu', False)
        assert hbr.conf.mu.slope.centered == args.get('centered_slope_mu', False)
        assert hbr.conf.mu.intercept.centered == args.get('centered_intercept_mu', False)
    assert hbr.is_from_args
    assert not hbr.conf.sigma.linear

    hbr.fit(fit_data)

    graph = hbr.model.to_graphviz()
    graph.render('hbr_fit_model_graph', format='png')

    assert hbr.is_fitted
    assert hbr.idata is not None
    assert hbr.model is not None
    
    mean, std = hbr.predict(predict_data)
    print(mean, std)