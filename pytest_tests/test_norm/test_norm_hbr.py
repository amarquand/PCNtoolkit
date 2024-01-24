import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
import pymc as pm
import pytest
import xarray

from pcntoolkit.dataio.norm_data import NormData
from pcntoolkit.normative_model.norm_base import NormBase
from pcntoolkit.normative_model.norm_factory import load_normative_model
from pcntoolkit.normative_model.norm_hbr import NormHBR
from pytest_tests.fixtures.data import *
from pytest_tests.fixtures.model import *
from pytest_tests.fixtures.paths import *


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
    return {"draws": 10, "tune": 10, "cores": 1}


@pytest.fixture
def fit_data():
    X = np.random.randn(1000, 2)
    y = np.random.randn(1000, 2)
    batch_effects = np.random.choice([0, 1], (1000, 2))
    return NormData.from_ndarrays("fit", X, y, batch_effects)


@pytest.fixture
def predict_data():
    X = np.random.randn(100, 2)
    y = np.random.randn(100, 2)
    batch_effects = np.random.choice([0, 1], (100, 2))
    return NormData.from_ndarrays("predict", X, y, batch_effects)


@pytest.fixture
def transfer_data():
    X = np.random.randn(33, 2)
    y = np.random.randn(33, 2)
    batch_effects = []
    batch_effects.append(np.random.choice([0, 1], (33, 1)))
    batch_effects.append(np.random.choice([2, 3, 4], (33, 1)))
    batch_effects = np.concatenate(batch_effects, axis=1)
    return NormData.from_ndarrays("transfer", X, y, batch_effects)


@pytest.mark.parametrize(
    "args",
    [
        {"likelihood": "Normal", "linear_mu": False, "random_mu": False},
        {"likelihood": "Normal", "linear_mu": False, "random_mu": True},
        {
            "likelihood": "Normal",
            "linear_mu": False,
            "random_mu": True,
            "centered_mu": True,
        },
        {
            "likelihood": "Normal",
            "linear_mu": True,
            "random_slope_mu": False,
            "random_intercept_mu": False,
        },
        {
            "likelihood": "Normal",
            "linear_mu": True,
            "random_slope_mu": True,
            "random_intercept_mu": False,
        },
        {
            "likelihood": "Normal",
            "linear_mu": True,
            "random_slope_mu": True,
            "centered_slope_mu": True,
            "random_intercept_mu": False,
        },
        {
            "likelihood": "Normal",
            "linear_mu": True,
            "random_slope_mu": True,
            "random_intercept_mu": True,
        },
        {
            "likelihood": "Normal",
            "linear_mu": True,
            "random_slope_mu": True,
            "centered_slope_mu": True,
            "random_intercept_mu": True,
            "centered_intercept_mu": True,
        },
    ],
)
def test_normhbr_from_dict(
    norm_args: dict[str, str], sample_args: dict[str, int], args: dict[str, str | bool]
):
    hbr = NormHBR.from_args(norm_args | sample_args | args)
    assert hbr.reg_conf.draws == 10
    assert hbr.reg_conf.tune == 10
    assert hbr.reg_conf.cores == 1
    assert hbr.reg_conf.likelihood == "Normal"
    assert hbr.reg_conf.mu.linear == args.get("linear_mu", False)
    if args.get("linear_mu", False):
        assert hbr.reg_conf.mu.slope.random == args.get("random_slope_mu", False)
        assert hbr.reg_conf.mu.intercept.random == args.get(
            "random_intercept_mu", False
        )
        assert hbr.reg_conf.mu.slope.centered == args.get("centered_slope_mu", False)
        assert hbr.reg_conf.mu.intercept.centered == args.get(
            "centered_intercept_mu", False
        )
    assert not hbr.reg_conf.sigma.linear


def test_normdata_to_hbrdata(fit_data: NormData):
    hbrdata = NormHBR.normdata_to_hbrdata(fit_data)
    assert hbrdata.X.shape == (1000, 2)
    assert hbrdata.y.shape == (1000, 2)
    assert hbrdata.batch_effects.shape == (1000, 2)
    assert tuple(hbrdata.covariate_dims) == ("covariate_0", "covariate_1")
    assert tuple(hbrdata.batch_effect_dims) == ("batch_effect_0", "batch_effect_1")
    assert hbrdata.batch_effects_maps == {
        "batch_effect_0": {0: 0, 1: 1},
        "batch_effect_1": {0: 0, 1: 1},
    }


def test_fit(
    norm_args: dict[str, str], fit_data: NormData, sample_args: dict[str, int]
):
    hbr = NormHBR.from_args(norm_args | sample_args)
    hbr.fit(fit_data)
    for model in hbr.models.values():
        assert model.is_fitted
        assert model.idata.posterior.mu.shape[:2] == (2, 10)
        assert model.idata.posterior.sigma.shape[:2] == (2, 10)


def test_predict(
    norm_args: dict[str, str],
    fit_data: NormData,
    predict_data: NormData,
    sample_args: dict[str, int],
):
    hbr = NormHBR.from_args(norm_args | sample_args)
    hbr.fit(fit_data)
    results = hbr.predict(predict_data)
    for model in hbr.models.values():
        assert model.is_fitted
        assert model.idata.posterior_predictive.y_pred.datapoints.shape == (100,)


def test_fit_predict(
    norm_args: dict[str, str],
    fit_data: NormData,
    predict_data: NormData,
    sample_args: dict[str, int],
):
    hbr = NormHBR.from_args(norm_args | sample_args)
    hbr.fit_predict(fit_data, predict_data)
    for model in hbr.models.values():
        assert model.is_fitted
        assert model.idata.posterior_predictive.y_pred.datapoints.shape == (100,)


def test_transfer(
    norm_args: dict[str, str],
    fit_data: NormData,
    transfer_data: NormData,
    sample_args: dict[str, int],
):
    hbr = NormHBR.from_args(norm_args | sample_args | {"random_mu": True})
    hbr.fit(fit_data)
    assert hbr.model.model.coords["batch_effect_1"] == (0, 1)
    hbr_transfered = hbr.transfer(transfer_data)
    for model in hbr_transfered.models.values():
        assert model.model.coords["batch_effect_1"] == (2, 3, 4)
        assert model.is_fitted
        assert model.idata.posterior.mu.shape[:2] == (2, 10)


def test_save(
    norm_args: dict[str, str],
    fit_data: NormData,
    sample_args: dict[str, int],
    resource_dir,
):
    norm_args["save_dir"] = os.path.join(resource_dir, "hbr", "save_load_test")
    norm_args["log_dir"] = os.path.join(resource_dir, "hbr", "log_test")
    hbr = NormHBR.from_args(norm_args | sample_args)
    hbr.fit(fit_data)
    os.makedirs(hbr.norm_conf.save_dir, exist_ok=True)
    hbr.save()
    for i in fit_data.response_vars.to_numpy().tolist():
        assert os.path.exists(os.path.join(hbr.norm_conf.save_dir, f"idata_{i}.nc"))

    assert os.path.exists(
        os.path.join(hbr.norm_conf.save_dir, "normative_model_dict.json")
    )


def test_load(resource_dir):
    load_path_1 = os.path.join(resource_dir, "hbr", "save_load_test_epmty")
    # Assert the following throws an error
    with pytest.raises(FileNotFoundError):
        load_normative_model(load_path_1)

    load_path_2 = os.path.join(resource_dir, "hbr", "save_load_test_without_idata")
    # Assert the following throws an error
    with pytest.raises(RuntimeError):
        load_normative_model(load_path_2)

    load_path3 = os.path.join(resource_dir, "hbr", "save_load_test")
    hbr = load_normative_model(load_path3)
    for model in hbr.models.values():
        if model.is_fitted:
            assert model.idata.posterior.mu.shape[:2] == (2, 10)
            assert model.idata.posterior.sigma.shape[:2] == (2, 10)
