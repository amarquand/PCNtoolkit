import os

import pytest

from pcntoolkit.dataio.norm_data import NormData
from pcntoolkit.normative_model.norm_base import NormBase
from pcntoolkit.normative_model.norm_factory import load_normative_model
from pcntoolkit.normative_model.norm_hbr import NormHBR
from pytest_tests.fixtures.data_fixtures import *
from pytest_tests.fixtures.model_fixtures import *
from pytest_tests.fixtures.path_fixtures import *
from pytest_tests.fixtures.norm_data_fixtures import *

"""
This file contains tests for the NormHBR class in the PCNtoolkit.

The tests cover the following aspects:
1. Creating NormHBR objects from arguments
2. Converting from NormData to HBRdata
3. Fitting the model
4. Saving and loading the model
5. Predicting with the model
6. Transfer learning
7. Computing centiles with the model
"""


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
def test_normhbr_from_args(
    norm_args: dict[str, str], sample_args: dict[str, int], args: dict[str, str | bool]
):
    hbr = NormHBR.from_args(norm_args | sample_args | args)
    assert hbr.default_reg_conf.draws == 10
    assert hbr.default_reg_conf.tune == 10
    assert hbr.default_reg_conf.cores == 1
    assert hbr.default_reg_conf.likelihood == "Normal"
    assert hbr.default_reg_conf.mu.linear == args.get("linear_mu", False)
    if args.get("linear_mu", False):
        assert hbr.default_reg_conf.mu.slope.random == args.get(
            "random_slope_mu", False
        )
        assert hbr.default_reg_conf.mu.intercept.random == args.get(
            "random_intercept_mu", False
        )
        assert hbr.default_reg_conf.mu.slope.centered == args.get(
            "centered_slope_mu", False
        )
        assert hbr.default_reg_conf.mu.intercept.centered == args.get(
            "centered_intercept_mu", False
        )
    assert not hbr.default_reg_conf.sigma.linear


def test_normdata_to_hbrdata(norm_data_from_arrays: NormData, n_train_datapoints):
    single_response_var = norm_data_from_arrays.sel(response_vars = 'response_var_0')
    hbrdata = NormHBR.normdata_to_hbrdata(single_response_var)

    assert hbrdata.X.shape == (n_train_datapoints, 2)
    assert hbrdata.y.shape == (n_train_datapoints,)
    assert hbrdata.response_var == 'response_var_0'

    assert hbrdata.batch_effects.shape == (n_train_datapoints, 2)
    assert tuple(hbrdata.covariate_dims) == ("covariate_0", "covariate_1")
    assert tuple(hbrdata.batch_effect_dims) == ("batch_effect_0", "batch_effect_1")
    assert hbrdata.batch_effects_maps == {
        "batch_effect_0": {0: 0, 1: 1},
        "batch_effect_1": {0: 0, 1: 1, 2: 2},
    }


def test_fit(fitted_norm_hbr_model, n_mcmc_samples):
    for model in fitted_norm_hbr_model.regression_models.values():
        assert model.is_fitted
        assert model.idata.posterior.mu_samples.shape[:2] == (2, n_mcmc_samples)
        assert model.idata.posterior.sigma_samples.shape[:2] == (2, n_mcmc_samples)


def test_save_load(fitted_norm_hbr_model: NormHBR, n_mcmc_samples):
    fitted_norm_hbr_model.save()
    for i in fitted_norm_hbr_model.response_vars:
        assert os.path.exists(
            os.path.join(fitted_norm_hbr_model.norm_conf.save_dir, f"idata_{i}.nc")
        )
        assert os.path.exists(
            os.path.join(fitted_norm_hbr_model.norm_conf.save_dir, f"model_{i}.json")
        )
    assert os.path.exists(
        os.path.join(fitted_norm_hbr_model.norm_conf.save_dir, "metadata.json")
    )

    load_path = fitted_norm_hbr_model.norm_conf.save_dir
    hbr = load_normative_model(load_path)
    for model in hbr.regression_models.values():
        if model.is_fitted:
            assert model.idata.posterior.mu_samples.shape[:2] == (2, n_mcmc_samples)
            assert model.idata.posterior.sigma_samples.shape[:2] == (
                2,
                n_mcmc_samples,
            )

    # remove the files
    for i in hbr.response_vars:
        os.remove(os.path.join(load_path, f"idata_{i}.nc"))
        os.remove(os.path.join(load_path, f"model_{i}.json"))
    os.remove(os.path.join(load_path, "metadata.json"))

    # Assert the following throws an error
    with pytest.raises(FileNotFoundError):
        load_normative_model(load_path)


def test_predict(
    fitted_norm_hbr_model: NormHBR,
    test_norm_data_from_arrays: NormData,
    n_test_datapoints,
):
    fitted_norm_hbr_model.predict(test_norm_data_from_arrays)
    for model in fitted_norm_hbr_model.regression_models.values():
        assert model.is_fitted
        assert model.idata.posterior_predictive.y_pred.datapoints.shape == (
            n_test_datapoints,
        )


def test_fit_predict(
    new_norm_hbr_model: NormHBR,
    norm_data_from_arrays: NormData,
    test_norm_data_from_arrays: NormData,
    n_test_datapoints,
):
    new_norm_hbr_model.fit_predict(norm_data_from_arrays, test_norm_data_from_arrays)
    for model in new_norm_hbr_model.regression_models.values():
        assert model.is_fitted
        assert model.idata.posterior_predictive.y_pred.datapoints.shape == (
            n_test_datapoints,
        )


def test_transfer(
    fitted_norm_hbr_model: NormHBR,
    transfer_norm_data_from_arrays: NormData,
    n_mcmc_samples,
):
    hbr_transfered = fitted_norm_hbr_model.transfer(transfer_norm_data_from_arrays)
    for model in hbr_transfered.regression_models.values():
        assert model.pymc_model.coords["batch_effect_1"] == (3,)
        assert model.is_fitted
        assert model.idata.posterior.mu_samples.shape[:2] == (2, n_mcmc_samples)


def test_centiles(fitted_norm_hbr_model: NormHBR, test_norm_data_from_arrays: NormData):
    fitted_norm_hbr_model.plot_centiles(
        data=test_norm_data_from_arrays,
        covariate="covariate_0",
        batch_effects={"batch_effect_1": [0]},
        show_data=True,
    )
