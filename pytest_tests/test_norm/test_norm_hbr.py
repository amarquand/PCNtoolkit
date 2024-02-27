import os

import pytest

from pcntoolkit.dataio.norm_data import NormData
from pcntoolkit.normative_model.norm_base import NormBase
from pcntoolkit.normative_model.norm_factory import load_normative_model
from pcntoolkit.normative_model.norm_hbr import NormHBR
from pytest_tests.fixtures.data import *
from pytest_tests.fixtures.model import *
from pytest_tests.fixtures.paths import *


@pytest.fixture
def sample_args():
    return {"draws": 10, "tune": 10, "cores": 1}


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


def test_normdata_to_hbrdata(train_norm_data: NormData, n_train_datapoints):
    hbrdata = NormHBR.normdata_to_hbrdata(train_norm_data)
    assert hbrdata.X.shape == (n_train_datapoints, 2)
    assert hbrdata.y.shape == (n_train_datapoints, 2)
    assert hbrdata.response_var_dims == train_norm_data.response_vars.values.tolist()
    assert hbrdata.batch_effects.shape == (n_train_datapoints, 2)
    assert tuple(hbrdata.covariate_dims) == ("X1", "X2")
    assert tuple(hbrdata.batch_effect_dims) == ("batch1", "batch2")
    assert hbrdata.batch_effects_maps == {
        "batch1": {0: 0, 1: 1},
        "batch2": {0: 0, 1: 1, 2: 2},
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
        os.path.join(
            fitted_norm_hbr_model.norm_conf.save_dir, "normative_model_dict.json"
        )
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

    # Assert the following throws an error
    with pytest.raises(RuntimeError):
        load_normative_model(load_path)

    # Remove the normative_model_dict.json file
    os.remove(os.path.join(hbr.norm_conf.save_dir, "normative_model_dict.json"))
    with pytest.raises(FileNotFoundError):
        load_normative_model(load_path)


def test_predict(
    fitted_norm_hbr_model: NormHBR,
    test_norm_data: NormData,
    n_test_datapoints,
):
    fitted_norm_hbr_model.predict(test_norm_data)
    for model in fitted_norm_hbr_model.regression_models.values():
        assert model.is_fitted
        assert model.idata.posterior_predictive.y_pred.datapoints.shape == (
            n_test_datapoints,
        )


def test_fit_predict(
    new_norm_hbr_model: NormHBR,
    train_norm_data: NormData,
    test_norm_data: NormData,
    n_test_datapoints,
):
    new_norm_hbr_model.fit_predict(train_norm_data, test_norm_data)
    for model in new_norm_hbr_model.regression_models.values():
        assert model.is_fitted
        assert model.idata.posterior_predictive.y_pred.datapoints.shape == (
            n_test_datapoints,
        )


def test_transfer(
    fitted_norm_hbr_model: NormHBR,
    transfer_norm_data: NormData,
    n_mcmc_samples,
):
    hbr_transfered = fitted_norm_hbr_model.transfer(transfer_norm_data)
    for model in hbr_transfered.regression_models.values():
        assert model.pymc_model.coords["batch2"] == (0, 1, 2)
        assert model.is_fitted
        assert model.idata.posterior.mu_samples.shape[:2] == (2, n_mcmc_samples)


def test_centiles(fitted_norm_hbr_model: NormHBR, test_norm_data: NormData):
    synth_test_norm_data = test_norm_data.create_synthetic_data(200)

    synth_test_norm_data_with_centiles = fitted_norm_hbr_model.compute_centiles(
        synth_test_norm_data
    )
    synth_test_norm_data_with_centiles.plot_centiles()
    synth_test_norm_data_with_centiles_2 = fitted_norm_hbr_model.compute_centiles(
        synth_test_norm_data_with_centiles,
        cummulative_densities=np.linspace(0, 1, 10),
    )
    synth_test_norm_data_with_centiles_2.plot_centiles()


# def test_compute_nll(fitted_norm_hbr_model, test_norm_data):
#     fitted_norm_hbr_model.create_measures_group(test_norm_data)
#     fitted_norm_hbr_model.evaluate_nll(test_norm_data)
