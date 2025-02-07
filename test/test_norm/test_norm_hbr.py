import os

import pytest

from pcntoolkit.dataio.norm_data import NormData
from pcntoolkit.normative_model.norm_factory import load_normative_model
from pcntoolkit.normative_model.norm_hbr import NormHBR
from pcntoolkit.regression_model.hbr.likelihood import NormalLikelihood
from pcntoolkit.regression_model.hbr.prior import (  # pylint: disable=E0611
    LinearPrior,
    RandomPrior,
)
from pcntoolkit.util.plotter import plot_centiles
from test.fixtures.data_fixtures import *
from test.fixtures.hbr_model_fixtures import *
from test.fixtures.norm_data_fixtures import *
from test.fixtures.path_fixtures import *

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
            "random_intercept_mu": True,
        },
    ],
)
def test_normhbr_from_args(norm_args: dict[str, str], sample_args: dict[str, int], args: dict[str, str | bool]):
    hbr = NormHBR.from_args(norm_args | sample_args | args)
    assert hbr.template_reg_conf.draws == sample_args.get("draws")
    assert hbr.template_reg_conf.tune == sample_args.get("tune")
    assert hbr.template_reg_conf.pymc_cores == sample_args.get("pymc_cores")
    assert isinstance(hbr.template_reg_conf.likelihood, NormalLikelihood)
    if args.get("linear_mu", False):
        if args.get("random_slope_mu", False):
            assert isinstance(hbr.template_reg_conf.likelihood.mu.slope, RandomPrior)
        if args.get("random_intercept_mu", False):
            assert isinstance(hbr.template_reg_conf.likelihood.mu.intercept, RandomPrior)
    assert isinstance(hbr.template_reg_conf.likelihood.sigma, LinearPrior)


def test_normdata_to_hbrdata(
    fitted_norm_hbr_model,
    norm_data_from_arrays: NormData,
    n_train_datapoints,
    batch_effect_values,
    n_covariates,
):
    fitted_norm_hbr_model.preprocess(norm_data_from_arrays)
    single_response_var = norm_data_from_arrays.sel(response_vars="response_var_0")
    hbrdata = fitted_norm_hbr_model.normdata_to_hbrdata(single_response_var)

    assert hbrdata.X.shape == (n_train_datapoints, n_covariates * 7)
    assert hbrdata.y.shape == (n_train_datapoints,)
    assert hbrdata.response_var == "response_var_0"

    assert hbrdata.batch_effects.shape == (n_train_datapoints, len(batch_effect_values))
    assert tuple(hbrdata.batch_effect_dims) == tuple(f"batch_effect_{i}" for i in range(len(batch_effect_values)))


def test_fit(fitted_norm_hbr_model, n_mcmc_samples):
    for model in fitted_norm_hbr_model.regression_models.values():
        assert model.is_fitted
        assert model.idata.posterior.mu_samples.shape[:2] == (2, n_mcmc_samples)
        assert model.idata.posterior.sigma_samples.shape[:2] == (2, n_mcmc_samples)


def test_save_load(fitted_norm_hbr_model: NormHBR, n_mcmc_samples):
    fitted_norm_hbr_model.save()
    for i in fitted_norm_hbr_model.response_vars:
        assert os.path.exists(os.path.join(fitted_norm_hbr_model.norm_conf.save_dir, "model", f"{i}", "idata.nc"))
        assert os.path.exists(
            os.path.join(
                fitted_norm_hbr_model.norm_conf.save_dir,
                "model",
                f"{i}",
                "regression_model.json",
            )
        )
    assert os.path.exists(os.path.join(fitted_norm_hbr_model.norm_conf.save_dir, "model", "normative_model.json"))

    load_path = fitted_norm_hbr_model.norm_conf.save_dir
    hbr: NormHBR = load_normative_model(load_path)
    for model in hbr.regression_models.values():
        if model.is_fitted:
            assert model.idata.posterior.mu_samples.shape[:2] == (2, n_mcmc_samples)
            assert model.idata.posterior.sigma_samples.shape[:2] == (
                2,
                n_mcmc_samples,
            )

    # remove the files
    for i in hbr.response_vars:
        os.remove(os.path.join(load_path, "model", f"{i}", "idata.nc"))
        os.remove(os.path.join(load_path, "model", f"{i}", "regression_model.json"))
    os.remove(os.path.join(load_path, "model", "normative_model.json"))

    # Assert the following throws an error
    with pytest.raises(ValueError):
        load_normative_model(load_path)


def test_predict(
    fitted_norm_hbr_model: NormHBR,
    test_norm_data_from_arrays: NormData,
    n_test_datapoints,
):
    fitted_norm_hbr_model.predict(test_norm_data_from_arrays)
    for model in fitted_norm_hbr_model.regression_models.values():
        assert model.is_fitted
        assert model.idata.predictions.y_pred.datapoints.shape == (n_test_datapoints,)


def test_fit_predict(
    new_norm_hbr_model: NormHBR,
    norm_data_from_arrays: NormData,
    test_norm_data_from_arrays: NormData,
    n_test_datapoints,
):
    new_norm_hbr_model.fit_predict(norm_data_from_arrays, test_norm_data_from_arrays)
    for model in new_norm_hbr_model.regression_models.values():
        assert model.is_fitted
        assert model.idata.predictions.y_pred.datapoints.shape == (n_test_datapoints,)


def test_transfer(
    fitted_norm_hbr_model: NormHBR,
    transfer_norm_data_from_arrays: NormData,
    n_mcmc_samples,
):
    transfer_samples = 10
    transfer_tune = 5
    transfer_cores = 1
    transfer_init = "jitter+adapt_diag_grad"
    transfer_chains = 2
    hbr_transfered = fitted_norm_hbr_model.transfer(
        transfer_norm_data_from_arrays,
        freedom=10,
        draws=transfer_samples,
        tune=transfer_tune,
        pymc_cores=transfer_cores,
        nuts_sampler="nutpie",
        init=transfer_init,
        chains=transfer_chains,
    )
    for model in hbr_transfered.regression_models.values():
        assert model.pymc_model.coords["batch_effect_1"] == ("3",)
        assert model.is_fitted
        assert model.idata.posterior.mu_samples.shape[:2] == (
            transfer_chains,
            transfer_samples,
        )


def test_centiles(fitted_norm_hbr_model: NormHBR, test_norm_data_from_arrays: NormData):
    plot_centiles(
        model=fitted_norm_hbr_model,
        data=test_norm_data_from_arrays,
        covariate="covariate_0",
        batch_effects={"batch_effect_1": ["0"]},
        show_data=True,
        resample=True,
    )
