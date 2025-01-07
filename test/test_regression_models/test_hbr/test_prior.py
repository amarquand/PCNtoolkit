from __future__ import annotations

import json
import logging

import numpy as np
import pymc as pm
import pytest
import xarray as xr

from pcntoolkit.normative_model.norm_hbr import NormHBR
from pcntoolkit.regression_model.hbr.hbr_data import HBRData
from pcntoolkit.regression_model.hbr.prior import (  # noqa: F401, F403
    BasePrior,
    LinearPrior,
    Prior,
    RandomPrior,
    make_prior,
    prior_from_args,
)
from test.fixtures.data_fixtures import *  # noqa: F401, F403
from test.fixtures.hbr_model_fixtures import *  # noqa: F401, F403
from test.fixtures.norm_data_fixtures import *  # noqa: F401, F403

logging.basicConfig(level=logging.INFO)


"""
This file contains tests for the make_prior class in the PCNtoolkit.

The tests cover:
1. Creating make_prior objects from arguments
2. Creating make_prior objects from dictionaries
"""


@pytest.fixture
def data(new_norm_hbr_model: NormHBR, norm_data_from_arrays):
    new_norm_hbr_model.register_batch_effects(norm_data_from_arrays)
    single_respvar = norm_data_from_arrays.sel(response_vars="response_var_1")
    data = new_norm_hbr_model.normdata_to_hbrdata(single_respvar)
    return data


@pytest.fixture
def model_and_data(data: HBRData) -> tuple[pm.Model, HBRData]:
    model = pm.Model(coords=data.coords)
    data.set_data_in_new_model(model)
    return model, data


def test_normal_fixed_prior(model_and_data):
    model, data = model_and_data
    prior: Prior = make_prior("theta", dist_name="Normal", dist_params=(0, 10))  # type: ignore
    prior.compile(model)
    assert prior.name == "theta"
    assert prior.dims is None
    assert prior.dist_name == "Normal"
    assert prior.dist_params == (0, 10)
    samples = prior.sample(data)
    assert tuple(samples.shape.eval()) == ()


def test_cauchy_fixed_prior(model_and_data):
    model, data = model_and_data
    prior: Prior = make_prior("fixed2", dist_name="Cauchy")  # type: ignore
    prior.compile(model)
    assert prior.name == "fixed2"
    assert prior.dims is None
    assert prior.dist_name == "Cauchy"
    assert prior.dist_params == (0, 10)
    samples = prior.sample(data)
    assert tuple(samples.shape.eval()) == ()


def test_normal_fixed_prior_with_covariate_dim(model_and_data):
    model, data = model_and_data
    prior: Prior = make_prior("fixed3", dist_name="Cauchy", dims=("covariates",))  # type: ignore
    prior.compile(model)
    assert prior.name == "fixed3"
    assert prior.dims == ("covariates",)
    assert prior.dist_name == "Cauchy"
    assert prior.dist_params == (0, 10)
    samples = prior.sample(data)
    assert tuple(samples.shape.eval()) == (data.n_covariates,)


def test_random_prior(model_and_data):
    model, data = model_and_data
    prior: RandomPrior = make_prior(name="mu", random=True)  # type: ignore
    prior.compile(model)
    assert prior.name == "mu"
    assert prior.dims is None
    samples = prior.sample(data)
    assert tuple(samples.shape.eval()) == (data.n_datapoints,)


def test_random_prior_with_covariate_dim(model_and_data):
    model, data = model_and_data
    prior: RandomPrior = make_prior("test_random3", random=True, dims=("covariates",))  # type: ignore
    prior.compile(model)
    samples = prior.sample(data)
    assert tuple(samples.shape.eval()) == (data.n_datapoints, data.n_covariates)


def test_linear_prior(model_and_data):
    model, data = model_and_data
    prior: LinearPrior = make_prior("test_linear1", linear=True)  # type: ignore
    prior.compile(model)
    samples = prior.sample(data)
    assert tuple(samples.shape.eval()) == (data.n_datapoints,)


def test_linear_prior_with_random_slope(model_and_data):
    model, data = model_and_data
    slope: RandomPrior = make_prior("test_slope", random=True)  # type: ignore
    prior: LinearPrior = make_prior("test_linear2", linear=True, slope=slope)  # type: ignore
    prior.compile(model)
    samples = prior.sample(data)
    assert tuple(samples.shape.eval()) == (data.n_datapoints,)


def test_linear_prior_with_random_intercept(model_and_data):
    model, data = model_and_data
    intercept: RandomPrior = make_prior("test_intercept", random=True)  # type: ignore
    prior: LinearPrior = make_prior("test_linear3", linear=True, intercept=intercept)  # type: ignore
    prior.compile(model)
    samples = prior.sample(data)
    assert tuple(samples.shape.eval()) == (data.n_datapoints,)


def test_linear_prior_with_random_intercept_and_slope(model_and_data):
    model, data = model_and_data
    intercept: RandomPrior = make_prior("test_intercept", random=True)  # type: ignore
    slope: RandomPrior = make_prior("test_slope", random=True)  # type: ignore
    prior: LinearPrior = make_prior("test_linear4", linear=True, intercept=intercept, slope=slope)  # type: ignore
    prior.compile(model)
    samples = prior.sample(data)
    assert tuple(samples.shape.eval()) == (data.n_datapoints,)


def test_priors_from_args_single(model_and_data):
    model, data = model_and_data
    prior_dict = {
        "dist_name_mu": "Normal",
        "dist_params_mu": (0, 1),
        "random_mu": False,
        "linear_mu": False,
        "intercept": None,
        "slope": None,
    }
    mu: Prior = prior_from_args("mu", prior_dict)  # type: ignore
    mu.compile(model)
    assert mu.name == "mu"
    assert mu.dims is None
    assert mu.dist_name == "Normal"
    assert mu.dist_params == (0, 1)
    assert isinstance(mu, Prior)

    samples = mu.sample(data)
    assert tuple(samples.shape.eval()) == ()


def test_prior_from_args_single_with_covariate_dim(model_and_data):
    model, data = model_and_data
    prior_dict = {
        "dist_name_mu": "Normal",
        "dist_params_mu": (0, 1),
        "random_mu": False,
        "centered_mu": False,
        "linear_mu": False,
        "intercept": None,
        "slope": None,
    }
    mu: Prior = prior_from_args("mu", prior_dict, dims=("covariates",))  # type: ignore
    mu.compile(model)
    assert mu.name == "mu"
    assert mu.dims == ("covariates",)
    assert mu.dist_name == "Normal"
    assert mu.dist_params == (0, 1)
    assert isinstance(mu, Prior)

    assert tuple(mu.dist.shape.eval()) == (data.n_covariates,)

    samples = mu.sample(data)
    assert tuple(samples.shape.eval()) == (data.n_covariates,)


def test_two_priors_from_args(model_and_data):
    model, data = model_and_data
    prior_dict = {
        "dist_name_mu": "Normal",
        "dist_params_mu": (0, 1),
        "dist_name_sigma": "LogNormal",
        "dist_params_sigma": (2.0,),
        "random_mu": False,
        "centered_mu": False,
        "linear_mu": False,
        "intercept": None,
        "slope": None,
        "random_sigma": False,
        "centered_sigma": False,
        "linear_sigma": False,
    }
    mu: Prior = prior_from_args("mu", prior_dict)  # type: ignore
    mu.compile(model)

    assert mu.name == "mu"
    assert mu.dims is None
    assert mu.dist_name == "Normal"
    assert mu.dist_params == (0, 1)

    samples = mu.sample(data)
    assert tuple(samples.shape.eval()) == ()

    sigma: Prior = prior_from_args("sigma", prior_dict)  # type: ignore
    sigma.compile(model)
    assert sigma.name == "sigma"
    assert sigma.dims is None
    assert sigma.dist_name == "LogNormal"
    assert sigma.dist_params == (2.0,)

    samples = sigma.sample(data)
    assert tuple(samples.shape.eval()) == ()


def test_prior_from_args_random_centered(model_and_data):
    model, data = model_and_data
    prior_dict = {
        "random_mu": True,
        "centered_mu": True,
        "linear_mu": False,
        "dist_params_mu_mu": (0, 3),
        "dist_name_sigma_mu": "LogNormal",
        "dist_params_sigma_mu": (2.0,),
        "intercept": None,
        "slope": None,
    }
    mu: RandomPrior = prior_from_args("mu", prior_dict)  # type: ignore
    mu.compile(model)

    assert mu.name == "mu"
    assert mu.dims is None
    assert mu.mu.name == "mu_mu"
    assert mu.mu.dims is None
    assert isinstance(mu.mu, Prior)
    assert mu.mu.dist_name == "Normal"
    assert mu.mu.dist_params == (0, 3)
    assert mu.sigma.name == "sigma_mu"
    assert mu.sigma.dims is None
    assert isinstance(mu.sigma, Prior)
    assert mu.sigma.dist_name == "LogNormal"
    assert mu.sigma.dist_params == (2.0,)
    assert mu.dist.name == "mu"

    samples = mu.sample(data)
    assert tuple(samples.shape.eval()) == (data.n_datapoints,)


def test_prior_from_args_random_centered_with_covariate_dim(model_and_data):
    model, data = model_and_data
    prior_dict = {
        "dist_name_mu": "Normal",
        "dist_params_mu": (0, 1),
        "dist_name_sigma_mu": "LogNormal",
        "dist_params_sigma_mu": (2.0,),
        "random_mu": True,
        "linear_mu": False,
        "intercept": None,
        "slope": None,
    }
    mu: RandomPrior = prior_from_args("mu", prior_dict, dims=("covariates",))  # type: ignore
    mu.compile(model)
    assert mu.name == "mu"
    assert mu.dims == ("covariates",)
    assert mu.mu.name == "mu_mu"
    assert mu.mu.dims == ("covariates",)
    assert isinstance(mu.mu, Prior)
    assert mu.mu.dist_name == "Normal"
    assert mu.mu.dist_params == (0, 1)
    assert mu.sigma.name == "sigma_mu"
    assert mu.sigma.dims == ("covariates",)
    assert isinstance(mu.sigma, Prior)
    assert mu.sigma.dist_name == "LogNormal"
    assert mu.sigma.dist_params == (2.0,)
    assert mu.dist.name == "mu"

    samples = mu.sample(data)
    assert tuple(samples.shape.eval()) == (data.n_datapoints, data.n_covariates)


def test_prior_from_args_random(model_and_data):
    model, data = model_and_data
    prior_dict = {
        "dist_name_mu": "Normal",
        "dist_params_mu": (0, 1),
        "random_mu": True,
        "linear_mu": False,
        "intercept": None,
        "slope": None,
    }
    mu: RandomPrior = prior_from_args("mu", prior_dict)  # type: ignore
    mu.compile(model)
    assert mu.name == "mu"
    assert mu.dims is None
    assert isinstance(mu, RandomPrior)
    assert isinstance(mu.mu, Prior)
    assert mu.mu.name == "mu_mu"
    assert mu.mu.dims is None
    assert mu.mu.dist_name == "Normal"
    assert mu.mu.dist_params == (0, 1)
    assert isinstance(mu.sigma, Prior)
    assert mu.sigma.name == "sigma_mu"
    assert mu.sigma.dims is None
    assert mu.sigma.dist_name == "HalfNormal"
    assert mu.sigma.dist_params == (1.0,)

    samples = mu.sample(data)
    assert tuple(samples.shape.eval()) == (data.n_datapoints,)


def test_prior_from_args_linear(model_and_data):
    model, data = model_and_data
    prior_dict = {"linear_mu": True}
    mu: LinearPrior = prior_from_args("mu", prior_dict)  # type: ignore
    mu.compile(model)
    assert mu.name == "mu"
    assert mu.dims is None
    assert isinstance(mu.intercept, Prior)
    assert mu.intercept.name == "intercept_mu"
    assert mu.intercept.dims is None
    assert mu.intercept.dist_name == "Normal"
    assert mu.intercept.dist_params == (0, 1)
    assert isinstance(mu.slope, Prior)
    assert mu.slope.name == "slope_mu"
    assert mu.slope.dims == ("covariates",)
    assert mu.slope.dist_name == "Normal"
    assert mu.slope.dist_params == (0, 1)

    samples = mu.sample(data)
    assert tuple(samples.shape.eval()) == (data.n_datapoints,)


def test_prior_from_args_linear_with_random_slope(model_and_data):
    model, data = model_and_data
    prior_dict = {"linear_mu": True, "random_slope_mu": True}
    mu: LinearPrior = prior_from_args("mu", prior_dict)  # type: ignore
    mu.compile(model)
    assert mu.name == "mu"
    assert mu.dims is None
    assert isinstance(mu.intercept, Prior)
    assert mu.intercept.dist_name == "Normal"
    assert mu.intercept.dist_params == (0, 1)
    assert isinstance(mu.slope, RandomPrior)
    assert mu.intercept.name == "intercept_mu"
    assert mu.intercept.dims is None
    assert mu.intercept.dist_name == "Normal"
    assert mu.intercept.dist_params == (0, 1)
    assert mu.slope.name == "slope_mu"
    assert mu.slope.dims == ("covariates",)
    assert isinstance(mu.slope.mu, Prior)
    assert mu.slope.mu.name == "mu_slope_mu"
    assert mu.slope.mu.dims == ("covariates",)
    assert mu.slope.mu.dist_name == "Normal"
    assert mu.slope.mu.dist_params == (0, 1)
    assert isinstance(mu.slope.sigma, Prior)
    assert mu.slope.sigma.name == "sigma_slope_mu"
    assert mu.slope.sigma.dims == ("covariates",)
    assert mu.slope.sigma.dist_name == "HalfNormal"
    assert mu.slope.sigma.dist_params == (1.0,)

    samples = mu.sample(data)
    assert tuple(samples.shape.eval()) == (data.n_datapoints,)


def test_prior_from_args_linear_with_random_intercept(model_and_data):
    model, data = model_and_data
    prior_dict = {"linear_mu": True, "random_intercept_mu": True}
    mu: LinearPrior = prior_from_args("mu", prior_dict)  # type: ignore
    mu.compile(model)
    assert mu.name == "mu"
    assert mu.dims is None
    assert isinstance(mu.slope, Prior)
    assert mu.slope.name == "slope_mu"
    assert mu.slope.dims == ("covariates",)
    assert mu.slope.dist_name == "Normal"
    assert mu.slope.dist_params == (0, 1)

    assert isinstance(mu.intercept, RandomPrior)
    assert isinstance(mu.intercept.mu, Prior)
    assert mu.intercept.mu.name == "mu_intercept_mu"
    assert mu.intercept.mu.dims is None
    assert mu.intercept.mu.dist_name == "Normal"
    assert mu.intercept.mu.dist_params == (0, 1)
    assert isinstance(mu.intercept.sigma, Prior)
    assert mu.intercept.sigma.name == "sigma_intercept_mu"
    assert mu.intercept.sigma.dims is None
    assert mu.intercept.sigma.dist_name == "HalfNormal"
    assert mu.intercept.sigma.dist_params == (1.0,)

    samples = mu.sample(data)
    assert tuple(samples.shape.eval()) == (data.n_datapoints,)


def test_prior_from_args_linear_with_random_intercept_and_slope(model_and_data):
    model, data = model_and_data
    prior_dict = {
        "linear_mu": True,
        "random_intercept_mu": True,
        "random_slope_mu": True,
    }
    mu: LinearPrior = prior_from_args("mu", prior_dict)  # type: ignore
    mu.compile(model)
    assert mu.name == "mu"
    assert mu.dims is None
    assert isinstance(mu.intercept, RandomPrior)
    assert mu.intercept.name == "intercept_mu"
    assert mu.intercept.dims is None
    assert isinstance(mu.intercept.mu, Prior)
    assert mu.intercept.mu.name == "mu_intercept_mu"
    assert mu.intercept.mu.dims is None
    assert mu.intercept.mu.dist_name == "Normal"
    assert mu.intercept.mu.dist_params == (0, 1)
    assert isinstance(mu.intercept.sigma, Prior)
    assert mu.intercept.sigma.name == "sigma_intercept_mu"
    assert mu.intercept.sigma.dims is None
    assert mu.intercept.sigma.dist_name == "HalfNormal"
    assert mu.intercept.sigma.dist_params == (1.0,)
    assert isinstance(mu.slope, RandomPrior)
    assert mu.slope.name == "slope_mu"
    assert mu.slope.dims == ("covariates",)
    assert isinstance(mu.slope.mu, Prior)
    assert mu.slope.mu.name == "mu_slope_mu"
    assert mu.slope.mu.dims == ("covariates",)
    assert mu.slope.mu.dist_name == "Normal"
    assert mu.slope.mu.dist_params == (0, 1)
    assert isinstance(mu.slope.sigma, Prior)
    assert mu.slope.sigma.name == "sigma_slope_mu"
    assert mu.slope.sigma.dims == ("covariates",)
    assert mu.slope.sigma.dist_name == "HalfNormal"
    assert mu.slope.sigma.dist_params == (1.0,)

    samples = mu.sample(data)
    assert tuple(samples.shape.eval()) == (data.n_datapoints,)


def test_prior_from_args_linear_with_random_centered_slope(model_and_data):
    model, data = model_and_data
    prior_dict = {"linear_mu": True, "random_slope_mu": True, "centered_slope_mu": True}
    mu: LinearPrior = prior_from_args("mu", prior_dict)  # type: ignore
    mu.compile(model)
    assert mu.name == "mu"
    assert mu.dims is None
    assert isinstance(mu.intercept, Prior)
    assert mu.intercept.name == "intercept_mu"
    assert mu.intercept.dims is None
    assert mu.intercept.dist_name == "Normal"
    assert mu.intercept.dist_params == (0, 1)
    assert isinstance(mu.slope, RandomPrior)
    assert mu.slope.name == "slope_mu"
    assert mu.slope.dims == ("covariates",)
    assert isinstance(mu.slope.mu, Prior)
    assert mu.slope.mu.name == "mu_slope_mu"
    assert mu.slope.mu.dims == ("covariates",)
    assert mu.slope.mu.dist_name == "Normal"
    assert mu.slope.mu.dist_params == (0, 1)
    assert isinstance(mu.slope.sigma, Prior)
    assert mu.slope.sigma.name == "sigma_slope_mu"
    assert mu.slope.sigma.dims == ("covariates",)
    assert mu.slope.sigma.dist_name == "HalfNormal"
    assert mu.slope.sigma.dist_params == (1.0,)

    samples = mu.sample(data)
    assert tuple(samples.shape.eval()) == (data.n_datapoints,)


def test_prior_from_args_linear_with_random_centered_intercept(model_and_data):
    model, data = model_and_data
    prior_dict = {
        "linear_mu": True,
        "random_intercept_mu": True,
        "centered_intercept_mu": True,
    }
    mu: LinearPrior = prior_from_args("mu", prior_dict)  # type: ignore
    mu.compile(model)

    assert mu.name == "mu"
    assert mu.dims is None
    assert isinstance(mu.intercept, RandomPrior)
    assert mu.intercept.name == "intercept_mu"
    assert mu.intercept.dims is None
    assert isinstance(mu.intercept.mu, Prior)
    assert mu.intercept.mu.name == "mu_intercept_mu"
    assert mu.intercept.mu.dims is None
    assert mu.intercept.mu.dist_name == "Normal"
    assert mu.intercept.mu.dist_params == (0, 1)
    assert isinstance(mu.intercept.sigma, Prior)
    assert mu.intercept.sigma.name == "sigma_intercept_mu"
    assert mu.intercept.sigma.dims is None
    assert mu.intercept.sigma.dist_name == "HalfNormal"
    assert mu.intercept.sigma.dist_params == (1.0,)

    samples = mu.sample(data)
    assert tuple(samples.shape.eval()) == (data.n_datapoints,)


def test_prior_from_args_linear_with_random_centered_intercept_and_slope(
    model_and_data,
):
    model, data = model_and_data
    prior_dict = {
        "linear_mu": True,
        "random_intercept_mu": True,
        "centered_intercept_mu": True,
        "random_slope_mu": True,
        "centered_slope_mu": True,
    }
    mu: LinearPrior = prior_from_args("mu", prior_dict)  # type: ignore
    mu.compile(model)

    assert mu.name == "mu"
    assert mu.dims is None
    assert isinstance(mu.intercept, RandomPrior)
    assert mu.intercept.name == "intercept_mu"
    assert mu.intercept.dims is None
    assert isinstance(mu.intercept.mu, Prior)
    assert mu.intercept.mu.name == "mu_intercept_mu"
    assert mu.intercept.mu.dims is None
    assert mu.intercept.mu.dist_name == "Normal"
    assert mu.intercept.mu.dist_params == (0, 1)
    assert isinstance(mu.intercept.sigma, Prior)
    assert mu.intercept.sigma.name == "sigma_intercept_mu"
    assert mu.intercept.sigma.dims is None
    assert mu.intercept.sigma.dist_name == "HalfNormal"
    assert mu.intercept.sigma.dist_params == (1.0,)

    samples = mu.sample(data)
    assert tuple(samples.shape.eval()) == (data.n_datapoints,)


def test_to_dict(model_and_data):
    model, data = model_and_data
    prior_dict = {
        "linear_mu": True,
        "random_intercept_mu": True,
        "centered_intercept_mu": True,
        "random_slope_mu": True,
        "centered_slope_mu": True,
    }
    mu = prior_from_args("mu", prior_dict)
    my_dict = mu.to_dict()
    assert my_dict["name"] == "mu"
    assert my_dict["intercept"]["name"] == "intercept_mu"
    assert my_dict["intercept"]["mu"]["name"] == "mu_intercept_mu"
    assert my_dict["intercept"]["mu"]["dist_name"] == "Normal"
    assert my_dict["intercept"]["mu"]["dist_params"] == (0, 1)
    assert my_dict["intercept"]["sigma"]["name"] == "sigma_intercept_mu"
    assert my_dict["intercept"]["sigma"]["dist_name"] == "HalfNormal"
    assert my_dict["intercept"]["sigma"]["dist_params"] == (1.0,)
    assert my_dict["slope"]["name"] == "slope_mu"
    assert my_dict["slope"]["dims"] == ("covariates",)

    # Check if json.dumps does not throw an error
    json.dumps(my_dict)


@pytest.mark.parametrize("mu, sigma", [(0, 1), (1, 2), (-1, 0.5)])
def test_approximate_normal(mu, sigma):
    np.random.seed(42)
    model = pm.Model()
    samples = xr.DataArray(pm.draw(pm.Normal.dist(mu=mu, sigma=sigma), draws=10000))
    prior: Prior = make_prior("test", dims=())
    dist_name = "Normal"
    prior.approximate(model, dist_name, samples)
    priors = prior.dist_params
    mu = priors[0]
    sigma = priors[1]
    assert mu == pytest.approx(mu, abs=0.01)
    assert sigma == pytest.approx(sigma, abs=0.01)


@pytest.mark.parametrize("sigma", [1, 2, 3])
def test_approximate_halfnormal(sigma):
    np.random.seed(42)
    model = pm.Model()
    samples = xr.DataArray(pm.draw(pm.HalfNormal.dist(sigma=sigma), draws=10000))
    Prior: Prior = make_prior("test", dims=())  # type: ignore
    dist_name = "HalfNormal"
    Prior.approximate(model, dist_name, samples)
    Priors = Prior.dist_params
    sigma = Priors[0]
    assert sigma == pytest.approx(sigma, abs=0.01)


@pytest.mark.parametrize("mu, sigma", [(0, 1), (1, 2), (-1, 0.5)])
def test_approximate_lognormal(mu, sigma):
    np.random.seed(42)
    model = pm.Model()
    samples = xr.DataArray(pm.draw(pm.Lognormal.dist(mu=mu, sigma=sigma), draws=10000))
    prior: Prior = make_prior("test", dims=())  # type: ignore
    dist_name = "LogNormal"
    prior.approximate(model, dist_name, samples)
    priors = prior.dist_params
    mu = priors[0]
    sigma = priors[1]
    assert mu == pytest.approx(mu, abs=0.01)
    assert sigma == pytest.approx(sigma, abs=0.01)


@pytest.mark.parametrize("alpha, beta", [(0, 1), (1, 2), (-1, 0.5)])
def test_approximate_cauchy(alpha, beta):
    np.random.seed(42)
    model = pm.Model()
    samples = xr.DataArray(pm.draw(pm.Cauchy.dist(alpha=alpha, beta=beta), draws=10000))
    prior: Prior = make_prior("test", dims=())  # type: ignore
    dist_name = "Cauchy"
    prior.approximate(model, dist_name, samples)
    priors = prior.dist_params
    alpha = priors[0]
    beta = priors[1]
    assert alpha == pytest.approx(alpha, abs=0.01)
    assert beta == pytest.approx(beta, abs=0.01)


@pytest.mark.parametrize("beta", [1, 2, 3])
def test_approximate_halfcauchy(beta):
    np.random.seed(42)
    model = pm.Model()
    samples = xr.DataArray(pm.draw(pm.HalfCauchy.dist(beta=beta), draws=10000))
    prior: Prior = make_prior("test", dims=())  # type: ignore
    dist_name = "HalfCauchy"
    prior.approximate(model, dist_name, samples)
    priors = prior.dist_params
    beta = priors[0]
    assert beta == pytest.approx(beta, abs=0.01)


@pytest.mark.parametrize("lower, upper", [(0, 1), (1, 2), (-1, 0.5)])
def test_approximate_uniform(lower, upper):
    np.random.seed(42)
    model = pm.Model()
    samples = xr.DataArray(pm.draw(pm.Uniform.dist(lower=lower, upper=upper), draws=10000))
    prior: Prior = make_prior("test", dims=())  # type: ignore
    dist_name = "Uniform"
    prior.approximate(model, dist_name, samples)
    priors = prior.dist_params
    lower = priors[0]
    upper = priors[1]
    assert lower == pytest.approx(lower, abs=0.01)
    assert upper == pytest.approx(upper, abs=0.01)
