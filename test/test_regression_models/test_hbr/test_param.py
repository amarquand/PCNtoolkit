from __future__ import annotations

import json
import logging

import numpy as np
import pymc as pm
import pytest
import xarray as xr

from pcntoolkit.normative_model.norm_hbr import NormHBR
from pcntoolkit.regression_model.hbr.hbr_data import HBRData
from pcntoolkit.regression_model.hbr.param import (  # noqa: F401, F403
    FixedParam,
    LinearParam,
    Param,
    RandomParam,
    make_param,
    param_from_args,
)
from test.fixtures.data_fixtures import *
from test.fixtures.norm_data_fixtures import *

logging.basicConfig(level=logging.INFO)


"""
This file contains tests for the make_param class in the PCNtoolkit.

The tests cover:
1. Creating make_param objects from arguments
2. Creating make_param objects from dictionaries
"""


@pytest.fixture
def data(norm_data_from_arrays):
    single_respvar = norm_data_from_arrays.sel(response_vars="response_var_1")
    data = NormHBR.normdata_to_hbrdata(single_respvar)
    return data


@pytest.fixture
def model_and_data(data: HBRData) ->  tuple[pm.Model ,HBRData]:
    model = pm.Model(coords=data.coords)
    data.set_data_in_new_model(model)
    return model, data


def test_normal_fixed_param(model_and_data):
    model, data = model_and_data
    param: FixedParam = make_param("theta", dist_name="Normal", dist_params=(0, 10)) # type: ignore
    param.create_graph(model)
    assert param.name == "theta"
    assert param.dims is None
    assert param.dist_name == "Normal"
    assert param.dist_params == (0, 10)
    samples = param.sample(data)
    assert tuple(samples.shape.eval()) == ()


def test_cauchy_fixed_param(model_and_data):
    model, data = model_and_data
    param: FixedParam = make_param("fixed2", dist_name="Cauchy") # type: ignore
    param.create_graph(model)
    assert param.name == "fixed2"
    assert param.dims is None
    assert param.dist_name == "Cauchy"
    assert param.dist_params == (0, 10)
    samples = param.sample(data)
    assert tuple(samples.shape.eval()) == ()


def test_normal_fixed_param_with_covariate_dim(model_and_data):
    model, data = model_and_data
    param: FixedParam = make_param("fixed3", dist_name="Cauchy", dims=("covariates",)) # type: ignore
    param.create_graph(model)
    assert param.name == "fixed3"
    assert param.dims == ("covariates",)
    assert param.dist_name == "Cauchy"
    assert param.dist_params == (0, 10)
    samples = param.sample(data)
    assert tuple(samples.shape.eval()) == (data.n_covariates,)


def test_random_param(model_and_data):
    model, data = model_and_data
    param: RandomParam = make_param(name="mu",random=True) # type: ignore
    param.create_graph(model)
    assert param.name == "mu"
    assert param.dims is None
    samples = param.sample(data)
    assert tuple(samples.shape.eval()) == (data.n_datapoints,)



def test_random_param_with_covariate_dim(model_and_data):
    model, data = model_and_data
    param: RandomParam = make_param("test_random3", random=True,  dims=("covariates",)) # type: ignore
    param.create_graph(model)
    samples = param.sample(data)
    assert tuple(samples.shape.eval()) == (data.n_datapoints, data.n_covariates)


def test_linear_param(model_and_data):
    model, data = model_and_data
    param: LinearParam = make_param("test_linear1", linear=True) # type: ignore
    param.create_graph(model)
    samples = param.sample(data)
    assert tuple(samples.shape.eval()) == (data.n_datapoints,)


def test_linear_param_with_random_slope(model_and_data):
    model, data = model_and_data
    slope: RandomParam = make_param("test_slope", random=True) # type: ignore
    param: LinearParam = make_param("test_linear2", linear=True, slope=slope) # type: ignore
    param.create_graph(model)
    samples = param.sample(data)
    sl = slope.sample(data)
    assert tuple(samples.shape.eval()) == (data.n_datapoints,)


def test_linear_param_with_random_intercept(model_and_data):
    model, data = model_and_data
    intercept: RandomParam = make_param("test_intercept", random=True) # type: ignore
    param: LinearParam = make_param("test_linear3", linear=True, intercept=intercept) # type: ignore
    param.create_graph(model)
    samples = param.sample(data)
    assert tuple(samples.shape.eval()) == (data.n_datapoints,)


def test_linear_param_with_random_intercept_and_slope(model_and_data):
    model, data = model_and_data
    intercept: RandomParam = make_param("test_intercept", random=True) # type: ignore
    slope: RandomParam = make_param("test_slope", random=True) # type: ignore
    param: LinearParam = make_param("test_linear4", linear=True, intercept=intercept, slope=slope) # type: ignore
    param.create_graph(model)
    samples = param.sample(data)
    assert tuple(samples.shape.eval()) == (data.n_datapoints,)


def test_params_from_args_single(model_and_data):
    model, data = model_and_data
    param_dict = {
        "dist_name_mu": "Normal",
        "dist_params_mu": (0, 1),
        "random_mu": False,
        "linear_mu": False,
        "intercept": None,
        "slope": None,
    }
    mu: FixedParam = param_from_args("mu", param_dict) # type: ignore
    mu.create_graph(model)
    assert mu.name == "mu"
    assert mu.dims is None
    assert mu.dist_name == "Normal"
    assert mu.dist_params == (0, 1)
    assert isinstance(mu, FixedParam)   

    samples = mu.sample(data)
    assert tuple(samples.shape.eval()) == ()


def test_param_from_args_single_with_covariate_dim(model_and_data):
    model, data = model_and_data
    param_dict = {
        "dist_name_mu": "Normal",
        "dist_params_mu": (0, 1),
        "random_mu": False,
        "centered_mu": False,
        "linear_mu": False,
        "intercept": None,
        "slope": None,
    }
    mu: FixedParam = param_from_args("mu", param_dict, dims=("covariates",)) # type: ignore
    mu.create_graph(model)
    assert mu.name == "mu"
    assert mu.dims == ("covariates",)
    assert mu.dist_name == "Normal"
    assert mu.dist_params == (0, 1)
    assert isinstance(mu, FixedParam)   

    assert tuple(mu.dist.shape.eval()) == (data.n_covariates,)

    samples = mu.sample(data)
    assert tuple(samples.shape.eval()) == (
        data.n_covariates,
    )


def test_two_params_from_args(model_and_data):
    model, data = model_and_data
    param_dict = {
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
        "intercept": None,
        "slope": None,
    }
    mu: FixedParam = param_from_args("mu", param_dict) # type: ignore
    mu.create_graph(model)

    assert mu.name == "mu"
    assert mu.dims is None
    assert mu.dist_name == "Normal"
    assert mu.dist_params == (0, 1)

    samples = mu.sample(data)
    assert tuple(samples.shape.eval()) == ()

    sigma: FixedParam = param_from_args("sigma", param_dict) # type: ignore
    sigma.create_graph(model)
    assert sigma.name == "sigma"
    assert sigma.dims is None
    assert sigma.dist_name == "LogNormal"
    assert sigma.dist_params == (2.0,)


    samples = sigma.sample(data)
    assert tuple(samples.shape.eval()) == ()


def test_param_from_args_random_centered(model_and_data):
    model, data = model_and_data
    param_dict = {
        "random_mu": True,
        "centered_mu": True,
        "linear_mu": False,
        "dist_params_mu_mu": (0, 3),
        "dist_name_sigma_mu": "LogNormal",
        "dist_params_sigma_mu": (2.0,),
        "intercept": None,
        "slope": None,
    }
    mu: RandomParam = param_from_args("mu", param_dict) # type: ignore
    mu.create_graph(model)

    assert mu.name == "mu"
    assert mu.dims is None
    assert mu.mu.name == "mu_mu"
    assert mu.mu.dims is None
    assert isinstance(mu.mu, FixedParam)
    assert mu.mu.dist_name == "Normal"
    assert mu.mu.dist_params == (0, 3)
    assert mu.sigma.name == "sigma_mu"
    assert mu.sigma.dims is None
    assert isinstance(mu.sigma, FixedParam)
    assert mu.sigma.dist_name == "LogNormal"
    assert mu.sigma.dist_params == (2.0,)
    assert mu.dist.name == "mu"

    samples = mu.sample(data)
    assert tuple(samples.shape.eval()) == (data.n_datapoints,)


def test_param_from_args_random_centered_with_covariate_dim(model_and_data):
    model, data = model_and_data
    param_dict = {
        "dist_name_mu": "Normal",
        "dist_params_mu": (0, 1),
        "dist_name_sigma_mu": "LogNormal",
        "dist_params_sigma_mu": (2.0,),
        "random_mu": True,
        "linear_mu": False,
        "intercept": None,
        "slope": None,
    }
    mu: RandomParam = param_from_args("mu", param_dict, dims=("covariates",)) # type: ignore
    mu.create_graph(model)
    assert mu.name == "mu"
    assert mu.dims == ("covariates",)
    assert mu.mu.name == "mu_mu"
    assert mu.mu.dims == ("covariates",)
    assert isinstance(mu.mu, FixedParam)
    assert mu.mu.dist_name == "Normal"
    assert mu.mu.dist_params == (0, 1)
    assert mu.sigma.name == "sigma_mu"
    assert mu.sigma.dims == ("covariates",)
    assert isinstance(mu.sigma, FixedParam)
    assert mu.sigma.dist_name == "LogNormal"
    assert mu.sigma.dist_params == (2.0,)
    assert mu.dist.name == "mu"

    samples = mu.sample(data)
    assert tuple(samples.shape.eval()) == (data.n_datapoints, data.n_covariates)


def test_param_from_args_random(model_and_data):
    model, data = model_and_data
    param_dict = {
        "dist_name_mu": "Normal",
        "dist_params_mu": (0, 1),
        "random_mu": True,
        "linear_mu": False,
        "intercept": None,
        "slope": None,
    }
    mu: RandomParam = param_from_args("mu", param_dict) # type: ignore
    mu.create_graph(model)
    assert mu.name == "mu"
    assert mu.dims is None
    assert isinstance(mu, RandomParam)
    assert isinstance(mu.mu, FixedParam)
    assert mu.mu.name == "mu_mu"
    assert mu.mu.dims is None
    assert mu.mu.dist_name == "Normal"
    assert mu.mu.dist_params == (0, 1)
    assert isinstance(mu.sigma, FixedParam)
    assert mu.sigma.name == "sigma_mu"
    assert mu.sigma.dims is None
    assert mu.sigma.dist_name == "HalfNormal"
    assert mu.sigma.dist_params == (1.0,)

    samples = mu.sample(data)
    assert tuple(samples.shape.eval()) == (data.n_datapoints,)



def test_param_from_args_linear(model_and_data):
    model, data = model_and_data
    param_dict = {"linear_mu": True}
    mu: LinearParam = param_from_args("mu", param_dict) # type: ignore
    mu.create_graph(model)
    assert mu.name == "mu"
    assert mu.dims is None
    assert isinstance(mu.intercept, FixedParam)
    assert mu.intercept.name == "intercept_mu"
    assert mu.intercept.dims is None
    assert mu.intercept.dist_name == "Normal"
    assert mu.intercept.dist_params == (0, 1)
    assert isinstance(mu.slope, FixedParam)
    assert mu.slope.name == "slope_mu"
    assert mu.slope.dims == ("covariates",)
    assert mu.slope.dist_name == "Normal"
    assert mu.slope.dist_params == (0, 1)

    samples = mu.sample(data)
    assert tuple(samples.shape.eval()) == (data.n_datapoints,)


def test_param_from_args_linear_with_random_slope(model_and_data):
    model, data = model_and_data
    param_dict = {"linear_mu": True, "random_slope_mu": True}
    mu: LinearParam = param_from_args("mu", param_dict) # type: ignore  
    mu.create_graph(model)
    assert mu.name == "mu"
    assert mu.dims is None
    assert isinstance(mu.intercept, FixedParam)
    assert mu.intercept.dist_name == "Normal"
    assert mu.intercept.dist_params == (0, 1)
    assert isinstance(mu.slope, RandomParam)
    assert mu.intercept.name == "intercept_mu"
    assert mu.intercept.dims is None
    assert mu.intercept.dist_name == "Normal"
    assert mu.intercept.dist_params == (0, 1)
    assert mu.slope.name == "slope_mu"
    assert mu.slope.dims == ("covariates",)
    assert isinstance(mu.slope.mu, FixedParam)
    assert mu.slope.mu.name == "mu_slope_mu"
    assert mu.slope.mu.dims == ("covariates",)
    assert mu.slope.mu.dist_name == "Normal"
    assert mu.slope.mu.dist_params == (0, 1)
    assert isinstance(mu.slope.sigma, FixedParam)
    assert mu.slope.sigma.name == "sigma_slope_mu"
    assert mu.slope.sigma.dims == ("covariates",)
    assert mu.slope.sigma.dist_name == "HalfNormal"
    assert mu.slope.sigma.dist_params == (1.0,)

    samples = mu.sample(data)
    assert tuple(samples.shape.eval()) == (data.n_datapoints,)


def test_param_from_args_linear_with_random_intercept(model_and_data):
    model, data = model_and_data
    param_dict = {"linear_mu": True, "random_intercept_mu": True}
    mu: LinearParam = param_from_args("mu", param_dict) # type: ignore
    mu.create_graph(model)
    assert mu.name == "mu"
    assert mu.dims is None
    assert isinstance(mu.slope, FixedParam)
    assert mu.slope.name == "slope_mu"
    assert mu.slope.dims == ("covariates",)
    assert mu.slope.dist_name == "Normal"
    assert mu.slope.dist_params == (0, 1)

    assert isinstance(mu.intercept, RandomParam)
    assert isinstance(mu.intercept.mu, FixedParam)
    assert mu.intercept.mu.name == "mu_intercept_mu"
    assert mu.intercept.mu.dims is None
    assert mu.intercept.mu.dist_name == "Normal"
    assert mu.intercept.mu.dist_params == (0, 1)
    assert isinstance(mu.intercept.sigma, FixedParam)
    assert mu.intercept.sigma.name == "sigma_intercept_mu"
    assert mu.intercept.sigma.dims is None
    assert mu.intercept.sigma.dist_name == "HalfNormal"
    assert mu.intercept.sigma.dist_params == (1.0,)

    samples = mu.sample(data)
    assert tuple(samples.shape.eval()) == (data.n_datapoints,)


def test_param_from_args_linear_with_random_intercept_and_slope(model_and_data):
    model, data = model_and_data
    param_dict = {
        "linear_mu": True,
        "random_intercept_mu": True,
        "random_slope_mu": True,
    }
    mu: LinearParam = param_from_args("mu", param_dict) # type: ignore
    mu.create_graph(model)
    assert mu.name == "mu"
    assert mu.dims is None
    assert isinstance(mu.intercept, RandomParam)
    assert mu.intercept.name == "intercept_mu"
    assert mu.intercept.dims is None
    assert isinstance(mu.intercept.mu, FixedParam)
    assert mu.intercept.mu.name == "mu_intercept_mu"
    assert mu.intercept.mu.dims is None
    assert mu.intercept.mu.dist_name == "Normal"
    assert mu.intercept.mu.dist_params == (0, 1)
    assert isinstance(mu.intercept.sigma, FixedParam)
    assert mu.intercept.sigma.name == "sigma_intercept_mu"
    assert mu.intercept.sigma.dims is None
    assert mu.intercept.sigma.dist_name == "HalfNormal"
    assert mu.intercept.sigma.dist_params == (1.0,)
    assert isinstance(mu.slope, RandomParam)
    assert mu.slope.name == "slope_mu"
    assert mu.slope.dims == ("covariates",)
    assert isinstance(mu.slope.mu, FixedParam)
    assert mu.slope.mu.name == "mu_slope_mu"
    assert mu.slope.mu.dims == ("covariates",)
    assert mu.slope.mu.dist_name == "Normal"
    assert mu.slope.mu.dist_params == (0, 1)
    assert isinstance(mu.slope.sigma, FixedParam)
    assert mu.slope.sigma.name == "sigma_slope_mu"
    assert mu.slope.sigma.dims == ("covariates",)
    assert mu.slope.sigma.dist_name == "HalfNormal"
    assert mu.slope.sigma.dist_params == (1.0,)

    samples = mu.sample(data)
    assert tuple(samples.shape.eval()) == (data.n_datapoints,)


def test_param_from_args_linear_with_random_centered_slope(model_and_data):
    model, data = model_and_data
    param_dict = {"linear_mu": True, "random_slope_mu": True, "centered_slope_mu": True}
    mu: LinearParam = param_from_args("mu", param_dict) # type: ignore
    mu.create_graph(model)
    assert mu.name == "mu"
    assert mu.dims is None  
    assert isinstance(mu.intercept, FixedParam)
    assert mu.intercept.name == "intercept_mu"
    assert mu.intercept.dims is None
    assert mu.intercept.dist_name == "Normal"
    assert mu.intercept.dist_params == (0, 1)
    assert isinstance(mu.slope, RandomParam)
    assert mu.slope.name == "slope_mu"
    assert mu.slope.dims == ("covariates",)
    assert isinstance(mu.slope.mu, FixedParam)
    assert mu.slope.mu.name == "mu_slope_mu"
    assert mu.slope.mu.dims == ("covariates",)
    assert mu.slope.mu.dist_name == "Normal"
    assert mu.slope.mu.dist_params == (0, 1)
    assert isinstance(mu.slope.sigma, FixedParam)
    assert mu.slope.sigma.name == "sigma_slope_mu"
    assert mu.slope.sigma.dims == ("covariates",)
    assert mu.slope.sigma.dist_name == "HalfNormal"
    assert mu.slope.sigma.dist_params == (1.0,)
        
    samples = mu.sample(data)
    assert tuple(samples.shape.eval()) == (data.n_datapoints,)


def test_param_from_args_linear_with_random_centered_intercept(model_and_data):
    model, data = model_and_data
    param_dict = {
        "linear_mu": True,
        "random_intercept_mu": True,
        "centered_intercept_mu": True,
    }
    mu: LinearParam = param_from_args("mu", param_dict) # type: ignore
    mu.create_graph(model)

    assert mu.name == "mu"
    assert mu.dims is None
    assert isinstance(mu.intercept, RandomParam)
    assert mu.intercept.name == "intercept_mu"
    assert mu.intercept.dims is None
    assert isinstance(mu.intercept.mu, FixedParam)
    assert mu.intercept.mu.name == "mu_intercept_mu"
    assert mu.intercept.mu.dims is None
    assert mu.intercept.mu.dist_name == "Normal"
    assert mu.intercept.mu.dist_params == (0, 1)
    assert isinstance(mu.intercept.sigma, FixedParam)
    assert mu.intercept.sigma.name == "sigma_intercept_mu"
    assert mu.intercept.sigma.dims is None
    assert mu.intercept.sigma.dist_name == "HalfNormal"
    assert mu.intercept.sigma.dist_params == (1.0,)

    samples = mu.sample(data)
    assert tuple(samples.shape.eval()) == (data.n_datapoints, )


def test_param_from_args_linear_with_random_centered_intercept_and_slope(
    model_and_data,
):
    model, data = model_and_data
    param_dict = {
        "linear_mu": True,
        "random_intercept_mu": True,
        "centered_intercept_mu": True,
        "random_slope_mu": True,
        "centered_slope_mu": True,
    }
    mu: LinearParam = param_from_args("mu", param_dict) # type: ignore
    mu.create_graph(model)

    assert mu.name == "mu"
    assert mu.dims is None
    assert isinstance(mu.intercept, RandomParam)
    assert mu.intercept.name == "intercept_mu"
    assert mu.intercept.dims is None
    assert isinstance(mu.intercept.mu, FixedParam)
    assert mu.intercept.mu.name == "mu_intercept_mu"
    assert mu.intercept.mu.dims is None
    assert mu.intercept.mu.dist_name == "Normal"
    assert mu.intercept.mu.dist_params == (0, 1)
    assert isinstance(mu.intercept.sigma, FixedParam)
    assert mu.intercept.sigma.name == "sigma_intercept_mu"
    assert mu.intercept.sigma.dims is None
    assert mu.intercept.sigma.dist_name == "HalfNormal"
    assert mu.intercept.sigma.dist_params == (1.0,)

    samples = mu.sample(data)
    assert tuple(samples.shape.eval()) == (data.n_datapoints,)


def test_to_dict(model_and_data):
    model, data = model_and_data
    param_dict = {
        "linear_mu": True,
        "random_intercept_mu": True,
        "centered_intercept_mu": True,
        "random_slope_mu": True,
        "centered_slope_mu": True,
    }
    mu = param_from_args("mu", param_dict)
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
def test_approximate_posterior_normal(mu, sigma):
    np.random.seed(42)
    model = pm.Model()
    samples = xr.DataArray(pm.draw(pm.Normal.dist(mu=mu, sigma=sigma), draws=10000))
    param: FixedParam = make_param("test", dims=())
    dist_name = "Normal"
    param.approximate_posterior(model, dist_name, samples)
    params = param.dist_params
    mu = params[0]
    sigma = params[1]
    assert mu == pytest.approx(mu, abs=0.01)
    assert sigma == pytest.approx(sigma, abs=0.01)


@pytest.mark.parametrize("sigma", [1, 2, 3])
def test_approximate_posterior_halfnormal(sigma):
    np.random.seed(42)
    model = pm.Model()
    samples = xr.DataArray(pm.draw(pm.HalfNormal.dist(sigma=sigma), draws=10000))
    param: FixedParam = make_param("test", dims=()) # type: ignore
    dist_name = "HalfNormal"
    param.approximate_posterior(model, dist_name, samples)
    params = param.dist_params
    sigma = params[0]
    assert sigma == pytest.approx(sigma, abs=0.01)


@pytest.mark.parametrize("mu, sigma", [(0, 1), (1, 2), (-1, 0.5)])
def test_approximate_posterior_lognormal(mu, sigma):
    np.random.seed(42)
    model = pm.Model()
    samples = xr.DataArray(pm.draw(pm.Lognormal.dist(mu=mu, sigma=sigma), draws=10000))
    param: FixedParam = make_param("test", dims=()) # type: ignore
    dist_name = "LogNormal"
    param.approximate_posterior(model, dist_name, samples)
    params = param.dist_params
    mu = params[0]
    sigma = params[1]
    assert mu == pytest.approx(mu, abs=0.01)
    assert sigma == pytest.approx(sigma, abs=0.01)


@pytest.mark.parametrize("alpha, beta", [(0, 1), (1, 2), (-1, 0.5)])
def test_approximate_posterior_cauchy(alpha, beta):
    np.random.seed(42)
    model = pm.Model()
    samples = xr.DataArray(pm.draw(pm.Cauchy.dist(alpha=alpha, beta=beta), draws=10000))
    param: FixedParam = make_param("test", dims=()) # type: ignore
    dist_name = "Cauchy"
    param.approximate_posterior(model, dist_name, samples)
    params = param.dist_params
    alpha = params[0]
    beta = params[1]
    assert alpha == pytest.approx(alpha, abs=0.01)
    assert beta == pytest.approx(beta, abs=0.01)


@pytest.mark.parametrize("beta", [1, 2, 3])
def test_approximate_posterior_halfcauchy(beta):
    np.random.seed(42)
    model = pm.Model()
    samples = xr.DataArray(pm.draw(pm.HalfCauchy.dist(beta=beta), draws=10000))
    param: FixedParam = make_param("test", dims=()) # type: ignore
    dist_name = "HalfCauchy"
    param.approximate_posterior(model, dist_name, samples)
    params = param.dist_params
    beta = params[0]
    assert beta == pytest.approx(beta, abs=0.01)


@pytest.mark.parametrize("lower, upper", [(0, 1), (1, 2), (-1, 0.5)])
def test_approximate_posterior_uniform(lower, upper):
    np.random.seed(42)
    model = pm.Model()
    samples = xr.DataArray(
        pm.draw(pm.Uniform.dist(lower=lower, upper=upper), draws=10000)
    )
    param: FixedParam = make_param("test", dims=()) # type: ignore
    dist_name = "Uniform"
    param.approximate_posterior(model, dist_name, samples)
    params = param.dist_params
    lower = params[0]
    upper = params[1]
    assert lower == pytest.approx(lower, abs=0.01)
    assert upper == pytest.approx(upper, abs=0.01)
