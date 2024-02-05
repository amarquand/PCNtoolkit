from __future__ import annotations

import json
import logging

import arviz as az
import numpy as np
import pymc as pm
import pytest

from pcntoolkit.normative_model.norm_hbr import NormHBR
from pcntoolkit.regression_model.hbr.hbr_data import HBRData
from pcntoolkit.regression_model.hbr.param import Param

logging.basicConfig(level=logging.INFO)


@pytest.fixture
def data():
    n_samples = 100
    X = np.random.rand(n_samples, 2)
    y = np.random.rand(n_samples, 1)
    batch_effect1 = np.random.choice(["random1.1", "random1.2", "random1.3"], n_samples)
    batch_effect2 = np.random.choice(["random2.1", "random2.2"], n_samples)

    data = HBRData(
        X,
        y,
        batch_effects=np.stack([batch_effect1, batch_effect2], axis=1),
        batch_effect_dims=["random1", "random2"],
    )
    return data


@pytest.fixture
def model_and_data(data: HBRData) -> (pm.Model, HBRData):
    model = pm.Model(coords=data.coords, coords_mutable=data.coords_mutable)
    data.add_to_model(model)
    return model, data


def test_normal_fixed_param(model_and_data):
    model, data = model_and_data
    param = Param("fixed")
    param.add_to(model)
    assert param.name == "fixed"
    assert param.dims == ()
    assert param.dist_name == "Normal"
    assert param.dist_params == (0, 1)
    samples = param.get_samples(data)
    assert tuple(samples.shape.eval()) == (100, 1)


def test_cauchy_fixed_param(model_and_data):
    model, data = model_and_data
    param = Param("fixed2", dist_name="Cauchy")
    param.add_to(model)
    assert param.name == "fixed2"
    assert param.dims == ()
    assert param.dist_name == "Cauchy"
    assert param.dist_params == (0, 1)
    samples = param.get_samples(data)
    assert tuple(samples.shape.eval()) == (100, 1)


def test_normal_fixed_param_with_covariate_dim(model_and_data):
    model, data = model_and_data
    param = Param("fixed3", dist_name="Cauchy", dims=("covariates",))
    param.add_to(model)
    assert param.name == "fixed3"
    assert param.dims == ("covariates",)
    assert param.dist_name == "Cauchy"
    assert param.dist_params == (0, 1)
    samples = param.get_samples(data)
    assert tuple(samples.shape.eval()) == (100, data.n_covariates)


def test_random_centered_param(model_and_data):
    model, data = model_and_data
    param = Param("mu", random=True, centered=True)
    param.add_to(model)
    assert param.name == "mu"
    assert param.dims == ()
    samples = param.get_samples(data)
    assert tuple(samples.shape.eval()) == (data.n_datapoints, 1)


def test_random_noncentered_param(model_and_data):
    model, data = model_and_data
    param = Param("test_random2", random=True, centered=False)
    param.add_to(model)
    assert param.name == "test_random2"
    assert param.dims == ()
    samples = param.get_samples(data)
    assert tuple(samples.shape.eval()) == (data.n_datapoints, 1)


def test_random_centered_param_with_covariate_dim(model_and_data):
    model, data = model_and_data
    param = Param("test_random3", random=True, centered=True, dims=("covariates",))
    param.add_to(model)
    samples = param.get_samples(data)
    assert tuple(samples.shape.eval()) == (data.n_datapoints, data.n_covariates)


def test_random_noncentered_param_with_covariate_dim(model_and_data):
    model, data = model_and_data
    param = Param("test_random4", random=True, centered=False, dims=("covariates",))
    param.add_to(model)
    samples = param.get_samples(data)
    assert tuple(samples.shape.eval()) == (data.n_datapoints, data.n_covariates)


def test_linear_param(model_and_data):
    model, data = model_and_data
    param = Param("test_linear1", linear=True)
    param.add_to(model)
    samples = param.get_samples(data)
    assert tuple(samples.shape.eval()) == (data.n_datapoints, 1)


def test_linear_param_with_random_slope(model_and_data):
    model, data = model_and_data
    slope = Param("test_slope", random=True)
    param = Param("test_linear2", linear=True, slope=slope)
    param.add_to(model)
    samples = param.get_samples(data)
    assert tuple(samples.shape.eval()) == (data.n_datapoints, 1)


def test_linear_param_with_random_intercept(model_and_data):
    model, data = model_and_data
    intercept = Param("test_intercept", random=True)
    param = Param("test_linear3", linear=True, intercept=intercept)
    param.add_to(model)
    samples = param.get_samples(data)
    assert tuple(samples.shape.eval()) == (data.n_datapoints, 1)


def test_linear_param_with_random_intercept_and_slope(model_and_data):
    model, data = model_and_data
    intercept = Param("test_intercept", random=True)
    slope = Param("test_slope", random=True)
    param = Param("test_linear4", linear=True, intercept=intercept, slope=slope)
    param.add_to(model)
    samples = param.get_samples(data)
    assert tuple(samples.shape.eval()) == (data.n_datapoints, 1)


def test_params_from_args_single(model_and_data):
    model, data = model_and_data
    param_dict = {
        "mu_dist_name": "Normal",
        "mu_dist_params": (0, 1),
        "random_mu": False,
        "centered_mu": False,
        "linear_mu": False,
        "intercept": None,
        "slope": None,
    }
    mu = Param.from_args("mu", param_dict)
    mu.add_to(model)
    assert mu.name == "mu"
    assert mu.dims == ()
    assert mu.dist_name == "Normal"
    assert mu.dist_params == (0, 1)
    assert mu.random == False
    assert mu.centered == False
    assert mu.linear == False
    assert mu.intercept == None
    assert mu.slope == None

    samples = mu.get_samples(data)
    assert tuple(samples.shape.eval()) == (100, 1)


def test_param_from_args_single_with_covariate_dim(model_and_data):
    model, data = model_and_data
    param_dict = {
        "mu_dist_name": "Normal",
        "mu_dist_params": (0, 1),
        "random_mu": False,
        "centered_mu": False,
        "linear_mu": False,
        "intercept": None,
        "slope": None,
    }
    mu = Param.from_args("mu", param_dict, dims=("covariates"))
    mu.add_to(model)
    assert mu.name == "mu"
    assert mu.dims == ("covariates",)
    assert mu.dist_name == "Normal"
    assert mu.dist_params == (0, 1)
    assert mu.random == False
    assert mu.centered == False
    assert mu.linear == False
    assert mu.intercept == None
    assert mu.slope == None

    assert tuple(mu.dist.shape.eval()) == (data.n_covariates,)

    samples = mu.get_samples(data)
    assert tuple(samples.shape.eval()) == (
        100,
        data.n_covariates,
    )


def test_param_from_args_double(model_and_data):
    model, data = model_and_data
    param_dict = {
        "mu_dist_name": "Normal",
        "mu_dist_params": (0, 1),
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
    mu = Param.from_args("mu", param_dict)
    mu.add_to(model)

    assert mu.name == "mu"
    assert mu.dims == ()
    assert mu.dist_name == "Normal"
    assert mu.dist_params == (0, 1)
    assert mu.random == False
    assert mu.centered == False
    assert mu.linear == False
    assert mu.intercept == None
    assert mu.slope == None

    samples = mu.get_samples(data)
    assert tuple(samples.shape.eval()) == (100, 1)

    sigma = Param.from_args("sigma", param_dict)
    sigma.add_to(model)
    assert sigma.name == "sigma"
    assert sigma.dims == ()
    assert sigma.dist_name == "HalfNormal"
    assert sigma.dist_params == (1.0,)
    assert sigma.random == False
    assert sigma.centered == False
    assert sigma.linear == False
    assert sigma.intercept == None
    assert sigma.slope == None

    samples = sigma.get_samples(data)
    assert tuple(samples.shape.eval()) == (100, 1)


def test_param_from_args_random_centered(model_and_data):
    model, data = model_and_data
    param_dict = {
        "mu_dist_name": "Normal",
        "mu_dist_params": (0, 1),
        "random_mu": True,
        "centered_mu": True,
        "linear_mu": False,
        "intercept": None,
        "slope": None,
    }
    mu = Param.from_args("mu", param_dict)
    mu.add_to(model)

    assert mu.name == "mu"
    assert mu.dims == ()
    assert mu.dist_name == "Normal"
    assert mu.dist_params == (0, 1)
    assert mu.random == True
    assert mu.centered == True
    assert mu.linear == False
    assert mu.intercept == None
    assert mu.slope == None
    assert mu.mu.name == "mu_mu"
    assert mu.mu.dims == ()
    assert mu.mu.dist_name == "Normal"
    assert mu.mu.dist_params == (0, 1)
    assert mu.sigma.name == "sigma_mu"
    assert mu.sigma.dims == ()
    assert mu.sigma.dist_name == "HalfNormal"
    assert mu.sigma.dist_params == (1.0,)
    assert mu.dist.name == "mu"

    assert tuple(mu.dist.shape.eval()) == (3, 2)
    samples = mu.get_samples(data)
    assert tuple(samples.shape.eval()) == (data.n_datapoints, 1)


def test_param_from_args_random_centered_with_covariate_dim(model_and_data):
    model, data = model_and_data
    param_dict = {
        "mu_dist_name": "Normal",
        "mu_dist_params": (0, 1),
        "random_mu": True,
        "centered_mu": True,
        "linear_mu": False,
        "intercept": None,
        "slope": None,
    }
    mu = Param.from_args("mu", param_dict, dims=("covariates",))
    mu.add_to(model)
    assert mu.name == "mu"
    assert mu.dims == ("covariates",)
    assert mu.dist_name == "Normal"
    assert mu.dist_params == (0, 1)
    assert mu.random == True
    assert mu.centered == True
    assert mu.linear == False
    assert mu.intercept == None
    assert mu.slope == None
    assert mu.mu.name == "mu_mu"
    assert mu.mu.dims == ("covariates",)
    assert mu.mu.dist_name == "Normal"
    assert mu.mu.dist_params == (0, 1)
    assert mu.sigma.name == "sigma_mu"
    assert mu.sigma.dims == ("covariates",)
    assert mu.sigma.dist_name == "HalfNormal"
    assert mu.sigma.dist_params == (1.0,)
    assert mu.dist.name == "mu"

    assert tuple(mu.dist.shape.eval()) == (3, 2, data.n_covariates)
    samples = mu.get_samples(data)
    assert tuple(samples.shape.eval()) == (data.n_datapoints, data.n_covariates)


def test_param_from_args_random_noncentered(model_and_data):
    model, data = model_and_data
    param_dict = {
        "mu_dist_name": "Normal",
        "mu_dist_params": (0, 1),
        "random_mu": True,
        "centered_mu": False,
        "linear_mu": False,
        "intercept": None,
        "slope": None,
    }
    mu = Param.from_args("mu", param_dict)
    mu.add_to(model)
    assert mu.name == "mu"
    assert mu.dims == ()
    assert mu.dist_name == "Normal"
    assert mu.dist_params == (0, 1)
    assert mu.random == True
    assert mu.centered == False
    assert mu.linear == False
    assert mu.intercept == None
    assert mu.slope == None
    assert mu.offset.name == "offset_mu"
    assert mu.mu.name == "mu_mu"
    assert mu.mu.dims == ()
    assert mu.mu.dist_name == "Normal"
    assert mu.mu.dist_params == (0, 1)
    assert mu.sigma.name == "sigma_mu"
    assert mu.sigma.dims == ()
    assert mu.sigma.dist_name == "HalfNormal"
    assert mu.sigma.dist_params == (1.0,)

    assert tuple(mu.dist.shape.eval()) == (3, 2)
    samples = mu.get_samples(data)
    assert tuple(samples.shape.eval()) == (data.n_datapoints, 1)


def test_param_from_args_random_noncentered_with_covariate_dim(model_and_data):
    model, data = model_and_data
    param_dict = {
        "mu_dist_name": "Normal",
        "mu_dist_params": (0, 1),
        "random_mu": True,
        "centered_mu": False,
        "linear_mu": False,
        "intercept": None,
        "slope": None,
    }
    mu = Param.from_args("mu", param_dict, dims=("covariates",))
    mu.add_to(model)
    assert mu.name == "mu"
    assert mu.dims == ("covariates",)
    assert mu.dist_name == "Normal"
    assert mu.dist_params == (0, 1)
    assert mu.random == True
    assert mu.centered == False
    assert mu.linear == False
    assert mu.intercept == None
    assert mu.slope == None
    assert mu.offset.name == "offset_mu"
    assert mu.mu.name == "mu_mu"
    assert mu.mu.dims == ("covariates",)
    assert mu.mu.dist_name == "Normal"
    assert mu.mu.dist_params == (0, 1)
    assert mu.sigma.name == "sigma_mu"
    assert mu.sigma.dims == ("covariates",)
    assert mu.sigma.dist_name == "HalfNormal"
    assert mu.sigma.dist_params == (1.0,)

    assert tuple(mu.dist.shape.eval()) == (3, 2, data.n_covariates)
    samples = mu.get_samples(data)
    assert tuple(samples.shape.eval()) == (data.n_datapoints, data.n_covariates)


def test_param_from_args_linear(model_and_data):
    model, data = model_and_data
    param_dict = {"linear_mu": True}
    mu = Param.from_args("mu", param_dict)
    mu.add_to(model)
    assert mu.name == "mu"
    assert mu.dims == ()
    assert mu.dist_name == "Normal"
    assert mu.dist_params == (0, 1)
    assert mu.random == False
    assert mu.centered == False
    assert mu.linear == True
    assert mu.intercept.name == "intercept_mu"
    assert mu.intercept.dims == ()
    assert mu.intercept.dist_name == "Normal"
    assert mu.intercept.dist_params == (0, 1)
    assert mu.slope.name == "slope_mu"
    assert mu.slope.dims == ("covariates",)
    assert mu.slope.dist_name == "Normal"
    assert mu.slope.dist_params == (0, 1)

    samples = mu.get_samples(data)
    assert tuple(samples.shape.eval()) == (data.n_datapoints, 1)


def test_param_from_args_linear_with_random_slope(model_and_data):
    model, data = model_and_data
    param_dict = {"linear_mu": True, "random_slope_mu": True}
    mu = Param.from_args("mu", param_dict)
    mu.add_to(model)
    assert mu.name == "mu"
    assert mu.dims == ()
    assert mu.dist_name == "Normal"
    assert mu.dist_params == (0, 1)
    assert mu.random == False
    assert mu.centered == False
    assert mu.linear == True
    assert mu.intercept.name == "intercept_mu"
    assert mu.intercept.dims == ()
    assert mu.intercept.dist_name == "Normal"
    assert mu.intercept.dist_params == (0, 1)
    assert mu.slope.name == "slope_mu"
    assert mu.slope.dims == ("covariates",)
    assert mu.slope.dist_name == "Normal"
    assert mu.slope.dist_params == (0, 1)
    assert mu.slope.random == True
    assert mu.slope.centered == False
    assert mu.slope.linear == False
    assert mu.slope.intercept == None
    assert mu.slope.slope == None
    assert mu.slope.mu.name == "mu_slope_mu"
    assert mu.slope.mu.dims == ("covariates",)
    assert mu.slope.mu.dist_name == "Normal"
    assert mu.slope.mu.dist_params == (0, 1)
    assert mu.slope.sigma.name == "sigma_slope_mu"
    assert mu.slope.sigma.dims == ("covariates",)
    assert mu.slope.sigma.dist_name == "HalfNormal"
    assert mu.slope.sigma.dist_params == (1.0,)
    assert mu.slope.offset.name == "offset_slope_mu"

    samples = mu.get_samples(data)
    assert tuple(samples.shape.eval()) == (data.n_datapoints, 1)


def test_param_from_args_linear_with_random_intercept(model_and_data):
    model, data = model_and_data
    param_dict = {"linear_mu": True, "random_intercept_mu": True}
    mu = Param.from_args("mu", param_dict)
    mu.add_to(model)
    assert mu.name == "mu"
    assert mu.dims == ()
    assert mu.dist_name == "Normal"
    assert mu.dist_params == (0, 1)
    assert mu.random == False
    assert mu.centered == False
    assert mu.linear == True
    assert mu.intercept.name == "intercept_mu"
    assert mu.intercept.dims == ()
    assert mu.intercept.dist_name == "Normal"
    assert mu.intercept.dist_params == (0, 1)
    assert mu.intercept.random == True
    assert mu.intercept.centered == False
    assert mu.intercept.linear == False
    assert mu.intercept.intercept == None
    assert mu.intercept.slope == None
    assert mu.intercept.mu.name == "mu_intercept_mu"
    assert mu.intercept.mu.dims == ()
    assert mu.intercept.mu.dist_name == "Normal"
    assert mu.intercept.mu.dist_params == (0, 1)
    assert mu.intercept.sigma.name == "sigma_intercept_mu"
    assert mu.intercept.sigma.dims == ()
    assert mu.intercept.sigma.dist_name == "HalfNormal"
    assert mu.intercept.sigma.dist_params == (1.0,)
    assert mu.intercept.offset.name == "offset_intercept_mu"
    assert mu.slope.name == "slope_mu"
    assert mu.slope.dims == ("covariates",)
    assert mu.slope.dist_name == "Normal"
    assert mu.slope.dist_params == (0, 1)

    samples = mu.get_samples(data)
    assert tuple(samples.shape.eval()) == (data.n_datapoints, 1)


def test_param_from_args_linear_with_random_intercept_and_slope(model_and_data):
    model, data = model_and_data
    param_dict = {
        "linear_mu": True,
        "random_intercept_mu": True,
        "random_slope_mu": True,
    }
    mu = Param.from_args("mu", param_dict)
    mu.add_to(model)
    assert mu.name == "mu"
    assert mu.dims == ()
    assert mu.dist_name == "Normal"
    assert mu.dist_params == (0, 1)
    assert mu.random == False
    assert mu.centered == False
    assert mu.linear == True
    assert mu.intercept.name == "intercept_mu"
    assert mu.intercept.dims == ()
    assert mu.intercept.dist_name == "Normal"
    assert mu.intercept.dist_params == (0, 1)
    assert mu.intercept.random == True
    assert mu.intercept.centered == False
    assert mu.intercept.linear == False
    assert mu.intercept.intercept == None
    assert mu.intercept.slope == None
    assert mu.intercept.mu.name == "mu_intercept_mu"
    assert mu.intercept.mu.dims == ()
    assert mu.intercept.mu.dist_name == "Normal"
    assert mu.intercept.mu.dist_params == (0, 1)
    assert mu.intercept.sigma.name == "sigma_intercept_mu"
    assert mu.intercept.sigma.dims == ()
    assert mu.intercept.sigma.dist_name == "HalfNormal"
    assert mu.intercept.sigma.dist_params == (1.0,)
    assert mu.intercept.offset.name == "offset_intercept_mu"
    assert mu.slope.name == "slope_mu"
    assert mu.slope.dims == ("covariates",)
    assert mu.slope.dist_name == "Normal"
    assert mu.slope.dist_params == (0, 1)
    assert mu.slope.random == True
    assert mu.slope.centered == False
    assert mu.slope.linear == False
    assert mu.slope.intercept == None
    assert mu.slope.slope == None
    assert mu.slope.mu.name == "mu_slope_mu"
    assert mu.slope.mu.dims == ("covariates",)
    assert mu.slope.mu.dist_name == "Normal"
    assert mu.slope.mu.dist_params == (0, 1)
    assert mu.slope.sigma.name == "sigma_slope_mu"
    assert mu.slope.sigma.dims == ("covariates",)
    assert mu.slope.sigma.dist_name == "HalfNormal"
    assert mu.slope.sigma.dist_params == (1.0,)
    assert mu.slope.offset.name == "offset_slope_mu"

    samples = mu.get_samples(data)
    assert tuple(samples.shape.eval()) == (data.n_datapoints, 1)


def test_param_from_args_linear_with_random_centered_slope(model_and_data):
    model, data = model_and_data
    param_dict = {"linear_mu": True, "random_slope_mu": True, "centered_slope_mu": True}
    mu = Param.from_args("mu", param_dict)
    mu.add_to(model)
    assert mu.name == "mu"
    assert mu.dims == ()
    assert mu.dist_name == "Normal"
    assert mu.dist_params == (0, 1)
    assert mu.random == False
    assert mu.centered == False
    assert mu.linear == True
    assert mu.intercept.name == "intercept_mu"
    assert mu.intercept.dims == ()
    assert mu.intercept.dist_name == "Normal"
    assert mu.intercept.dist_params == (0, 1)
    assert mu.slope.name == "slope_mu"
    assert mu.slope.dims == ("covariates",)
    assert mu.slope.dist_name == "Normal"
    assert mu.slope.dist_params == (0, 1)
    assert mu.slope.random == True
    assert mu.slope.centered == True
    assert mu.slope.linear == False
    assert mu.slope.intercept == None
    assert mu.slope.slope == None
    assert mu.slope.mu.name == "mu_slope_mu"
    assert mu.slope.mu.dims == ("covariates",)
    assert mu.slope.mu.dist_name == "Normal"
    assert mu.slope.mu.dist_params == (0, 1)
    assert mu.slope.sigma.name == "sigma_slope_mu"
    assert mu.slope.sigma.dims == ("covariates",)
    assert mu.slope.sigma.dist_name == "HalfNormal"
    assert mu.slope.sigma.dist_params == (1.0,)
    assert mu.slope.dist.name == "slope_mu"
    assert tuple(mu.slope.dist.shape.eval()) == (3, 2, data.n_covariates)

    samples = mu.get_samples(data)
    assert tuple(samples.shape.eval()) == (data.n_datapoints, 1)


def test_param_from_args_linear_with_random_centered_intercept(model_and_data):
    model, data = model_and_data
    param_dict = {
        "linear_mu": True,
        "random_intercept_mu": True,
        "centered_intercept_mu": True,
    }
    mu = Param.from_args("mu", param_dict)
    mu.add_to(model)

    assert mu.name == "mu"
    assert mu.dims == ()
    assert mu.dist_name == "Normal"
    assert mu.dist_params == (0, 1)
    assert mu.random == False
    assert mu.centered == False
    assert mu.linear == True
    assert mu.intercept.name == "intercept_mu"
    assert mu.intercept.dims == ()
    assert mu.intercept.dist_name == "Normal"
    assert mu.intercept.dist_params == (0, 1)
    assert mu.intercept.random == True
    assert mu.intercept.centered == True
    assert mu.intercept.linear == False
    assert mu.intercept.intercept == None
    assert mu.intercept.slope == None
    assert mu.intercept.mu.name == "mu_intercept_mu"
    assert mu.intercept.mu.dims == ()
    assert mu.intercept.mu.dist_name == "Normal"
    assert mu.intercept.mu.dist_params == (0, 1)
    assert mu.intercept.sigma.name == "sigma_intercept_mu"
    assert mu.intercept.sigma.dims == ()
    assert mu.intercept.sigma.dist_name == "HalfNormal"
    assert mu.intercept.sigma.dist_params == (1.0,)
    assert tuple(mu.intercept.dist.shape.eval()) == (3, 2)

    samples = mu.get_samples(data)
    assert tuple(samples.shape.eval()) == (data.n_datapoints, 1)


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
    mu = Param.from_args("mu", param_dict)
    mu.add_to(model)

    assert mu.name == "mu"
    assert mu.dims == ()
    assert mu.dist_name == "Normal"
    assert mu.dist_params == (0, 1)
    assert mu.random == False
    assert mu.centered == False
    assert mu.linear == True
    assert mu.intercept.name == "intercept_mu"
    assert mu.intercept.dims == ()
    assert mu.intercept.dist_name == "Normal"
    assert mu.intercept.dist_params == (0, 1)
    assert mu.intercept.random == True
    assert mu.intercept.centered == True
    assert mu.intercept.linear == False
    assert mu.intercept.intercept == None
    assert mu.intercept.slope == None
    assert mu.intercept.mu.name == "mu_intercept_mu"
    assert mu.intercept.mu.dims == ()
    assert mu.intercept.mu.dist_name == "Normal"
    assert mu.intercept.mu.dist_params == (0, 1)
    assert mu.intercept.sigma.name == "sigma_intercept_mu"
    assert mu.intercept.sigma.dims == ()
    assert mu.intercept.sigma.dist_name == "HalfNormal"
    assert mu.intercept.sigma.dist_params == (1.0,)
    assert tuple(mu.intercept.dist.shape.eval()) == (3, 2)
    assert mu.slope.name == "slope_mu"
    assert mu.slope.dims == ("covariates",)
    assert mu.slope.dist_name == "Normal"
    assert mu.slope.dist_params == (0, 1)
    assert mu.slope.random == True
    assert mu.slope.centered == True
    assert mu.slope.linear == False
    assert mu.slope.intercept == None
    assert mu.slope.slope == None
    assert mu.slope.mu.name == "mu_slope_mu"
    assert mu.slope.mu.dims == ("covariates",)
    assert mu.slope.mu.dist_name == "Normal"
    assert mu.slope.mu.dist_params == (0, 1)
    assert mu.slope.sigma.name == "sigma_slope_mu"
    assert mu.slope.sigma.dims == ("covariates",)
    assert mu.slope.sigma.dist_name == "HalfNormal"
    assert mu.slope.sigma.dist_params == (1.0,)
    assert mu.slope.dist.name == "slope_mu"
    assert tuple(mu.slope.dist.shape.eval()) == (3, 2, data.n_covariates)

    samples = mu.get_samples(data)
    assert tuple(samples.shape.eval()) == (data.n_datapoints, 1)


def test_to_dict(model_and_data):
    model, data = model_and_data
    param_dict = {
        "linear_mu": True,
        "random_intercept_mu": True,
        "centered_intercept_mu": True,
        "random_slope_mu": True,
        "centered_slope_mu": True,
    }
    mu = Param.from_args("mu", param_dict)
    my_dict = mu.to_dict()
    assert my_dict["name"] == "mu"
    assert my_dict["dims"] == ()
    assert my_dict["random"] == False
    assert my_dict["centered"] == False
    assert my_dict["linear"] == True
    assert my_dict["intercept"]["name"] == "intercept_mu"
    assert my_dict["intercept"]["dims"] == ()
    assert my_dict["intercept"]["random"] == True
    assert my_dict["intercept"]["centered"] == True
    assert my_dict["intercept"]["linear"] == False
    assert my_dict["intercept"]["mu"]["name"] == "mu_intercept_mu"
    assert my_dict["intercept"]["mu"]["dims"] == ()
    assert my_dict["intercept"]["mu"]["dist_name"] == "Normal"
    assert my_dict["intercept"]["mu"]["dist_params"] == (0, 1)
    assert my_dict["intercept"]["sigma"]["name"] == "sigma_intercept_mu"
    assert my_dict["intercept"]["sigma"]["dims"] == ()
    assert my_dict["intercept"]["sigma"]["dist_name"] == "HalfNormal"
    assert my_dict["intercept"]["sigma"]["dist_params"] == (1.0,)
    assert my_dict["slope"]["name"] == "slope_mu"
    assert my_dict["slope"]["dims"] == ("covariates",)
    assert my_dict["slope"]["random"] == True
    assert my_dict["slope"]["centered"] == True
    assert my_dict["slope"]["linear"] == False

    # Check if json.dumps does not throw an error
    json.dumps(my_dict)


@pytest.mark.parametrize("mu, sigma", [(0, 1), (1, 2), (-1, 0.5)])
def test_approximate_marginal_normal(mu, sigma):
    np.random.seed(42)
    model = pm.Model()
    samples = pm.draw(pm.Normal.dist(mu=mu, sigma=sigma), draws=10000)
    param = Param("test", dims=())
    dist_name = "Normal"
    param.approximate_marginal(model, dist_name, samples)
    params = param.dist_params
    mu = params[0]
    sigma = params[1]
    assert mu == pytest.approx(mu, abs=0.01)
    assert sigma == pytest.approx(sigma, abs=0.01)


@pytest.mark.parametrize("sigma", [1, 2, 3])
def test_approximate_marginal_halfnormal(sigma):
    np.random.seed(42)
    model = pm.Model()
    samples = pm.draw(pm.HalfNormal.dist(sigma=sigma), draws=10000)
    param = Param("test", dims=())
    dist_name = "HalfNormal"
    param.approximate_marginal(model, dist_name, samples)
    params = param.dist_params
    sigma = params[0]
    assert sigma == pytest.approx(sigma, abs=0.01)


@pytest.mark.parametrize("mu, sigma", [(0, 1), (1, 2), (-1, 0.5)])
def test_approximate_marginal_lognormal(mu, sigma):
    np.random.seed(42)
    model = pm.Model()
    samples = pm.draw(pm.Lognormal.dist(mu=mu, sigma=sigma), draws=10000)
    param = Param("test", dims=())
    dist_name = "LogNormal"
    param.approximate_marginal(model, dist_name, samples)
    params = param.dist_params
    mu = params[0]
    sigma = params[1]
    assert mu == pytest.approx(mu, abs=0.01)
    assert sigma == pytest.approx(sigma, abs=0.01)


@pytest.mark.parametrize("alpha, beta", [(0, 1), (1, 2), (-1, 0.5)])
def test_approximate_marginal_cauchy(alpha, beta):
    np.random.seed(42)
    model = pm.Model()
    samples = pm.draw(pm.Cauchy.dist(alpha=alpha, beta=beta), draws=10000)
    param = Param("test", dims=())
    dist_name = "Cauchy"
    param.approximate_marginal(model, dist_name, samples)
    params = param.dist_params
    alpha = params[0]
    beta = params[1]
    assert alpha == pytest.approx(alpha, abs=0.01)
    assert beta == pytest.approx(beta, abs=0.01)


@pytest.mark.parametrize("beta", [1, 2, 3])
def test_approximate_marginal_halfcauchy(beta):
    np.random.seed(42)
    model = pm.Model()
    samples = pm.draw(pm.HalfCauchy.dist(beta=beta), draws=10000)
    param = Param("test", dims=())
    dist_name = "HalfCauchy"
    param.approximate_marginal(model, dist_name, samples)
    params = param.dist_params
    beta = params[0]
    assert beta == pytest.approx(beta, abs=0.01)


@pytest.mark.parametrize("lower, upper", [(0, 1), (1, 2), (-1, 0.5)])
def test_approximate_marginal_uniform(lower, upper):
    np.random.seed(42)
    model = pm.Model()
    samples = pm.draw(pm.Uniform.dist(lower=lower, upper=upper), draws=10000)
    param = Param("test", dims=())
    dist_name = "Uniform"
    param.approximate_marginal(model, dist_name, samples)
    params = param.dist_params
    lower = params[0]
    upper = params[1]
    assert lower == pytest.approx(lower, abs=0.01)
    assert upper == pytest.approx(upper, abs=0.01)
