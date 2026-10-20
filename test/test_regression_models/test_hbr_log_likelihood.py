"""Test likelihood extraction with known draws, without stochastic fitting."""

from pathlib import Path

import numpy as np
import pymc as pm
import pytest
import xarray as xr
from scipy.stats import norm

from pcntoolkit.math_functions.likelihood import NormalLikelihood, SHASHbLikelihood
from pcntoolkit.math_functions.prior import make_prior
from pcntoolkit.math_functions.shash import SHASHb
from pcntoolkit.regression_model.hbr import HBR


def make_data() -> tuple[xr.DataArray, xr.DataArray, xr.DataArray]:
    observations = ["p3", "p1", "p2"]
    X = xr.DataArray(
        [[0.0], [1.0], [2.0]],
        dims=("observations", "covariates"),
        coords={"observations": observations, "covariates": ["age"]},
    )
    be = xr.DataArray(
        np.zeros((3, 1), dtype=int),
        dims=("observations", "batch_effect_dims"),
        coords={"observations": observations, "batch_effect_dims": ["site"]},
    )
    Y = xr.DataArray(
        [-0.8, 0.3, 1.7], dims="observations", coords={"observations": observations}
    )
    return X, be, Y


def make_model(shash: bool = False) -> HBR:
    params = {
        "mu": make_prior(),
        "sigma": make_prior(dist_name="HalfNormal", dist_params=(1.0,)),
    }
    draws = {
        "mu": np.array([[-0.4, 0.2, 0.8], [0.1, 0.6, -0.2]]),
        "sigma": np.array([[0.7, 1.1, 1.4], [0.9, 1.6, 1.2]]),
    }
    if shash:
        params.update(
            epsilon=make_prior(),
            delta=make_prior(dist_name="HalfNormal", dist_params=(1.0,)),
        )
        draws.update(
            epsilon=np.array([[0.2, -0.3, 0.5], [-0.1, 0.4, -0.2]]),
            delta=np.array([[0.8, 1.2, 1.5], [0.9, 1.1, 1.3]]),
        )
    likelihood = SHASHbLikelihood(**params) if shash else NormalLikelihood(**params)
    model = HBR("test", likelihood=likelihood)
    model.be_maps = {"site": {"A": 0}}
    # Known draws isolate likelihood extraction from sampler convergence.
    posterior = xr.Dataset(
        {name: (("chain", "draw"), values) for name, values in draws.items()},
        coords={"chain": [4, 9], "draw": [11, 15, 19]},
    )
    model.idata = xr.DataTree.from_dict({"posterior": posterior})
    model.is_fitted = True
    return model


def expected_logp(model: HBR, Y: xr.DataArray, shash: bool) -> np.ndarray:
    posterior = model.idata["posterior"].dataset
    mu = posterior.mu.values[..., None]
    sigma = posterior.sigma.values[..., None]
    if not shash:
        return norm.logpdf(Y.values, loc=mu, scale=sigma)
    distribution = SHASHb.dist(
        mu=mu,
        sigma=sigma,
        epsilon=posterior.epsilon.values[..., None],
        delta=posterior.delta.values[..., None],
    )
    return pm.logp(distribution, Y.values).eval()


@pytest.mark.parametrize("shash", [False, True], ids=["normal", "shash"])
def test_samples_preserve_values_dimensions_and_legacy_mean(shash: bool) -> None:
    model = make_model(shash)
    X, be, Y = make_data()
    before = model.idata.copy(deep=True)
    samples = model.log_likelihood_samples(X, be, Y)
    assert samples.dims == ("chain", "draw", "observations")
    assert samples.chain.values.tolist() == [4, 9]
    assert samples.draw.values.tolist() == [11, 15, 19]
    assert samples.observations.values.tolist() == Y.observations.values.tolist()
    expected = expected_logp(model, Y, shash)
    np.testing.assert_allclose(samples.values, expected)
    legacy = model.elemwise_logp(X, be, Y)
    assert legacy.dims == ("observations",)
    np.testing.assert_allclose(legacy.values, expected.mean(axis=(0, 1)))
    xr.testing.assert_identical(model.idata, before)


@pytest.mark.parametrize("shash", [False, True], ids=["normal", "shash"])
def test_samples_update_observations_without_changing_previous_result(
    shash: bool,
) -> None:
    model = make_model(shash)
    X, be, Y = make_data()
    original = model.log_likelihood_samples(X, be, Y)
    snapshot = original.copy(deep=True)
    indices = [2, 0]
    X, be, Y = (a.isel(observations=indices) for a in (X, be, Y))
    Y = Y + 0.5
    updated = model.log_likelihood_samples(X, be, Y)
    assert updated.shape == (2, 3, 2)
    assert updated.observations.values.tolist() == ["p2", "p3"]
    np.testing.assert_allclose(updated.values, expected_logp(model, Y, shash))
    xr.testing.assert_identical(original, snapshot)


@pytest.mark.parametrize("shash", [False, True], ids=["normal", "shash"])
def test_samples_from_saved_model(tmp_path: Path, shash: bool) -> None:
    model = make_model(shash)
    X, be, Y = make_data()
    expected = model.log_likelihood_samples(X, be, Y)
    metadata = model.to_dict(str(tmp_path))
    loaded = HBR.from_dict(metadata, str(tmp_path))
    assert loaded.pymc_model is None
    xr.testing.assert_allclose(loaded.log_likelihood_samples(X, be, Y), expected)


@pytest.mark.parametrize("method", ["log_likelihood_samples", "elemwise_logp"])
def test_likelihood_requires_fitted_model(method: str) -> None:
    model = HBR("unfitted")
    with pytest.raises(ValueError):
        getattr(model, method)(*make_data())
