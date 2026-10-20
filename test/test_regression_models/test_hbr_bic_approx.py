"""Tests for the explicitly provisional sampled-likelihood BIC score."""

import numpy as np
import pytest
import xarray as xr

from pcntoolkit.regression_model.hbr import HBR
from pcntoolkit.util.evaluator import Evaluator
from test.test_regression_models.test_hbr_log_likelihood import (
    expected_logp,
    make_data,
    make_model,
)


@pytest.mark.parametrize("shash", [False, True], ids=["normal", "shash"])
def test_bic_approx_uses_one_joint_draw_and_structural_count(shash: bool) -> None:
    model = make_model(shash)
    X, be, Y = make_data()
    before = model.idata.copy(deep=True)
    likelihoods = expected_logp(model, Y, shash)
    best = likelihoods.sum(axis=2).max()
    # Selecting the best draw for each observation is a different, invalid score.
    assert not np.isclose(best, likelihoods.max(axis=(0, 1)).sum())
    with pytest.warns(UserWarning, match="BIC_approx.*not.*validated BIC"):
        result = model.bic_approx(X, be, Y)
    expected = -2 * best + model.n_parameters() * np.log(Y.size)
    assert result == pytest.approx(expected)
    xr.testing.assert_identical(model.idata, before)


@pytest.mark.parametrize("bad_value", [np.nan, np.inf, -np.inf])
def test_bic_approx_rejects_nonfinite_likelihoods(
    monkeypatch: pytest.MonkeyPatch, bad_value: float
) -> None:
    model = make_model()
    samples = xr.DataArray(
        [[[bad_value, -1.0], [-2.0, -3.0]]],
        dims=("chain", "draw", "observations"),
    )
    # Inject invalid log likelihoods to test the aggregation boundary directly.
    monkeypatch.setattr(model, "log_likelihood_samples", lambda *args: samples)
    with pytest.raises(ValueError, match="finite"):
        model.bic_approx(*make_data())


@pytest.mark.parametrize("shape", [(0, 2, 3), (1, 0, 3), (1, 2, 0)])
def test_bic_approx_rejects_empty_likelihoods(
    monkeypatch: pytest.MonkeyPatch, shape: tuple[int, int, int]
) -> None:
    model = make_model()
    samples = xr.DataArray(np.zeros(shape), dims=("chain", "draw", "observations"))
    monkeypatch.setattr(model, "log_likelihood_samples", lambda *args: samples)
    with pytest.raises(ValueError, match="non-empty"):
        model.bic_approx(*make_data())


def test_bic_approx_requires_fitted_model() -> None:
    with pytest.raises(ValueError):
        HBR("unfitted").bic_approx(*make_data())


@pytest.mark.parametrize("count", [0, 4, np.int64(4)])
def test_bic_approx_formula(monkeypatch: pytest.MonkeyPatch, count: int) -> None:
    model = make_model()
    samples = xr.DataArray(
        np.full((1, 2, 10), -1.25), dims=("chain", "draw", "observations")
    )
    monkeypatch.setattr(model, "log_likelihood_samples", lambda *args: samples)
    monkeypatch.setattr(model, "n_parameters", lambda: count)
    with pytest.warns(UserWarning, match="BIC_approx"):
        result = model.bic_approx(*make_data())
    assert result == pytest.approx(25 + count * np.log(10))


@pytest.mark.parametrize("count", [-1, 1.5, True, np.nan, np.inf])
def test_bic_approx_rejects_invalid_parameter_counts(
    monkeypatch: pytest.MonkeyPatch, count: int | float
) -> None:
    model = make_model()
    samples = xr.DataArray([[[-1.0]]], dims=("chain", "draw", "observations"))
    monkeypatch.setattr(model, "log_likelihood_samples", lambda *args: samples)
    monkeypatch.setattr(model, "n_parameters", lambda: count)
    with pytest.raises(ValueError, match="n_params"):
        model.bic_approx(*make_data())


def test_bic_approx_is_not_enabled_in_ordinary_evaluation() -> None:
    from test.test_util.test_bic_parameter_counts import make_data as make_norm_data

    data = make_norm_data()
    Evaluator().evaluate(data, statistics=["RMSE", "BIC", "BIC_approx"])
    assert data.statistic.values.tolist() == ["RMSE"]
