"""Parameter plumbing only; these tests do not establish BIC validity."""

import numpy as np
import pytest

from pcntoolkit.dataio.norm_data import NormData
from pcntoolkit.normative_model import NormativeModel
from pcntoolkit.regression_model.blr import BLR
from pcntoolkit.regression_model.hbr import HBR
from pcntoolkit.util.evaluator import Evaluator


def make_data() -> NormData:
    data = NormData.from_ndarrays(
        name="counts",
        X=np.arange(4.0)[:, None],
        Y=np.array([[1.0, 2.0], [2.0, 4.0], [3.0, 6.0], [4.0, 8.0]]),
        attrs={"covariates": ["age"], "response_vars": ["a", "b"]},
    )
    data["Yhat"] = data.Y - 1
    return data


def make_model() -> NormativeModel:
    model = NormativeModel(BLR("template"), savemodel=False)
    for name, width in [("a", 2), ("b", 5)]:
        reg = BLR(name)
        reg.m = np.zeros(width)
        reg.hyp = np.zeros(2)
        reg.is_fitted = True
        model.regression_models[name] = reg
    return model


def test_counts_flow_from_models_to_bic_by_response() -> None:
    model = make_model()
    data = make_data().sel(response_vars=["b", "a"])
    counts = model.n_parameters(data.response_var_list)
    assert counts == {"b": 7, "a": 4}
    evaluator = Evaluator()
    evaluator.statistics = ["BIC"]
    evaluator.create_statistics_group(data)
    evaluator.evaluate_bic(data, counts)
    for name, count in counts.items():
        assert data.statistics.sel(
            response_vars=name, statistic="BIC"
        ) == pytest.approx(count * np.log(4))


def test_counts_are_current_and_subset_is_lazy() -> None:
    model = make_model()
    model.regression_models["old"] = HBR("old")
    assert model.n_parameters(["a"]) == {"a": 4}
    model.regression_models["a"].m = np.zeros(3)
    assert model.n_parameters(["a"]) == {"a": 5}
    with pytest.raises(ValueError, match="fitted HBR"):
        model.n_parameters()
    with pytest.raises(KeyError, match="missing"):
        model.n_parameters(["missing"])
    assert "missing" not in model.regression_models


def test_restored_hbr_count_is_reused() -> None:
    reg = HBR("a")
    reg.is_fitted = True
    reg._parameter_dimensions = {"coefficients": 4, "sigma": 1}
    model = make_model()
    model.regression_models = {"a": HBR.from_dict(reg.to_dict())}
    assert model.n_parameters() == {"a": 5}


def test_missing_count_does_not_fall_back_to_zero() -> None:
    evaluator = Evaluator()
    data = make_data()
    evaluator.statistics = ["BIC"]
    evaluator.create_statistics_group(data)
    with pytest.raises(ValueError, match="b"):
        evaluator.evaluate_bic(data, {"a": 4})
    assert np.isnan(data.statistics.values).all()


def test_scalar_helper_uses_explicit_count() -> None:
    data = make_data().sel(response_vars="a")
    assert Evaluator()._evaluate_bic(data, 3) == pytest.approx(3 * np.log(4))


def test_ordinary_evaluation_needs_no_counts() -> None:
    data = make_data()
    Evaluator().evaluate(data, statistics=["RMSE", "BIC"])
    assert data.statistic.values.tolist() == ["RMSE"]
    np.testing.assert_allclose(data.statistics.values, 1)


def test_default_normative_evaluation_does_not_request_hbr_counts() -> None:
    model = make_model()
    old_hbr = HBR("a")
    old_hbr.is_fitted = True
    model.regression_models = {"a": old_hbr}
    with pytest.raises(ValueError, match="unavailable"):
        model.n_parameters()
    data = make_data().sel(response_vars=["a"])
    data["Z"] = data.Y / 2
    data["logp"] = data.Y * 0 - 1
    data["baseline_logp"] = data.Y * 0 - 2
    data["centiles"] = data.Y.expand_dims(centile=[0.25, 0.5, 0.75])
    model.evaluate(data)
    assert "BIC" not in data.statistic.values
    assert data.statistics.sel(statistic="RMSE").item() == pytest.approx(1)
