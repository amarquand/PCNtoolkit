import pytest

from pcntoolkit.regression_model.hbr.hbr import HBR
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
def test_hbr_to_and_from_dict_and_args(sample_args, args):
    # Testing from args first
    hbr = HBR.from_args("test_name", sample_args | args)
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
    assert hbr.is_from_dict
    assert not hbr.reg_conf.sigma.linear

    # Testing to_dict
    hbr_dict = hbr.to_dict()
    assert hbr_dict["reg_conf"]["draws"] == 10
    assert hbr_dict["reg_conf"]["tune"] == 10
    assert hbr_dict["reg_conf"]["cores"] == 1
    assert hbr_dict["reg_conf"]["likelihood"] == "Normal"
    assert hbr_dict["reg_conf"]["mu"]["linear"] == args.get("linear_mu", False)
    if args.get("linear_mu", False):
        assert hbr_dict["reg_conf"]["mu"]["slope"]["random"] == args.get(
            "random_slope_mu", False
        )
        assert hbr_dict["reg_conf"]["mu"]["intercept"]["random"] == args.get(
            "random_intercept_mu", False
        )
        assert hbr_dict["reg_conf"]["mu"]["slope"]["centered"] == args.get(
            "centered_slope_mu", False
        )
        assert hbr_dict["reg_conf"]["mu"]["intercept"]["centered"] == args.get(
            "centered_intercept_mu", False
        )
    assert hbr.is_from_dict
    assert not hbr_dict["reg_conf"]["sigma"]["linear"]

    # Testing from_dict
    del hbr
    hbr = HBR.from_dict(hbr_dict)
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
    assert hbr.is_from_dict
