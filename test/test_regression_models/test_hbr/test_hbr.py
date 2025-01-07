import pytest

from pcntoolkit.regression_model.hbr.hbr import HBR
from pcntoolkit.regression_model.hbr.prior import LinearPrior, Prior, RandomPrior
from test.fixtures.data_fixtures import *
from test.fixtures.hbr_model_fixtures import *
from test.fixtures.path_fixtures import *

"""
This file contains tests for the HBR class in the PCNtoolkit.
The tests cover:
1. Creating HBR objects from arguments
2. Converting HBR objects to dictionaries
3. Creating HBR objects from dictionaries
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
def test_hbr_to_and_from_dict_and_args(sample_args, args):
    # Testing from args first
    hbr = HBR.from_args("test_name", sample_args | args)
    assert hbr.reg_conf.draws == sample_args.get("draws")
    assert hbr.reg_conf.tune == sample_args.get("tune")
    assert hbr.reg_conf.pymc_cores == sample_args.get("pymc_cores")
    assert hbr.reg_conf.likelihood == "Normal"
    if args.get("linear_mu", False):
        if args.get("random_slope_mu", False):
            assert isinstance(hbr.reg_conf.mu.slope, RandomPrior)
        else:
            assert isinstance(hbr.reg_conf.mu.slope, Prior)
        if args.get("random_intercept_mu", False):
            assert isinstance(hbr.reg_conf.mu.intercept, RandomPrior)
        else:
            assert isinstance(hbr.reg_conf.mu.intercept, Prior)
    assert hbr.is_from_dict
    assert not isinstance(hbr.reg_conf.sigma, LinearPrior) and not isinstance(hbr.reg_conf.sigma, RandomPrior)

    # Testing to_dict
    hbr_dict = hbr.to_dict()
    assert hbr_dict["reg_conf"]["draws"] == sample_args.get("draws")
    assert hbr_dict["reg_conf"]["tune"] == sample_args.get("tune")
    assert hbr_dict["reg_conf"]["pymc_cores"] == sample_args.get("pymc_cores")
    assert hbr_dict["reg_conf"]["likelihood"] == "Normal"
    assert args.get("linear_mu", False) == (hbr_dict["reg_conf"]["mu"]["type"] == "LinearPrior")
    if args.get("linear_mu", False):
        assert hbr_dict["reg_conf"]["mu"]["type"] == "LinearPrior"
        assert (hbr_dict["reg_conf"]["mu"]["slope"]["type"] == "RandomPrior") == args.get("random_slope_mu", False)
        assert (hbr_dict["reg_conf"]["mu"]["intercept"]["type"] == "RandomPrior") == args.get("random_intercept_mu", False)
    assert hbr.is_from_dict
    assert not hbr_dict["reg_conf"]["sigma"]["type"] == "LinearPrior"

    # Testing from_dict
    del hbr
    hbr = HBR.from_dict(hbr_dict)
    assert hbr.reg_conf.draws == sample_args.get("draws")
    assert hbr.reg_conf.tune == sample_args.get("tune")
    assert hbr.reg_conf.pymc_cores == sample_args.get("pymc_cores")
    assert hbr.reg_conf.likelihood == "Normal"
    if args.get("linear_mu", False):
        assert isinstance(hbr.reg_conf.mu, LinearPrior)
        if args.get("random_slope_mu", False):
            assert isinstance(hbr.reg_conf.mu.slope, RandomPrior)
        else:
            assert isinstance(hbr.reg_conf.mu.slope, Prior)
        if args.get("random_intercept_mu", False):
            assert isinstance(hbr.reg_conf.mu.intercept, RandomPrior)
        else:
            assert isinstance(hbr.reg_conf.mu.intercept, Prior)

    assert hbr.is_from_dict
