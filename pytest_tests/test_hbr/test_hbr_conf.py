import json
import os

import pytest

from pcntoolkit.regression_model.hbr.hbr_conf import HBRConf


def test_from_dict():
    my_dict = {"draws": 1000, "tune": 1000, "cores": 1, "likelihood": "Normal"}
    my_dict = my_dict | {
        "likelihood": "Normal",
        "linear_mu": True,
        "random_slope_mu": True,
        "centered_slope_mu": True,
        "random_intercept_mu": True,
        "centered_intercept_mu": True,
    }
    conf = HBRConf.from_args(my_dict)
    assert conf.draws == 1000
    assert conf.tune == 1000
    assert conf.cores == 1
    assert conf.likelihood == "Normal"
    assert conf.mu.linear == True
    assert conf.mu.slope.random == True
    assert conf.mu.slope.centered == True


def test_to_dict():
    """
    Tests the to_dict method.
    """
    args = {"draws": 1000, "tune": 1000, "cores": 1, "likelihood": "Normal"}
    args = args | {
        "likelihood": "Normal",
        "linear_mu": True,
        "random_slope_mu": True,
        "centered_slope_mu": True,
        "random_intercept_mu": True,
        "centered_intercept_mu": True,
    }
    conf = HBRConf.from_args(args)

    conf_dict = conf.to_dict()

    assert conf_dict["draws"] == 1000
    assert conf_dict["tune"] == 1000
    assert conf_dict["cores"] == 1
    assert conf_dict["likelihood"] == "Normal"
    assert conf_dict["mu"]["linear"] == True
    assert conf_dict["mu"]["slope"]["random"] == True
    assert conf_dict["mu"]["slope"]["centered"] == True
    assert conf_dict["mu"]["intercept"]["random"] == True
    assert conf_dict["mu"]["intercept"]["centered"] == True
