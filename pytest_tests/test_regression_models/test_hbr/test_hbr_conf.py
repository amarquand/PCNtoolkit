import json
import os

import pytest

from pcntoolkit.regression_model.hbr.hbr_conf import HBRConf


def test_hbrconf_from_args_to_dict_from_dict():
    dict_1 = {"draws": 1000, "tune": 1000, "cores": 1, "likelihood": "Normal"}
    dict_1 = dict_1 | {
        "likelihood": "Normal",
        "linear_mu": True,
        "random_slope_mu": True,
        "centered_slope_mu": True,
        "random_intercept_mu": True,
        "centered_intercept_mu": True,
    }

    conf_1 = HBRConf.from_args(dict_1)
    assert conf_1.draws == 1000
    assert conf_1.tune == 1000
    assert conf_1.cores == 1
    assert conf_1.likelihood == "Normal"
    assert conf_1.mu.linear == True
    assert conf_1.mu.slope.random == True
    assert conf_1.mu.slope.centered == True
    assert conf_1.mu.intercept.random == True
    assert conf_1.mu.intercept.centered == True

    dict_2 = conf_1.to_dict()
    assert dict_2["draws"] == 1000
    assert dict_2["tune"] == 1000
    assert dict_2["cores"] == 1
    assert dict_2["likelihood"] == "Normal"
    assert dict_2["mu"]["linear"] == True
    assert dict_2["mu"]["slope"]["random"] == True
    assert dict_2["mu"]["slope"]["centered"] == True
    assert dict_2["mu"]["intercept"]["random"] == True
    assert dict_2["mu"]["intercept"]["centered"] == True

    conf_2 = HBRConf.from_dict(dict_2)
    assert conf_2.draws == 1000
    assert conf_2.tune == 1000
    assert conf_2.cores == 1
    assert conf_2.likelihood == "Normal"
    assert conf_2.mu.linear == True
    assert conf_2.mu.slope.random == True
    assert conf_2.mu.slope.centered == True
