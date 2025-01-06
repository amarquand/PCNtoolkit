

from pcntoolkit.regression_model.hbr.hbr_conf import HBRConf
from pcntoolkit.regression_model.hbr.prior import LinearPrior, RandomPrior

"""
This file contains tests for the HBRConf class in the PCNtoolkit.

The tests cover the following aspects:
1. Creating HBRConf objects from arguments
2. Converting HBRConf objects to dictionaries
3. Creating HBRConf objects from dictionaries
"""


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
    assert conf_1.pymc_cores == 1
    assert conf_1.likelihood == "Normal"
    assert isinstance(conf_1.mu, LinearPrior)
    assert isinstance(conf_1.mu.slope, RandomPrior)
    assert isinstance(conf_1.mu.intercept, RandomPrior)

    dict_2 = conf_1.to_dict()
    assert dict_2["draws"] == 1000
    assert dict_2["tune"] == 1000
    assert dict_2["pymc_cores"] == 1
    assert dict_2["likelihood"] == "Normal"
    assert dict_2["mu"]['type'] == "LinearPrior"
    assert dict_2["mu"]["slope"]['type'] == "RandomPrior"
    assert dict_2["mu"]["intercept"]["type"] == "RandomPrior"

    conf_2 = HBRConf.from_dict(dict_2)
    assert conf_2.draws == 1000
    assert conf_2.tune == 1000
    assert conf_2.pymc_cores == 1
    assert conf_2.likelihood == "Normal"
    assert isinstance(conf_2.mu, LinearPrior)
    assert isinstance(conf_2.mu.slope, RandomPrior)
    assert isinstance(conf_2.mu.intercept, RandomPrior)