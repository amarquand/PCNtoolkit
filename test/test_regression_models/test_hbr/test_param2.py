from pcntoolkit.regression_model.hbr.param2 import (
    FixedParam,
    LinearParam,
    Param,
    RandomParam,
)


def test_fixed_to_and_from_dict():
    fp = FixedParam("mu", ("covariates",), "identity", (0, 1), "Normal", (0, 10))
    dct = fp.to_dict()
    assert dct == {
        "type": "FixedParam",
        "name": "mu",
        "dims": ("covariates",),
        "dist_name": "Normal",
        "dist_params": (0, 10),
        "mapping": "identity",
        "mapping_params": (0, 1),
    }
    reconstructed = Param.from_dict(dct)
    assert fp == reconstructed


def test_random_centered_param_to_and_from_dict():
    rp = RandomParam(name="slope_mu")
    dct = rp.to_dict()
    assert dct['dims'] == ('covariates',)
    assert dct['mu']['name'] == "mu_slope_mu"
    assert Param.from_dict(dct) == rp

def test_linear_param_to_and_from_dict():
    lp = LinearParam(name="mu")
    dct = lp.to_dict()
    assert dct['name'] == "mu"
    assert dct['mapping'] == "identity"
    assert dct['mapping_params'] == (0, 1)
    assert dct['slope']['dims'] == ('covariates',)
    assert dct['slope']['name'] == "slope_mu"
    assert dct['slope']['dist_name'] == "Normal"
    assert dct['slope']['dist_params'] == (0, 10)
    assert Param.from_dict(dct) == lp
