import numpy as np
import pytest

from pcntoolkit.dataio.norm_data import NormData
from pcntoolkit.normative_model.norm_blr import NormBLR
from pcntoolkit.regression_model.blr.blr import BLR
from pcntoolkit.regression_model.blr.blr_conf import BLRConf
from test.fixtures.blr_model_fixtures import *
from test.fixtures.norm_data_fixtures import *
from test.fixtures.path_fixtures import *


@pytest.mark.parametrize("n_iter,tol,ard", [(100, 1e-3, False), (1, 1e-6, True)])
def test_blr_to_and_from_dict_and_args(n_iter, tol, ard):
    args = {"n_iter": n_iter, "tol": tol, "ard": ard}
    blr1 = BLR.from_args("test_blr", args)
    rconf: BLRConf = blr1.reg_conf  # type: ignore
    assert rconf.n_iter == n_iter
    assert rconf.tol == tol
    assert rconf.ard == ard
    assert rconf.optimizer == "l-bfgs-b"
    assert rconf.l_bfgs_b_l == 0.1
    assert rconf.l_bfgs_b_epsilon == 0.1
    assert rconf.l_bfgs_b_norm == "l2"

    dict2 = blr1.to_dict()
    assert dict2["reg_conf"]["n_iter"] == n_iter
    assert dict2["reg_conf"]["tol"] == tol
    assert dict2["reg_conf"]["ard"] == ard
    assert dict2["reg_conf"]["optimizer"] == "l-bfgs-b"
    assert dict2["reg_conf"]["l_bfgs_b_l"] == 0.1
    assert dict2["reg_conf"]["l_bfgs_b_epsilon"] == 0.1
    assert dict2["reg_conf"]["l_bfgs_b_norm"] == "l2"

    blr2 = BLR.from_dict(dict2)
    rconf2: BLRConf = blr2.reg_conf  # type: ignore
    assert rconf2.n_iter == n_iter
    assert rconf2.tol == tol
    assert rconf2.ard == ard
    assert rconf2.optimizer == "l-bfgs-b"
    assert rconf2.l_bfgs_b_l == 0.1
    assert rconf2.l_bfgs_b_epsilon == 0.1
    assert rconf2.l_bfgs_b_norm == "l2"


def test_parse_hyps(blr: BLR, norm_data_from_arrays, fitted_norm_blr_model: NormBLR):
    X = norm_data_from_arrays.X.to_numpy()
    var_X = norm_data_from_arrays.X.to_numpy()
    blrdata = fitted_norm_blr_model.normdata_to_blrdata(norm_data_from_arrays.sel(response_vars="response_var_1"))
    blr.D = X.shape[1]
    hyp = blr.init_hyp(blrdata)
    alpha, beta, gamma = blr.parse_hyps(hyp, X, var_X)
    assert np.all(alpha == 1)
    assert np.all(beta == 1)


def test_post(blr, norm_data_from_arrays: NormData, fitted_norm_blr_model: NormBLR):
    X = norm_data_from_arrays.X.to_numpy()
    var_X = norm_data_from_arrays.X.to_numpy()
    blr.var_D = var_X.shape[1]
    blr.D = X.shape[1]
    for response_var in norm_data_from_arrays.response_vars:
        blrdata = fitted_norm_blr_model.normdata_to_blrdata(norm_data_from_arrays.sel(response_vars=response_var))
        y = norm_data_from_arrays.sel(response_vars=response_var).y.to_numpy()
        hyp = blr.init_hyp(blrdata)
        blr.post(hyp, X, y, var_X)
        assert (blr.hyp == hyp).all()


def test_loglik(blr, norm_data_from_arrays: NormData, fitted_norm_blr_model: NormBLR):
    X = norm_data_from_arrays.X.to_numpy()
    var_X = norm_data_from_arrays.X.to_numpy()
    blr.D = X.shape[1]
    blr.var_D = var_X.shape[1]
    for response_var in norm_data_from_arrays.response_vars:
        y = norm_data_from_arrays.sel(response_vars=response_var).y.to_numpy()
        blrdata = fitted_norm_blr_model.normdata_to_blrdata(norm_data_from_arrays.sel(response_vars=response_var))
        hyp = blr.init_hyp(blrdata)
        blr.loglik(hyp, X, y, var_X)
        assert (blr.hyp == hyp).all()


def test_penalized_loglik(
    blr, norm_data_from_arrays: NormData, fitted_norm_blr_model: NormBLR, regularizer_strength=0.1, norm="L1"
):
    X = norm_data_from_arrays.X.to_numpy()
    var_X = norm_data_from_arrays.X.to_numpy()
    blr.D = X.shape[1]
    blr.var_D = var_X.shape[1]
    for response_var in norm_data_from_arrays.response_vars:
        y = norm_data_from_arrays.sel(response_vars=response_var).y.to_numpy()
        blrdata = fitted_norm_blr_model.normdata_to_blrdata(norm_data_from_arrays.sel(response_vars=response_var))
        hyp = blr.init_hyp(blrdata)
        blr.penalized_loglik(hyp, X, y, var_X, regularizer_strength=regularizer_strength, norm=norm)
        assert (blr.hyp == hyp).all()


# def test_dloglik(blr, norm_data_from_arrays: NormData):
#     X = norm_data_from_arrays.X.to_numpy()
#     var_X = norm_data_from_arrays.X.to_numpy()
#     print(var_X.shae)
#     for response_var in norm_data_from_arrays.response_vars:
#         y = norm_data_from_arrays.sel(response_vars=response_var).y.to_numpy()
#         hyp = np.zeros(X.shape[1] + 1)
#         blr.dloglik(hyp, X, y, var_X)
#         assert (blr.hyp == hyp).all()
