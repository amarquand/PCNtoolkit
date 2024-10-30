import pytest

from pcntoolkit.dataio.norm_data import NormData
from pcntoolkit.regression_model.blr.blr import BLR
from pcntoolkit.regression_model.blr.blr_conf import BLRConf
from pytest_tests.fixtures.path_fixtures import *
from pytest_tests.fixtures.norm_data_fixtures import *
from pytest_tests.fixtures.blr_model_fixtures import *


@pytest.mark.parametrize("n_iter,tol,ard", [(100, 1e-3, False), (1, 1e-6, True)])
def test_blr_to_and_from_dict_and_args(n_iter, tol, ard):
    args = {"n_iter": n_iter, "tol": tol, "ard": ard}
    blr1 = BLR.from_args("test_blr", args)
    assert blr1.reg_conf.n_iter == n_iter
    assert blr1.reg_conf.tol == tol
    assert blr1.reg_conf.ard == ard
    assert blr1.reg_conf.optimizer == "l-bfgs-b"
    assert blr1.reg_conf.l_bfgs_b_l == 0.1
    assert blr1.reg_conf.l_bfgs_b_epsilon == 0.1
    assert blr1.reg_conf.l_bfgs_b_norm == "l2"

    dict2 = blr1.to_dict()
    assert dict2["reg_conf"]["n_iter"] == n_iter
    assert dict2["reg_conf"]["tol"] == tol
    assert dict2["reg_conf"]["ard"] == ard
    assert dict2["reg_conf"]["optimizer"] == "l-bfgs-b"
    assert dict2["reg_conf"]["l_bfgs_b_l"] == 0.1
    assert dict2["reg_conf"]["l_bfgs_b_epsilon"] == 0.1
    assert dict2["reg_conf"]["l_bfgs_b_norm"] == "l2"

    blr2 = BLR.from_dict(dict2)
    assert blr2.reg_conf.n_iter == n_iter
    assert blr2.reg_conf.tol == tol
    assert blr2.reg_conf.ard == ard
    assert blr2.reg_conf.optimizer == "l-bfgs-b"
    assert blr2.reg_conf.l_bfgs_b_l == 0.1
    assert blr2.reg_conf.l_bfgs_b_epsilon == 0.1
    assert blr2.reg_conf.l_bfgs_b_norm == "l2"


def test_parse_hyps(blr, norm_data_from_arrays):
    X = norm_data_from_arrays.X
    hyp = np.zeros(X.shape[1] + 1)
    alpha, beta = blr.parse_hyps(hyp, norm_data_from_arrays.X)
    assert np.all(alpha == 1)
    assert np.all(beta == 1)


def test_post(blr, norm_data_from_arrays: NormData):
    X = norm_data_from_arrays.X
    for response_var in norm_data_from_arrays.response_vars:
        y = norm_data_from_arrays.sel(response_vars=response_var).y
        hyp = np.zeros(X.shape[1] + 1)
        blr.post(hyp, X.to_numpy(), y.to_numpy())
        assert (blr.hyp == hyp).all()


def test_loglik(blr, norm_data_from_arrays: NormData):
    X = norm_data_from_arrays.X
    for response_var in norm_data_from_arrays.response_vars:
        y = norm_data_from_arrays.sel(response_vars=response_var).y
        hyp = np.zeros(X.shape[1] + 1)
        blr.loglik(hyp, X.to_numpy(), y.to_numpy())
        assert (blr.hyp == hyp).all()


def test_penalized_loglik(blr, norm_data_from_arrays: NormData, l=0.1, norm="L1"):
    X = norm_data_from_arrays.X
    for response_var in norm_data_from_arrays.response_vars:
        y = norm_data_from_arrays.sel(response_vars=response_var).y
        hyp = np.zeros(X.shape[1] + 1)
        blr.penalized_loglik(hyp, X.to_numpy(), y.to_numpy(), l=l, norm=norm)
        assert (blr.hyp == hyp).all()


def test_dloglik(blr, norm_data_from_arrays: NormData):
    X = norm_data_from_arrays.X
    for response_var in norm_data_from_arrays.response_vars:
        y = norm_data_from_arrays.sel(response_vars=response_var).y
        hyp = np.zeros(X.shape[1] + 1)
        dloglik = blr.dloglik(hyp, X.to_numpy(), y.to_numpy())
        assert dloglik.shape == hyp.shape


def test_fit(blr, norm_data_from_arrays: NormData):
    X = norm_data_from_arrays.X
    for response_var in norm_data_from_arrays.response_vars:
        y = norm_data_from_arrays.sel(response_vars=response_var).y
        blr.fit(X.to_numpy(), y.to_numpy())
        assert blr.is_fitted
        break
