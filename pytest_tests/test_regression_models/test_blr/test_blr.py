import pytest

from pcntoolkit.regression_model.blr.blr import BLR


@pytest.mark.parametrize("n_iter,tol,ard",[(100,1e-3,False),(1,1e-6,True)])
def test_blr_to_and_from_dict_and_args(n_iter, tol, ard):
    args = {"n_iter":n_iter,"tol":tol,"ard":ard}
    blr1 = BLR.from_args("test_blr", args)
    assert blr1.reg_conf.n_iter == n_iter
    assert blr1.reg_conf.tol == tol
    assert blr1.reg_conf.ard == ard
    assert blr1.reg_conf.optimizer == "l-bfgs-b"
    assert blr1.reg_conf.l_bfgs_b_l == 0.1
    assert blr1.reg_conf.l_bfgs_b_epsilon == 0.1
    assert blr1.reg_conf.l_bfgs_b_norm == "l2"

    dict2 = blr1.to_dict()
    assert dict2['reg_conf']["n_iter"] == n_iter
    assert dict2['reg_conf']["tol"] == tol
    assert dict2['reg_conf']["ard"] == ard
    assert dict2['reg_conf']["optimizer"] == "l-bfgs-b"
    assert dict2['reg_conf']["l_bfgs_b_l"] == 0.1
    assert dict2['reg_conf']["l_bfgs_b_epsilon"] == 0.1
    assert dict2['reg_conf']["l_bfgs_b_norm"] == "l2"

    blr2 = BLR.from_dict(dict2)
    assert blr2.reg_conf.n_iter == n_iter
    assert blr2.reg_conf.tol == tol
    assert blr2.reg_conf.ard == ard
    assert blr2.reg_conf.optimizer == "l-bfgs-b"
    assert blr2.reg_conf.l_bfgs_b_l == 0.1
    assert blr2.reg_conf.l_bfgs_b_epsilon == 0.1
    assert blr2.reg_conf.l_bfgs_b_norm == "l2"