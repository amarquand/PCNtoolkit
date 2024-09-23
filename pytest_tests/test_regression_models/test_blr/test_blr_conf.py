import pytest
from pcntoolkit.regression_model.blr.blr_conf import BLRConf


def test_detect_configuration_problems():
    # Test with valid configuration
    conf = BLRConf(n_iter=100, tol=1e-3, optimizer="l-bfgs-b")
    assert conf.detect_configuration_problems() == []

    # Test with invalid configurations
    with pytest.raises(ValueError):
        conf = BLRConf(n_iter=0, tol=1e-3, optimizer="l-bfgs-b")
    
    with pytest.raises(ValueError):
        conf = BLRConf(n_iter=100, tol=0, optimizer="l-bfgs-b")

    with pytest.raises(ValueError):
        conf = BLRConf(n_iter=100, tol=1e-3, optimizer="invalid_optimizer")



@pytest.mark.parametrize("n_iter,tol,ard",[(100,1e-3,False),(1,1e-6,True)])
def test_from_args_to_dict_from_dict(n_iter,tol,ard):
    args = {"n_iter":n_iter,"tol":tol,"ard":ard}
    conf1 = BLRConf.from_args(args)
    assert conf1.n_iter == n_iter
    assert conf1.tol == tol
    assert conf1.ard == ard
    assert conf1.optimizer == "l-bfgs-b"
    assert conf1.l_bfgs_b_l == 0.1
    assert conf1.l_bfgs_b_epsilon == 0.1
    assert conf1.l_bfgs_b_norm == "l2"

    dict2 = conf1.to_dict()
    assert dict2["n_iter"] == n_iter
    assert dict2["tol"] == tol
    assert dict2["ard"] == ard
    assert dict2["optimizer"] == "l-bfgs-b"
    assert dict2["l_bfgs_b_l"] == 0.1
    assert dict2["l_bfgs_b_epsilon"] == 0.1
    assert dict2["l_bfgs_b_norm"] == "l2"

    conf2 = BLRConf.from_dict(dict2)
    assert conf2.n_iter == n_iter
    assert conf2.tol == tol
    assert conf2.ard == ard
    assert conf2.optimizer == "l-bfgs-b"
    assert conf2.l_bfgs_b_l == 0.1
    assert conf2.l_bfgs_b_epsilon == 0.1
    assert conf2.l_bfgs_b_norm == "l2"

            