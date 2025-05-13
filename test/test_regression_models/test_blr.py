
import numpy as np
import pytest

from pcntoolkit.dataio.norm_data import NormData
from pcntoolkit.regression_model.blr import BLR
from test.fixtures.blr_model_fixtures import *
from test.fixtures.norm_data_fixtures import *
from test.fixtures.path_fixtures import *


@pytest.mark.parametrize("n_iter,tol,ard", [(100, 1e-3, False), (1, 1e-6, True)])
def test_blr_to_and_from_dict_and_args(n_iter, tol, ard):
    args = {"n_iter": n_iter, "tol": tol, "ard": ard}
    blr1 = BLR.from_args("test_blr", args)
    assert blr1.n_iter == n_iter
    assert blr1.tol == tol
    assert blr1.ard == ard
    assert blr1.optimizer == "l-bfgs-b"
    assert blr1.l_bfgs_b_l == 0.1
    assert blr1.l_bfgs_b_epsilon == 0.1
    assert blr1.l_bfgs_b_norm == "l2"

    dict2 = blr1.to_dict()
    assert dict2["n_iter"] == n_iter
    assert dict2["tol"] == tol
    assert dict2["ard"] == ard
    assert dict2["optimizer"] == "l-bfgs-b"
    assert dict2["l_bfgs_b_l"] == 0.1
    assert dict2["l_bfgs_b_epsilon"] == 0.1
    assert dict2["l_bfgs_b_norm"] == "l2"

    blr2 = BLR.from_dict(dict2)
    assert blr2.n_iter == n_iter
    assert blr2.tol == tol
    assert blr2.ard == ard
    assert blr2.optimizer == "l-bfgs-b"
    assert blr2.l_bfgs_b_l == 0.1
    assert blr2.l_bfgs_b_epsilon == 0.1
    assert blr2.l_bfgs_b_norm == "l2"




def test_fit(blr_model: BLR, norm_data_from_arrays: NormData, fitted_norm_blr_model: NormativeModel):
    print("fitting")
    be_maps = fitted_norm_blr_model.batch_effects_maps
    response_var = norm_data_from_arrays.response_vars[0]
    X, be, be_maps, Y, _ = fitted_norm_blr_model.extract_data(norm_data_from_arrays.sel(response_vars=response_var))
    blr_model.fit(X, be, be_maps, Y)
    assert blr_model.is_fitted


def test_forward_backward(fitted_blr_model: BLR, norm_data_from_arrays: NormData, fitted_norm_blr_model: NormativeModel):
    be_maps = fitted_norm_blr_model.batch_effects_maps
    response_var = norm_data_from_arrays.response_vars[0]
    X, be, be_maps, Y, _ = fitted_norm_blr_model.extract_data(norm_data_from_arrays.sel(response_vars=response_var))
    Z = fitted_blr_model.forward(X, be, be_maps, Y)
    assert Z.shape == Y.shape
    Y_prime = fitted_blr_model.backward(X, be, be_maps, Z)
    assert Y_prime.shape == Y.shape
    assert np.allclose(Y_prime, Y)


def test_parse_hyps(norm_data_from_arrays: NormData):
    X = norm_data_from_arrays.X.to_numpy()
    var_X = norm_data_from_arrays.X.to_numpy()
    blr = BLR("test_blr")
    blr.D = X.shape[1]
    blr.var_D = var_X.shape[1]
    hyp = blr.init_hyp()
    alpha, beta, gamma = blr.parse_hyps(hyp, X, var_X)
    assert True