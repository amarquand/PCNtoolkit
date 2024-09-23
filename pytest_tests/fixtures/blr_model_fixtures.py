import pytest

from pcntoolkit.normative_model.norm_blr import NormBLR
from pcntoolkit.normative_model.norm_conf import NormConf
from pcntoolkit.normative_model.norm_factory import (
    create_normative_model,
    load_normative_model,
)
from pcntoolkit.regression_model.blr.blr import BLR
from pcntoolkit.regression_model.blr.blr_conf import BLRConf
from pcntoolkit.dataio.norm_data import NormData
from pytest_tests.fixtures.norm_data_fixtures import *
from pytest_tests.fixtures.path_fixtures import *

@pytest.fixture
def cvfolds():
    return 4

@pytest.fixture
def savemodel():
    return True


@pytest.fixture
def norm_conf_for_blr_test_model(cvfolds, savemodel, save_dir, log_dir):
    return NormConf(
        perform_cv=False,
        cv_folds=cvfolds,
        savemodel=savemodel,
        save_dir=save_dir + "/blr",
        log_dir=log_dir + "/blr",
        basis_function="none",
        inscaler="standardize",
        outscaler="standardize",
        saveresults=True,
    )


@pytest.fixture
def blrconf():
    return BLRConf(n_iter=100, 
                   tol=1e-3, 
                   ard=False, 
                   optimizer="l-bfgs-b", 
                   l_bfgs_b_l=0.1, 
                   l_bfgs_b_epsilon=0.1, 
                   l_bfgs_b_norm="l2")

@pytest.fixture
def blr(blrconf):
    return BLR("test_blr", blrconf)

@pytest.fixture
def new_norm_blr_model(norm_conf_for_blr_test_model, blrconf):
    return NormBLR(norm_conf_for_blr_test_model, blrconf)

@pytest.fixture
def fitted_norm_blr_model(new_norm_blr_model: NormBLR, norm_data_from_arrays: NormData):
    new_norm_blr_model.fit(norm_data_from_arrays)
    return new_norm_blr_model