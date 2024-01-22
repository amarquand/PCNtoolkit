import pytest

from pcntoolkit.normative_model.norm_conf import NormConf
from pcntoolkit.normative_model.norm_hbr import NormHBR
from pcntoolkit.regression_model.hbr.hbr import HBR
from pcntoolkit.regression_model.hbr.hbr_conf import HBRConf
from pcntoolkit.regression_model.hbr.param import Param
from pytest_tests.fixtures.data import *
from pytest_tests.fixtures.paths import *


@pytest.fixture
def cvfolds():
    return 4


@pytest.fixture
def alg():
    return "hbr"


@pytest.fixture
def savemodel():
    return True


@pytest.fixture
def conf_dict(
    cvfolds,
    alg,
    responsefile,
    maskfile,
    covfile,
    testcov,
    testresp,
    savemodel,
    trbefile,
    tsbefile,
    save_dir,
    log_dir,
):
    return {
        "responses": responsefile,
        "maskfile": maskfile,
        "covfile": covfile,
        "cvfolds": cvfolds,
        "testcov": testcov,
        "testresp": testresp,
        "alg": alg,
        "savemodel": savemodel,
        "trbefile": trbefile,
        "tsbefile": tsbefile,
        "save_dir": save_dir,
        "log_dir": log_dir,
    }


@pytest.fixture
def norm_args(log_dir, save_dir):
    return {"log_dir": log_dir, "save_dir": save_dir}


@pytest.fixture
def normconf(cvfolds, savemodel, save_dir, log_dir):
    return NormConf(
        perform_cv=False,
        cv_folds=cvfolds,
        savemodel=savemodel,
        save_dir=save_dir,
        log_dir=log_dir,
        basis_function="bspline",
        order=3,
        n_knots=10,
        inscaler="standardize",
        outscaler="standardize",
        saveresults=True,
    )


@pytest.fixture
def mu():
    return Param(
        name="mu",
        linear=True,
        intercept=Param(name="intercept_mu", random=True, centered=False),
        slope=Param(name="slope_mu", random=False),
    )


@pytest.fixture
def sigma():
    return Param(
        name="sigma",
        linear=False,
        mu=Param(name="mu_sigma", random=True, centered=True),
        sigma=Param(name="sigma_sigma", random=False),
    )


@pytest.fixture
def hbrconf(mu, sigma):
    return HBRConf(draws=1000, tune=1000, chains=2, cores=1, mu=mu, sigma=sigma)


@pytest.fixture
def hbr(hbrconf):
    return HBR(hbrconf)


@pytest.fixture
def new_norm_hbr_from_objs(normconf, hbrconf):
    return NormHBR(normconf, hbrconf)


@pytest.fixture
def fitted_norm_hbr_from_file():
    savepath = "pytest_tests/resources/hbr_save_load_test"
    return NormHBR.load(savepath)
