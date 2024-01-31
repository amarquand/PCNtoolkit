import pytest

from pcntoolkit.normative_model.norm_conf import NormConf
from pcntoolkit.normative_model.norm_hbr import NormHBR
from pcntoolkit.regression_model.hbr.hbr import HBR
from pcntoolkit.regression_model.hbr.hbr_conf import HBRConf
from pcntoolkit.regression_model.hbr.param import Param
from pytest_tests.fixtures.data import *
from pytest_tests.fixtures.paths import *


@pytest.fixture(scope="session")
def cvfolds():
    return 4


@pytest.fixture(scope="session")
def alg():
    return "hbr"


@pytest.fixture(scope="session")
def savemodel():
    return True


@pytest.fixture(scope="session")
def hbr_norm_conf_dict(
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
    resource_dir,
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
        "save_dir": resource_dir + "/hbr/save_load_test",
        "log_dir": resource_dir + "/hbr/log_test",
    }


@pytest.fixture(scope="session")
def norm_args():
    return {
        "log_dir": "nonexistant",
        "save_dir": "nonexistant",
        "inscaler": "standardize",
        "outscaler": "standardize",
    }


@pytest.fixture(scope="session")
def normconf_for_hbr(cvfolds, savemodel, resource_dir):
    return NormConf(
        perform_cv=False,
        cv_folds=cvfolds,
        savemodel=savemodel,
        save_dir=resource_dir + "/hbr/save_load_test",
        log_dir=resource_dir + "/hbr/log_test",
        basis_function="bspline",
        order=3,
        nknots=10,
        inscaler="standardize",
        outscaler="standardize",
        saveresults=True,
    )


@pytest.fixture(scope="session")
def mu():
    return Param(
        name="mu",
        linear=True,
        intercept=Param(name="intercept_mu", random=True, centered=False),
        slope=Param(name="slope_mu", random=False),
    )


@pytest.fixture(scope="session")
def sigma():
    return Param(
        name="sigma",
        linear=False,
        mu=Param(name="mu_sigma", random=True, centered=True),
        sigma=Param(name="sigma_sigma", random=False),
    )


@pytest.fixture(scope="session")
def hbrconf(mu, sigma):
    return HBRConf(draws=1000, tune=1000, chains=2, cores=1, mu=mu, sigma=sigma)


@pytest.fixture(scope="session")
def hbr(hbrconf):
    return HBR(hbrconf)


@pytest.fixture(scope="session")
def new_norm_hbr_model(normconf_for_hbr, hbrconf):
    return NormHBR(normconf_for_hbr, hbrconf)


@pytest.fixture(scope="session")
def fitted_norm_hbr_model(new_norm_hbr_model: NormHBR, train_norm_data: NormData):
    new_norm_hbr_model.fit(train_norm_data)
    return new_norm_hbr_model
