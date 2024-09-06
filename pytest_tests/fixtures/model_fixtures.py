import pytest

from pcntoolkit.normative_model.norm_conf import NormConf
from pcntoolkit.normative_model.norm_hbr import NormHBR
from pcntoolkit.regression_model.hbr.hbr import HBR
from pcntoolkit.regression_model.hbr.hbr_conf import HBRConf
from pcntoolkit.regression_model.hbr.param import Param
from pytest_tests.fixtures.data_fixtures import *
from pytest_tests.fixtures.path_fixtures import *


"""
This file contains pytest fixtures used for model generation and testing in the PCNtoolkit.

The fixtures defined here include:
1. Configuration parameters for normative modeling (e.g., cvfolds, alg, savemodel)
2. MCMC sampling parameters (n_mcmc_samples, sample_args)
3. Normative model configuration (norm_args, norm_conf_dict_for_generic_model, norm_conf_for_generic_model)
4. Specific configuration for HBR (Hierarchical Bayesian Regression) models
5. File paths for various resources (imported from pytest_tests.fixtures.paths)
6. Data-related fixtures (imported from pytest_tests.fixtures.data)

These fixtures are used to set up consistent testing environments and configurations
across different test files in the PCNtoolkit testing suite. They provide reusable
components for creating and configuring normative models, particularly focusing on
the Hierarchical Bayesian Regression (HBR) approach.
"""


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
def n_mcmc_samples():
    return 10


@pytest.fixture
def sample_args():
    return {"draws": 10, "tune": 10, "cores": 1}


@pytest.fixture
def norm_args(log_dir, save_dir):
    return {"log_dir": log_dir, "save_dir": save_dir}


@pytest.fixture
def norm_conf_dict_for_generic_model(log_dir, save_dir):
    return {"log_dir": log_dir, "save_dir": save_dir}


@pytest.fixture
def norm_conf_for_generic_model(log_dir, save_dir):
    return NormConf(log_dir=log_dir, save_dir=save_dir)


@pytest.fixture
def norm_conf_dict_for_hbr_test_model(
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
        "save_dir": save_dir + "/hbr",
        "log_dir": log_dir + "/hbr",
    }


@pytest.fixture
def norm_conf_for_hbr_test_model(cvfolds, savemodel, save_dir, log_dir):
    return NormConf(
        perform_cv=False,
        cv_folds=cvfolds,
        savemodel=savemodel,
        save_dir=save_dir + "/hbr",
        log_dir=log_dir + "/hbr",
        basis_function="bspline",
        order=3,
        nknots=10,
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
        slope=Param(name="slope_mu", dims=("covariates,"), random=False),
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
def hbrconf(mu, sigma, n_mcmc_samples):
    return HBRConf(draws=n_mcmc_samples, tune=10, chains=2, cores=2, mu=mu, sigma=sigma)


@pytest.fixture
def hbr(hbrconf):
    return HBR(hbrconf)


@pytest.fixture
def new_norm_hbr_model(norm_conf_for_hbr_test_model, hbrconf):
    return NormHBR(norm_conf_for_hbr_test_model, hbrconf)


@pytest.fixture
def fitted_norm_hbr_model(new_norm_hbr_model: NormHBR, train_norm_data: pd.DataFrame):
    new_norm_hbr_model.fit(train_norm_data)
    return new_norm_hbr_model
