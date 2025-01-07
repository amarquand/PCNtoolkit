import pytest

from pcntoolkit.dataio.norm_data import NormData
from pcntoolkit.normative_model.norm_conf import NormConf
from pcntoolkit.normative_model.norm_hbr import NormHBR
from pcntoolkit.regression_model.hbr.hbr import HBR
from pcntoolkit.regression_model.hbr.hbr_conf import HBRConf
from pcntoolkit.regression_model.hbr.prior import make_prior
from test.fixtures.data_fixtures import *
from test.fixtures.path_fixtures import *

"""
This file contains pytest fixtures used for model generation and testing in the PCNtoolkit.

The fixtures defined here include:
1. Configuration parameters for normative modeling (e.g., cvfolds, alg, savemodel)
2. MCMC sampling parameters (n_mcmc_samples, sample_args)
3. Normative model configuration (norm_args, norm_conf_dict_for_generic_model, norm_conf_for_generic_model)
4. Specific configuration for HBR (Hierarchical Bayesian Regression) models
5. File paths for various resources (imported from test.fixtures.paths)
6. Data-related fixtures (imported from test.fixtures.data)

These fixtures are used to set up consistent testing environments and configurations
across different test files in the PCNtoolkit testing suite. They provide reusable
components for creating and configuring normative models, particularly focusing on
the Hierarchical Bayesian Regression (HBR) approach.
"""

N_SAMPLES = 15
N_TUNES = 5
N_PYMC_CORES = 4
N_CHAINS = 2


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
    return N_SAMPLES


@pytest.fixture
def n_tunes():
    return N_TUNES


@pytest.fixture
def n_pymc_cores():
    return N_PYMC_CORES


@pytest.fixture
def n_mcmc_chains():
    return 2


@pytest.fixture
def sample_args():
    return {"draws": N_SAMPLES, "tune": N_TUNES, "pymc_cores": N_PYMC_CORES}


@pytest.fixture
def norm_args(log_dir, save_dir):
    return {"log_dir": log_dir, "save_dir": save_dir}


@pytest.fixture
def norm_conf_dict_for_generic_model(log_dir, save_dir):
    return {"log_dir": log_dir, "save_dir": save_dir}


@pytest.fixture
def norm_conf_for_generic_model(log_dir, save_dir):
    return NormConf(save_dir=save_dir)


@pytest.fixture
def norm_conf_dict_for_hbr_test_model(
    alg,
    fit_files,
    maskfile,
    test_files,
    savemodel,
    save_dir,
):
    responsefile, covfile, trbefile = fit_files
    testcov, testresp, tsbefile = test_files
    return {
        "resp": responsefile,
        "maskfile": maskfile,
        "cov": covfile,
        "t_resp": testresp,
        "t_cov": testcov,
        "alg": alg,
        "savemodel": savemodel,
        "be": trbefile,
        "t_be": tsbefile,
        "save_dir": save_dir + "/hbr",
    }


@pytest.fixture
def hbr_conf_dict(save_dir, fit_files, test_files, maskfile):
    responsefile, covfile, trbefile = fit_files
    testresp, testcov, tsbefile = test_files
    return {
        "resp": responsefile,
        "maskfile": maskfile,
        "cov": covfile,
        "t_resp": testresp,
        "t_cov": testcov,
        "alg": "hbr",
        "savemodel": True,
        "be": trbefile,
        "t_be": tsbefile,
        "save_dir": save_dir + "/hbr",
        "basis_function": "bspline",
        "linear_mu": True,
        "linear_sigma": True,
        "mapping_sigma": "softplus",
        "pymc_cores": N_PYMC_CORES,
        "draws": N_SAMPLES,
        "tune": N_TUNES,
        "chains": N_CHAINS,
    }


@pytest.fixture
def norm_conf_for_hbr_test_model(savemodel, save_dir):
    return NormConf(
        savemodel=savemodel,
        save_dir=save_dir + "/hbr",
        basis_function="bspline",
        basis_function_kwargs={"order": 3, "nknots": 5},
        inscaler="standardize",
        outscaler="standardize",
        saveresults=True,
    )


@pytest.fixture
def mu():
    return make_prior(linear=True, intercept=make_prior(random=True), slope=make_prior(dist_name="Normal", dist_params=(0, 10)))


@pytest.fixture
def sigma():
    return make_prior(
        linear=True,
        slope=make_prior(dist_name="Normal", dist_params=(0, 2.0)),
        intercept=make_prior(dist_name="Normal", dist_params=(0, 2.0)),
        mapping="softplus",
        mapping_params=(0.0, 3.0),
    )


@pytest.fixture
def hbrconf(mu, sigma):
    return HBRConf(
        draws=N_SAMPLES,
        tune=N_TUNES,
        chains=N_CHAINS,
        pymc_cores=N_PYMC_CORES,
        likelihood="SHASHb",
        mu=mu,
        sigma=sigma,
        nuts_sampler="nutpie",
    )


@pytest.fixture
def hbr(hbrconf: HBRConf):
    return HBR("test_hbr", reg_conf=hbrconf)


@pytest.fixture
def new_norm_hbr_model(norm_conf_for_hbr_test_model, hbrconf):
    return NormHBR(norm_conf_for_hbr_test_model, hbrconf)


@pytest.fixture
def fitted_norm_hbr_model(new_norm_hbr_model: NormHBR, norm_data_from_arrays: NormData):
    new_norm_hbr_model.fit(norm_data_from_arrays)
    return new_norm_hbr_model
