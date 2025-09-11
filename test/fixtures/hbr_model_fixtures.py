import pytest

from pcntoolkit.dataio.norm_data import NormData
from pcntoolkit.math_functions.likelihood import NormalLikelihood
from pcntoolkit.math_functions.prior import make_prior
from pcntoolkit.normative_model import NormativeModel
from pcntoolkit.regression_model.hbr import HBR
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
CVFOLDS = 4
ALG = "hbr"
SAVEMODEL = True


@pytest.fixture
def sample_args():
    return {
        "draws": N_SAMPLES,
        "tune": N_TUNES,
        "cores": N_PYMC_CORES,
        "chains": N_CHAINS,
    }


@pytest.fixture(scope="module")
def mu():
    return make_prior(
        linear=True,
        intercept=make_prior(random=True),
        slope=make_prior(dist_name="Normal", dist_params=(0, 10)),
    )


@pytest.fixture(scope="module")
def sigma():
    return make_prior(
        linear=True,
        slope=make_prior(dist_name="Normal", dist_params=(0, 2.0)),
        intercept=make_prior(dist_name="Normal", dist_params=(1.0, 2.0)),
        mapping="softplus",
        mapping_params=(0.0, 3.0),
    )


@pytest.fixture(scope="module")
def hbr_model(mu, sigma):
    return HBR(
        "test_hbr",
        likelihood=NormalLikelihood(mu=mu, sigma=sigma),
        nuts_sampler="nutpie",
        draws=N_SAMPLES,
        tune=N_TUNES,
        chains=N_CHAINS,
        cores=N_PYMC_CORES,
    )


@pytest.fixture(scope="module")
def new_norm_hbr_model(hbr_model, save_dir_hbr):
    return NormativeModel(
        hbr_model,
        save_dir=save_dir_hbr,
        inscaler="standardize",
        outscaler="standardize",
    )


@pytest.fixture(scope="module")
def train_and_save_hbr_model(
    new_norm_hbr_model,
    norm_data_from_arrays: NormData,
):
    # pass
    if os.path.exists(new_norm_hbr_model.save_dir):
        shutil.rmtree(new_norm_hbr_model.save_dir)
    os.makedirs(new_norm_hbr_model.save_dir, exist_ok=True)
    new_norm_hbr_model.fit(norm_data_from_arrays)
    new_norm_hbr_model.save()
    return new_norm_hbr_model


@pytest.fixture(scope="module")
def fitted_norm_hbr_model(train_and_save_hbr_model):
    loaded_model = NormativeModel.load(train_and_save_hbr_model.save_dir)
    # loaded_model = NormativeModel.load(save_dir_hbr)
    return loaded_model
