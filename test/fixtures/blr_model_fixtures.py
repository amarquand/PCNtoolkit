import pytest

from pcntoolkit.dataio.norm_data import NormData
from pcntoolkit.math_functions.basis_function import BasisFunction, BsplineBasisFunction, LinearBasisFunction
from pcntoolkit.normative_model import NormativeModel
from pcntoolkit.regression_model.blr import BLR
from test.fixtures.norm_data_fixtures import *
from test.fixtures.path_fixtures import *


@pytest.fixture
def cvfolds():
    return 4


@pytest.fixture
def savemodel():
    return True


@pytest.fixture
def blr_model():
    return BLR(
        "test_blr",
        n_iter=1000,
        tol=1e-3,
        ard=False,
        optimizer="l-bfgs-b",
        l_bfgs_b_l=0.1,
        l_bfgs_b_epsilon=0.1,
        l_bfgs_b_norm="l1",
        heteroskedastic=True,
        intercept=True,
        fixed_effect=True,
        warp_name="WarpSinhArcsinh",
        basis_function_mean=BsplineBasisFunction(basis_column=0, degree=3, nknots=5),
        warp_reparam=True,
    )


@pytest.fixture
def fitted_blr_model(blr_model: BLR, norm_data_from_arrays: NormData, fitted_norm_blr_model: NormativeModel):
    if os.path.exists(fitted_norm_blr_model.save_dir):
        shutil.rmtree(fitted_norm_blr_model.save_dir)
    os.makedirs(fitted_norm_blr_model.save_dir, exist_ok=True)
    be_maps = fitted_norm_blr_model.batch_effects_maps
    response_var = norm_data_from_arrays.response_vars[0]
    X, be, be_maps, Y, _ = fitted_norm_blr_model.extract_data(norm_data_from_arrays.sel(response_vars=response_var))
    blr_model.fit(X, be, be_maps, Y)
    return blr_model

@pytest.fixture
def new_norm_blr_model(blr_model, save_dir_blr):
    if os.path.exists(save_dir_blr):
        shutil.rmtree(save_dir_blr)
    os.makedirs(save_dir_blr, exist_ok=True)
    return NormativeModel(
        blr_model,
        save_dir=save_dir_blr,
        inscaler="standardize",
        outscaler="standardize",
    )


@pytest.fixture
def fitted_norm_blr_model(new_norm_blr_model: NormativeModel, norm_data_from_arrays: NormData):
    print("removing items")
    if os.path.exists(new_norm_blr_model.save_dir):
        shutil.rmtree(new_norm_blr_model.save_dir)
    os.makedirs(new_norm_blr_model.save_dir, exist_ok=True)
    new_norm_blr_model.fit(norm_data_from_arrays)
    return new_norm_blr_model
