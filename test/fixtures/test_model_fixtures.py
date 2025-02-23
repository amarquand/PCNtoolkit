import pytest

from pcntoolkit.dataio.norm_data import NormData
from pcntoolkit.normative_model import NormativeModel
from pcntoolkit.regression_model.test_model import TestModel
from test.fixtures.norm_data_fixtures import *
from test.fixtures.path_fixtures import *


@pytest.fixture
def test_model_args(save_dir_test_model):
    return {
        "savemodel": False,
        "saveresults": False,
        "evaluate_model": False,
        "saveplots": False,
        "save_dir": save_dir_test_model,
        "inscaler": "standardize",
        "outscaler": "standardize",
        "normative_model_name": "test_model",
        "alg": "test_model",
        "success_ratio": 1.0,
    }


@pytest.fixture
def test_model():
    return TestModel("test_model")


@pytest.fixture
def new_norm_test_model(test_model, save_dir_test_model):
    return NormativeModel(test_model, save_dir=save_dir_test_model)


@pytest.fixture
def fitted_norm_test_model(new_norm_test_model: NormativeModel, norm_data_from_arrays: NormData):
    new_norm_test_model.fit(norm_data_from_arrays)
    return new_norm_test_model
