import os
import pytest

from pcntoolkit.dataio.norm_data import NormData
from pcntoolkit.normative_model.norm_blr import NormBLR
from pcntoolkit.normative_model.norm_base import NormBase
from pcntoolkit.normative_model.norm_factory import load_normative_model
from pytest_tests.fixtures.norm_data_fixtures import *
from pytest_tests.fixtures.path_fixtures import *
from pytest_tests.fixtures.blr_model_fixtures import *
    
def test_fit(new_norm_blr_model: NormBLR, norm_data_from_arrays: NormData):
    new_norm_blr_model.fit(norm_data_from_arrays)
    for model in new_norm_blr_model.regression_models.values():
        assert model.is_fitted