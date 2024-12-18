
from pcntoolkit.dataio.norm_data import NormData
from pcntoolkit.normative_model.norm_blr import NormBLR
from test.fixtures.blr_model_fixtures import *
from test.fixtures.norm_data_fixtures import *
from test.fixtures.path_fixtures import *


def test_fit(new_norm_blr_model: NormBLR, norm_data_from_arrays: NormData):
    new_norm_blr_model.fit(norm_data_from_arrays)
    for model in new_norm_blr_model.regression_models.values():
        assert model.is_fitted


def test_predict(fitted_norm_blr_model: NormBLR, norm_data_from_arrays: NormData):
    fitted_norm_blr_model.predict(norm_data_from_arrays)
    print(fitted_norm_blr_model.norm_conf.save_dir)
    for model in fitted_norm_blr_model.regression_models.values():
        assert model.is_fitted

def test_fit_predict(new_norm_blr_model: NormBLR, norm_data_from_arrays: NormData, test_norm_data_from_arrays: NormData):
    new_norm_blr_model.fit_predict(norm_data_from_arrays, test_norm_data_from_arrays)
    for model in new_norm_blr_model.regression_models.values():
        assert model.is_fitted

def test_save_load(new_norm_blr_model: NormBLR, norm_data_from_arrays: NormData, test_norm_data_from_arrays: NormData):
    new_norm_blr_model.fit(norm_data_from_arrays)
    new_norm_blr_model.save("test_save_load")
    loaded_norm_blr_model = NormBLR.load("test_save_load")
    try:
        loaded_norm_blr_model.predict(test_norm_data_from_arrays)
    except Exception as e:
        print(e)
        assert False
    assert loaded_norm_blr_model.regression_models["response_var_0"].is_fitted
