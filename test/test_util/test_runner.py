from pcntoolkit.util.runner import Runner
from test.fixtures.test_model_fixtures import *
import shutil


def cleanup(model, runner):
    shutil.rmtree(os.path.join(model.save_dir))
    shutil.rmtree(os.path.join(runner.log_dir))
    shutil.rmtree(os.path.join(runner.temp_dir))

def test_runner_fit(new_norm_test_model: NormativeModel, norm_data_from_arrays: NormData):
    runner = Runner(cross_validate=False, parallelize=False)
    runner.fit(new_norm_test_model, norm_data_from_arrays, observe=True)
    assert new_norm_test_model.is_fitted
    assert os.path.exists(os.path.join(new_norm_test_model.save_dir, "model", "normative_model.json"))
    cleanup(new_norm_test_model, runner)

def test_runner_fit_kfold(new_norm_test_model: NormativeModel, norm_data_from_arrays: NormData):
    runner = Runner(cross_validate=True, cv_folds=2, parallelize=False)
    runner.fit(new_norm_test_model, norm_data_from_arrays, observe=True)
    assert new_norm_test_model.is_fitted
    assert os.path.exists(os.path.join(new_norm_test_model.save_dir, "folds", "fold_0", "model", "normative_model.json"))
    assert os.path.exists(os.path.join(new_norm_test_model.save_dir, "folds", "fold_1", "model", "normative_model.json"))
    cleanup(new_norm_test_model, runner)

def test_runner_predict(new_norm_test_model: NormativeModel, norm_data_from_arrays: NormData):
    runner = Runner(cross_validate=False, parallelize=False)
    runner.predict(new_norm_test_model, norm_data_from_arrays, observe=True)
    assert new_norm_test_model.is_fitted
    assert os.path.exists(os.path.join(new_norm_test_model.save_dir, "predictions", "normative_model.json"))
    assert os.path.exists(os.path.join(new_norm_test_model.save_dir, "results",))
    assert os.path.exists(os.path.join(new_norm_test_model.save_dir, "plots" ,f"centiles_{norm_data_from_arrays.response_vars[0]}.csv"))
    cleanup(new_norm_test_model, runner)

def test_runner_predict_kfold(new_norm_test_model: NormativeModel, norm_data_from_arrays: NormData):
    runner = Runner(cross_validate=True, cv_folds=2, parallelize=False)
    # assert this throws an error
    with pytest.raises(ValueError):
        runner.predict(new_norm_test_model, norm_data_from_arrays, observe=True)

def test_runner_fit_predict(new_norm_test_model: NormativeModel, norm_data_from_arrays: NormData):
    train, test = norm_data_from_arrays.train_test_split(test_size=0.2)
    runner = Runner(cross_validate=False, parallelize=False)
    runner.fit_predict(new_norm_test_model, train, test, observe=True)
    assert new_norm_test_model.is_fitted
    assert os.path.exists(os.path.join(new_norm_test_model.save_dir, "model", "normative_model.json"))
    assert os.path.exists(os.path.join(new_norm_test_model.save_dir, "results",))
    assert os.path.exists(os.path.join(new_norm_test_model.save_dir, "plots" ,f"centiles_{test.response_vars[0]}.csv"))
    cleanup(new_norm_test_model, runner)
 
def test_runner_fit_predict_kfold(new_norm_test_model: NormativeModel, norm_data_from_arrays: NormData):
    train, test = norm_data_from_arrays.train_test_split(test_size=0.2)
    runner = Runner(cross_validate=True, cv_folds=2, parallelize=False)
    runner.fit_predict(new_norm_test_model, train, test, observe=True)
    assert new_norm_test_model.is_fitted
    assert os.path.exists(os.path.join(new_norm_test_model.save_dir, "folds", "fold_0", "model", "normative_model.json"))
    assert os.path.exists(os.path.join(new_norm_test_model.save_dir, "folds", "fold_1", "model", "normative_model.json"))
    assert os.path.exists(os.path.join(new_norm_test_model.save_dir, "folds", "fold_0", "results",))
    assert os.path.exists(os.path.join(new_norm_test_model.save_dir, "folds", "fold_1", "results",))
    assert os.path.exists(os.path.join(new_norm_test_model.save_dir, "folds", "fold_0", "plots" ,f"centiles_{test.response_vars[0]}.csv"))
    assert os.path.exists(os.path.join(new_norm_test_model.save_dir, "folds", "fold_1", "plots" ,f"centiles_{test.response_vars[0]}.csv"))
    cleanup(new_norm_test_model, runner)


