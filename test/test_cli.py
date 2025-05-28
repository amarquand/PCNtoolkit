import os
import shutil
import subprocess
import sys

import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest


@pytest.fixture
def save_dir():
    return "test/resources/save_dir"


@pytest.fixture
def cov_txt_path():
    # synthesize some data
    cov = np.random.rand(100, 2)
    np.savetxt("test/resources/cov.txt", cov)
    return "test/resources/cov.txt"


@pytest.fixture
def resp_txt_path():
    resp = np.random.rand(100, 3)
    np.savetxt("test/resources/resp.txt", resp)
    return "test/resources/resp.txt"

@pytest.fixture 
def batch_effect_path():
    batch_effect1 = np.random.choice(2, (100,1))
    batch_effect2 = np.random.choice(2, (100,1))
    batch_effect = np.stack((batch_effect1, batch_effect2), axis=1)
    np.savetxt("test/resources/batch_effect.txt", batch_effect)
    return "test/resources/batch_effect.txt"


def test_simple_cli(cov_txt_path, resp_txt_path, save_dir):
    # create the save directory
    this_function_name = sys._getframe().f_code.co_name
    save_dir = os.path.join(save_dir, this_function_name)
    os.makedirs(save_dir, exist_ok=True)
    cmd = f"normative -c {cov_txt_path} -a blr -r {resp_txt_path} save_dir={save_dir}"
    # check that the command runs without error
    subprocess.run(cmd, shell=True)
    assert os.path.exists(os.path.join(save_dir))
    assert os.path.exists(os.path.join(save_dir, "model","normative_model.json"))
    assert os.path.exists(os.path.join(save_dir, "plots","centiles_response_var_0_fit_data_harmonized.png"))
    assert os.path.exists(os.path.join(save_dir, "results","centiles_fit_data.csv"))

    # Remove the save directory
    shutil.rmtree(save_dir)
    shutil.rmtree("temp")
    shutil.rmtree("logs")

def test_simple_cli_with_folds(cov_txt_path, resp_txt_path, save_dir):
    # create the save directory
    this_function_name = sys._getframe().f_code.co_name
    save_dir = os.path.join(save_dir, this_function_name)
    os.makedirs(save_dir, exist_ok=True)
    cmd = f"normative -c {cov_txt_path} -a blr -r {resp_txt_path} -k 3 save_dir={save_dir}"
    # check that the command runs without error
    subprocess.run(cmd, shell=True)
    assert os.path.exists(os.path.join(save_dir))
    assert os.path.exists(os.path.join(save_dir,"folds", "fold_0", "model","normative_model.json"))
    assert os.path.exists(os.path.join(save_dir,"folds", "fold_1", "plots","centiles_response_var_0_fit_data_fold_1_train_harmonized.png"))
    assert os.path.exists(os.path.join(save_dir,"folds", "fold_2", "results","centiles_fit_data_fold_2_train.csv"))

    # Remove the save directory
    shutil.rmtree(save_dir)
    shutil.rmtree("temp")
    shutil.rmtree("logs")


# normative -c test/resources/cov.txt -a blr -r test/resources/resp.txt -k 3 save_dir=test/resources/save_dir/test_simple_cli_with_folds

# def test_with_batch_effects(cov_txt_path, resp_txt_path, batch_effect_path, save_dir):
#     # create the save directory
#     this_function_name = sys._getframe().f_code.co_name
#     save_dir = os.path.join(save_dir, this_function_name)
#     os.makedirs(save_dir, exist_ok=True)
#     cmd = f"normative -c {cov_txt_path} -a blr -r {resp_txt_path} save_dir={save_dir} "
#     # check that the command runs without error
#     subprocess.run(cmd, shell=True)
#     assert os.path.exists(os.path.join(save_dir))
#     assert os.path.exists(os.path.join(save_dir, "model","normative_model.json"))
#     assert os.path.exists(os.path.join(save_dir, "plots","centiles_response_var_0_fit_data_harmonized.png"))
#     assert os.path.exists(os.path.join(save_dir, "results","centiles_fit_data.csv"))

#     # Remove the save directory
#     shutil.rmtree(save_dir)