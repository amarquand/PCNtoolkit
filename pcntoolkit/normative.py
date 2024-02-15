import argparse
import os
import sys

import numpy as np
from sklearn.model_selection import StratifiedKFold

from pcntoolkit.dataio import fileio
from pcntoolkit.dataio.norm_data import NormData
from pcntoolkit.normative_model.norm_base import NormBase
from pcntoolkit.normative_model.norm_conf import NormConf
from pcntoolkit.normative_model.norm_factory import (
    create_normative_model,
    create_normative_model_from_args,
    load_normative_model,
)


def fit(conf_dict: dict):
    """
    Fit a normative model.

    :param conf_dict: Dictionary containing configuration options
    """

    # Load the data
    fit_data = load_data(conf_dict)

    # Create the normative model
    normative_model: NormBase = create_normative_model_from_args(conf_dict)

    # Fit the normative model
    normative_model.fit(fit_data)

    # Save the normative model
    if normative_model.norm_conf.savemodel:
        normative_model.save()


def predict(conf_dict: dict):
    """
    Predict response variables using a normative model.

    :param conf_dict: Dictionary containing configuration options
    """

    # Load the data
    predict_data = load_data(conf_dict)

    # Load the normative model
    normative_model = load_normative_model(conf_dict["save_dir"])

    # Predicts on new data
    normative_model.predict(predict_data)

    # # Save the predicted response variables
    # fileio.save(Y_pred, os.path.join(conf_dict["save_dir"], "Y_pred.csv"))


def fit_predict(conf_dict: dict):
    """
    Fit a normative model and predict response variables.

    :param conf_dict: Dictionary containing configuration options
    """

    # Load the data
    fit_data = load_data(conf_dict)
    predict_data = load_test_data(conf_dict)

    assert fit_data.is_compatible_with(predict_data)

    # Create the normative model
    normative_model: NormBase = create_normative_model_from_args(conf_dict)

    # Fit and predict the normative model
    normative_model.fit_predict(fit_data, predict_data)

    # Save the normative model
    if normative_model.norm_conf.savemodel:
        normative_model.save()


def estimate(conf_dict: dict):
    fit_predict(conf_dict)


def load_data(conf_dict: dict) -> NormData:
    respfile = conf_dict.pop("responses")
    covfile = conf_dict.pop("covfile")
    maskfile = conf_dict.pop("maskfile", None)

    # Load the covariates
    X = fileio.load(covfile)

    # Load the response variables
    Y, volmask = load_response_vars(respfile, maskfile=maskfile)

    # Load the batch effects
    batch_effects = conf_dict.pop("trbefile", None)
    if batch_effects:
        batch_effects = fileio.load(batch_effects)
    else:
        batch_effects = np.zeros((Y.shape[0], 1))

    data = NormData.from_ndarrays("fit_data", X, Y, batch_effects)

    return data


def load_test_data(conf_dict: dict) -> NormData:
    respfile = conf_dict.pop("testresp")
    covfile = conf_dict.pop("testcov")
    maskfile = conf_dict.pop("maskfile", None)

    # Load the covariates
    X = fileio.load(covfile)

    # Load the response variables
    Y, volmask = load_response_vars(respfile, maskfile=maskfile)

    # Load the batch effects
    batch_effects = conf_dict.pop("tsbefile", None)
    if batch_effects:
        batch_effects = fileio.load(batch_effects)
    else:
        batch_effects = np.zeros((Y.shape[0], 1))

    data = NormData.from_ndarrays("predict_data", X, Y, batch_effects)

    return data


def load_response_vars(datafile, maskfile=None, vol=True):
    """
    Load response variables from file. This will load the data and mask it if
    necessary. If the data is in ascii format it will be converted into a numpy
    array. If the data is in neuroimaging format it will be reshaped into a
    2D array (subjects x variables) and a mask will be created if necessary.

    :param datafile: File containing the response variables
    :param maskfile: Mask file (nifti only)
    :param vol: If True, load the data as a 4D volume (nifti only)
    :returns Y: Response variables
    :returns volmask: Mask file (nifti only)
    """

    if fileio.file_type(datafile) == "nifti":
        dat = fileio.load_nifti(datafile, vol=vol)
        volmask = fileio.create_mask(dat, mask=maskfile)
        Y = fileio.vol2vec(dat, volmask).T
    else:
        Y = fileio.load(datafile)
        volmask = None
        if fileio.file_type(datafile) == "cifti":
            Y = Y.T

    return Y, volmask


def get_argparser():
    #  parse arguments
    parser = argparse.ArgumentParser(description="Normative Modeling")
    parser.add_argument("responses")
    parser.add_argument(
        "-f", "--func", help="Function to call", dest="func", default="estimate"
    )
    parser.add_argument(
        "-m", "--maskfile", help="mask file", dest="maskfile", default=None
    )
    parser.add_argument(
        "-c", "--covariates", help="covariates file", dest="covfile", default=None
    )
    parser.add_argument(
        "-k", "--cvfolds", help="cross-validation folds", dest="cvfolds", default=None
    )
    parser.add_argument(
        "-t", "--testcov", help="covariates (test data)", dest="testcov", default=None
    )
    parser.add_argument(
        "-r", "--testresp", help="responses (test data)", dest="testresp", default=None
    )
    parser.add_argument("-a", "--alg", help="algorithm", dest="alg", default="gpr")
    # parser.add_argument("-x", '--configparam', help="algorithm specific config options", dest="configparam", default=None, nargs=)
    return parser


def get_conf_dict_from_args():
    parser = get_argparser()
    known, unknown = parser.parse_known_args()

    conf_dict = {}

    for arg in vars(known):
        conf_dict[arg] = getattr(known, arg)

    for arg in unknown:
        if "=" in arg:
            key, value = arg.split("=")
            if key in conf_dict:
                raise ValueError(f"Argument {key} is specified twice.")
            conf_dict[key] = value

    for k, v in conf_dict.items():
        if v:
            try:
                conf_dict[k] = int(v)
            except ValueError:
                try:
                    conf_dict[k] = float(v)
                except ValueError:
                    try:
                        if v.lower() == "true":
                            conf_dict[k] = True
                        elif v.lower() == "false":
                            conf_dict[k] = False
                    except AttributeError:
                        pass
    return conf_dict


def make_synthetic_data():
    np.random.seed(42)
    X = np.random.randn(1000, 2)
    Y = np.random.randn(1000, 2)
    batch_effects = []
    for i in range(2):
        batch_effects.append(np.random.choice(list(range(i + 2)), size=1000))
    batch_effects = np.stack(batch_effects, axis=1)
    np.savetxt("covariates.csv", X)
    np.savetxt("responses.csv", Y)
    np.savetxt("batch_effects.csv", batch_effects)

    X = []
    X.append(np.linspace(-3, 4, 200))
    X.append(np.full(200, 0))
    X = np.stack(X, axis=1)
    Y = np.random.randn(200, 2)
    batch_effects = np.zeros((200, 2))
    np.savetxt("covariates_test.csv", X)
    np.savetxt("responses_test.csv", Y)
    np.savetxt("batch_effects_test.csv", batch_effects)


def main():
    # make_synthetic_data()

    # conf_dict = get_conf_dict_from_args()
    conf_dict = {
        "responses": "/home/stijn/Projects/PCNtoolkit/pytest_tests/resources/data/responses.csv",
        "func": "predict",
        "maskfile": None,
        "covfile": "/home/stijn/Projects/PCNtoolkit/pytest_tests/resources/data/covariates.csv",
        "cvfolds": None,
        "testcov": None,
        "testresp": None,
        "alg": "hbr",
        "trbefile": "/home/stijn/Projects/PCNtoolkit/pytest_tests/resources/data/batch_effects.csv",
        "save_dir": "/home/stijn/Projects/PCNtoolkit/pytest_tests/resources/hbr/save_load_test",
        "log_dir": "/home/stijn/Projects/PCNtoolkit/pytest_tests/resources/hbr/log_test",
        "basis_function": "bspline",
    }

    func = conf_dict.pop("func")
    if func == "fit":
        fit(conf_dict)
    elif func == "predict":
        predict(conf_dict)
    elif func == "estimate":
        estimate(conf_dict)
    else:
        raise ValueError(f"Unknown function {func}.")


if __name__ == "__main__":
    main()
