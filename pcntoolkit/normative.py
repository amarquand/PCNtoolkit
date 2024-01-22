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
    create_normative_model_from_dict,
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
    normative_model: NormBase = create_normative_model_from_dict(conf_dict)

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
    maskfile = conf_dict.pop("tsbefile", None)

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


def main():
    # conf_dict = get_conf_dict_from_args()
    conf_dict = {
        "responses": "/home/stijn/Projects/PCNtoolkit/pytest_tests/resources/data/responses.csv",
        "func": "fit",
        "maskfile": None,
        "covfile": "/home/stijn/Projects/PCNtoolkit/pytest_tests/resources/data/covariates.csv",
        "cvfolds": None,
        "testcov": None,
        "testresp": None,
        "alg": "hbr",
        "trbefile": "/home/stijn/Projects/PCNtoolkit/pytest_tests/resources/data/batch_effects.csv",
        "save_dir": "/home/stijn/Projects/PCNtoolkit/pytest_tests/resources/save_load_test",
        "log_dir": "/home/stijn/Projects/PCNtoolkit/pytest_tests/resources/log_test",
    }

    # Y = np.random.randn(1000, 2)
    # np.savetxt(conf_dict["responses"], Y)

    func = conf_dict.pop("func")
    if func == "fit":
        fit(conf_dict)
    else:
        raise ValueError(f"Unknown function {func}.")


if __name__ == "__main__":
    main()
