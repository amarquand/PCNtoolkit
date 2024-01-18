import argparse
import sys
from unittest.mock import patch

import pytest

import pcntoolkit.normative as normative


@pytest.fixture
def conf_dict():
    return {
        "responses": "/home/stijn/Projects/PCNtoolkit/pytest_tests/resources/data/responses.csv",
        "maskfile": None,
        "covfile": "/home/stijn/Projects/PCNtoolkit/pytest_tests/resources/data/covariates.csv",
        "cvfolds": None,
        "testcov": "/home/stijn/Projects/PCNtoolkit/pytest_tests/resources/data/covariates_test.csv",
        "testresp": "/home/stijn/Projects/PCNtoolkit/pytest_tests/resources/data/responses_test.csv",
        "alg": "hbr",
        "savemodel": True,
        "trbefile": "/home/stijn/Projects/PCNtoolkit/pytest_tests/resources/data/batch_effects.csv",
        "tsbefile": "/home/stijn/Projects/PCNtoolkit/pytest_tests/resources/data/batch_effects_test.csv",
        "save_dir": "/home/stijn/Projects/PCNtoolkit/pytest_tests/resources/save_load_test",
        "log_dir": "/home/stijn/Projects/PCNtoolkit/pytest_tests/resources/log_test",
    }


def test_fit(conf_dict):
    conf_dict["func"] = "fit"
    func = conf_dict.pop("func")
    if func == "fit":
        normative.fit(conf_dict)
    else:
        raise ValueError(f"Unknown function {func}.")


def test_predict(conf_dict):
    conf_dict["func"] = "predict"
    func = conf_dict.pop("func")
    if func == "predict":
        normative.predict(conf_dict)
    else:
        raise ValueError(f"Unknown function {func}.")
