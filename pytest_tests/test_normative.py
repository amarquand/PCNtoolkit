import argparse
import sys
from unittest.mock import patch

import pytest

import pcntoolkit.normative as normative

from pytest_tests.fixtures.model_fixtures import hbr_conf_dict


def test_fit(hbr_conf_dict):
    hbr_conf_dict["func"] = "fit"
    normative.fit(hbr_conf_dict)


def test_predict(hbr_conf_dict):
    hbr_conf_dict["func"] = "predict"
    normative.predict(hbr_conf_dict)


def test_estimate(hbr_conf_dict):
    hbr_conf_dict["func"] = "estimate"
    normative.estimate(hbr_conf_dict)


# bash_command = "python /home/stijn/Projects/PCNtoolkit/pcntoolkit/normative.py /home/stijn/Projects/PCNtoolkit/pytest_tests/resources/data/responses.csv -f estimate -a hbr -c /home/stijn/Projects/PCNtoolkit/pytest_tests/resources/data/covariates.csv -t /home/stijn/Projects/PCNtoolkit/pytest_tests/resources/data/covariates_test.csv -r /home/stijn/Projects/PCNtoolkit/pytest_tests/resources/data/responses_test.csv trbefile=/home/stijn/Projects/PCNtoolkit/pytest_tests/resources/data/batch_effects.csv tsbefile=/home/stijn/Projects/PCNtoolkit/pytest_tests/resources/data/batch_effects_test.csv save_dir=/home/stijn/Projects/PCNtoolkit/pytest_tests/resources/save_load_test log_dir=/home/stijn/Projects/PCNtoolkit/pytest_tests/resources/log_test"
