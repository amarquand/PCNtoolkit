
from pcntoolkit.normative_model.norm_conf import NormConf

"""
This file contains tests for the NormConf class in the PCNtoolkit.

The tests cover the following aspects:
1. Creating NormConf objects from arguments
2. Converting NormConf objects to dictionaries
3. Creating NormConf objects from dictionaries
"""


def test_norm_conf_to_dict():

    norm_conf_from_args = NormConf.from_args(
        {
            "savemodel": True,
            "saveresults": True,
            "save_dir": "wow",
            "basis_function": "polynomial",
            "basis_column": 10,
            "order": 3,
            "nknots": 5,
            "inscaler": "minmax",
            "outscaler": "minmax",
            "normative_model_name": None,
        }
    )

    assert norm_conf_from_args.to_dict() == {
        "savemodel": True,
        "saveresults": True,
        "save_dir": "wow",
        "basis_function": "polynomial",
        "basis_column": 10,
        "order": 3,
        "nknots": 5,
        "inscaler": "minmax",
        "outscaler": "minmax",
        "normative_model_name": None,
    }

    norm_conf = NormConf(
        savemodel=True,
        saveresults=True,
        save_dir="wow",
        basis_function="polynomial",
        basis_column=10,
        order=3,
        nknots=5,
        inscaler="minmax",
        outscaler="minmax",
        normative_model_name=None,
    )

    assert norm_conf.to_dict() == {
        "savemodel": True,
        "saveresults": True,
        "save_dir": "wow",
        "basis_function": "polynomial",
        "basis_column": 10,
        "order": 3,
        "nknots": 5,
        "inscaler": "minmax",
        "outscaler": "minmax",
        "normative_model_name": None,
    }

    norm_conf2 = NormConf.from_dict(norm_conf.to_dict())

    assert norm_conf2.to_dict() == norm_conf.to_dict()
