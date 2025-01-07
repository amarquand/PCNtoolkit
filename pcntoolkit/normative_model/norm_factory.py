"""
Factory methods for creating and loading normative models.

This module provides functions to create and load different types of normative models
based on given configurations or command line arguments.

Functions
---------
create_normative_model(norm_conf, reg_conf)
    Create a normative model based on the given configuration.

load_normative_model(path)
    Load a normative model from a specified path.

create_normative_model_from_args(args)
    Create a normative model from command line arguments.

Examples
--------
>>> norm_conf = NormConf()
>>> reg_conf = HBRConf()
>>> model = create_normative_model(norm_conf, reg_conf)
>>> isinstance(model, NormHBR)
True
"""

import json
import os

from pcntoolkit.normative_model.norm_base import NormBase
from pcntoolkit.normative_model.norm_blr import NormBLR
from pcntoolkit.normative_model.norm_conf import NormConf

# from pcntoolkit.normative_model.norm_gpr import NormGPR
from pcntoolkit.normative_model.norm_hbr import NormHBR
from pcntoolkit.regression_model.blr.blr_conf import BLRConf
from pcntoolkit.regression_model.hbr.hbr_conf import HBRConf
from pcntoolkit.regression_model.reg_conf import RegConf


def create_normative_model(norm_conf: NormConf, reg_conf: RegConf) -> NormBase:
    """
    Create a normative model based on the given configuration.

    Parameters
    ----------
    norm_conf : NormConf
        The normative model configuration.
    reg_conf : RegConf
        The regression model configuration.

    Returns
    -------
    NormBase
        An instance of a normative model.

    Raises
    ------
    ValueError
        If the regression model configuration is unknown.
    """
    if isinstance(reg_conf, HBRConf):
        return NormHBR(norm_conf, reg_conf)
    elif isinstance(reg_conf, BLRConf):
        return NormBLR(norm_conf, reg_conf)
    # elif isinstance(reg_conf, GPRConf):
    #     return NormGPR(norm_conf, reg_conf)
    else:
        raise ValueError(f"Unknown regression model configuration: {reg_conf.__class__.__name__}")


def load_normative_model(path: str) -> NormBase:
    """
    Load a normative model from a specified path.

    Parameters
    ----------
    path : str
        The file path to load the normative model from.

    Returns
    -------
    NormBase
        An instance of a loaded normative model.

    Raises
    ------
    FileNotFoundError
        If the specified path does not exist.
    ValueError
        If the model name is not recognized.
    """
    try:
        with open(os.path.join(path, "model", "normative_model.json"), mode="r", encoding="utf-8") as f:
            metadata = json.load(f)
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"Path {path} does not exist.") from exc

    norm_conf = NormConf.from_dict(metadata["norm_conf"])
    model_name = norm_conf.normative_model_name

    if model_name == "NormHBR":
        return NormHBR.load(path)
    elif model_name == "NormBLR":
        return NormBLR.load(path)
    # elif model_name == "NormGPR":
    #     return NormGPR.load(path)
    else:
        raise ValueError(f"Model name {model_name} not recognized.")


def create_normative_model_from_args(args: dict[str, str]) -> NormBase:
    """
    Create a normative model from command line arguments.

    Parameters
    ----------
    args : dict[str, str]
        A dictionary of command line arguments.

    Returns
    -------
    NormBase
        An instance of a normative model.

    Raises
    ------
    ValueError
        If the regression model specified in the arguments is unknown.
    """
    norm_conf = NormConf.from_args(args)
    if args["alg"] == "hbr":
        reg_conf = HBRConf.from_args(args)
    elif args["alg"] == "blr":
        reg_conf = BLRConf.from_args(args)
    # elif args["alg"] == "gpr":
    #     reg_conf = GPRConf.from_args(args)
    else:
        raise ValueError(f"Unknown regression model: {args['alg']}")
    return create_normative_model(norm_conf, reg_conf)
