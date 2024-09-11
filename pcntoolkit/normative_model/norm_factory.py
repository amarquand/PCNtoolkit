"""
Putting the factory method in the NormBase class would be more elegant,
but then we would have to import all the regression models in the NormBase class, which would create a circular dependency.
Therefore, we put the factory method in this separate file.
"""

import json
import os

from pcntoolkit.normative_model.norm_base import NormBase
from pcntoolkit.normative_model.norm_blr import NormBLR
from pcntoolkit.normative_model.norm_conf import NormConf
from pcntoolkit.normative_model.norm_gpr import NormGPR
from pcntoolkit.normative_model.norm_hbr import NormHBR
from pcntoolkit.regression_model.blr.blr_conf import BLRConf
from pcntoolkit.regression_model.gpr.gpr_conf import GPRConf
from pcntoolkit.regression_model.hbr.hbr_conf import HBRConf
from pcntoolkit.regression_model.reg_conf import RegConf


def create_normative_model(norm_conf: NormConf, reg_conf: RegConf) -> NormBase:
    """
    Factory method for creating a normative model.
    """
    # If the subclass of the regconf is HBRConf, then create a NormHBR.
    if reg_conf.__class__.__name__ == "HBRConf":
        return NormHBR(norm_conf, reg_conf)

    # If the subclass of the regconf is BLRConf, then create a NormBLR.
    elif reg_conf.__class__.__name__ == "BLRConf":
        return NormBLR(norm_conf, reg_conf)

    # If the subclass of the regconf is GPRConf, then create a NormGPR.
    elif reg_conf.__class__.__name__ == "GPRConf":
        return NormGPR(norm_conf, reg_conf)

    # If the subclass of the regconf is not HBRConf, BLRConf, or GPRConf, then raise a ValueError.
    else:
        raise ValueError(
            f"Unknown regression model configuration: {reg_conf.__class__.__name__}"
        )


def load_normative_model(path) -> NormBase:
    """
    Loads the normative model from a directory.
    """
    try:
        with open(os.path.join(path, "metadata.json"), "r") as f:
            metadata = json.load(f)
    except FileNotFoundError:
        raise FileNotFoundError(f"Path {path} does not exist.")

    norm_conf = NormConf.from_dict(metadata["norm_conf"])
    model_name = norm_conf.normative_model_name

    if model_name == "NormHBR":
        return NormHBR.load(path)
    elif model_name == "NormBLR":
        return NormBLR.load(path)
    elif model_name == "NormGPR":
        return NormGPR.load(path)
    else:
        raise ValueError(f"Model name {model_name} not recognized.")


def create_normative_model_from_args(args: dict[str, str]) -> NormBase:
    """
    Creates a normative model from command line arguments.
    """
    norm_conf = NormConf.from_args(args)
    if args["alg"] == "hbr":
        reg_conf = HBRConf.from_args(args)
    elif args["alg"] == "blr":
        reg_conf = BLRConf.from_args(args)
    elif args["alg"] == "gpr":
        reg_conf = GPRConf.from_args(args)
    return create_normative_model(norm_conf, reg_conf)
