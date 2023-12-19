

"""
Putting the factory method in the NormBase class would be more elegant, but then we would have to import all the regression models in the NormBase class, which would create a circular dependency. Therefore, we put the factory method in this separate file.
"""

from pcntoolkit.normative_model.norm_base import NormBase
from pcntoolkit.normative_model.norm_blr import NormBLR
from pcntoolkit.normative_model.norm_conf import NormConf
from pcntoolkit.normative_model.norm_gpr import NormGPR
from pcntoolkit.normative_model.norm_hbr import NormHBR
from pcntoolkit.regression_model.reg_conf import RegConf


def create_normative_model(norm_conf: NormConf, reg_conf: RegConf) -> NormBase:
    """
    Factory method for creating a normative model.
    """
    # If the subclass of the regconf is HBRConf, then create a NormHBR.
    if reg_conf.__class__.__name__ == 'HBRConf':
        return NormHBR(norm_conf, reg_conf)
    
    # If the subclass of the regconf is BLRConf, then create a NormBLR.
    elif reg_conf.__class__.__name__ == 'BLRConf':
        return NormBLR(norm_conf, reg_conf)
    
    # If the subclass of the regconf is GPRConf, then create a NormGPR.
    elif reg_conf.__class__.__name__ == 'GPRConf':
        return NormGPR(norm_conf, reg_conf)
    
    # If the subclass of the regconf is not HBRConf, BLRConf, or GPRConf, then raise a ValueError.
    else:
        raise ValueError(f'Unknown regression model configuration: {reg_conf.__class__.__name__}')
    