from pcntoolkit.normative_model.norm_base import NormBase
from pcntoolkit.normative_model.norm_conf import NormConf
from pcntoolkit.normative_model.norm_blr import NormBLR
from pcntoolkit.normative_model.norm_gpr import NormGPR
from pcntoolkit.normative_model.norm_hbr import NormHBR
from pcntoolkit.regression_model.reg_conf import RegConf

"""
Putting the factory method in the NormBase class would be more elegant, but then we would have to import all the regression models in the NormBase class, which would create a circular dependency. Therefore, we put the factory method in this separate file.
"""

def create_normative_model(normconf: NormConf, regconf: RegConf) -> NormBase:
    """
    Factory method for creating a normative model.
    """
    # If the subclass of the regconf is HBRConf, then create a NormHBR.
    if regconf.__class__.__name__ == 'HBRConf':
        return NormHBR(normconf, regconf)
    
    # If the subclass of the regconf is BLRConf, then create a NormBLR.
    elif regconf.__class__.__name__ == 'BLRConf':
        return NormBLR(normconf, regconf)
    
    # If the subclass of the regconf is GPRConf, then create a NormGPR.
    elif regconf.__class__.__name__ == 'GPRConf':
        return NormGPR(normconf, regconf)
    
    # If the subclass of the regconf is not HBRConf, BLRConf, or GPRConf, then raise a ValueError.
    else:
        raise ValueError(f'Unknown regression model configuration: {regconf.__class__.__name__}')
    