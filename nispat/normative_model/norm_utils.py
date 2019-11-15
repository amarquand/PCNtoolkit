from norm_blr import NormBLR
from norm_gpr import NormGPR
from norm_rfa import NormRFA
from norm_hbr import NormHBR
from norm_np import NormNP

def norm_init(X, y=None, theta=None, alg='gpr', configparam=None):
    if alg == 'gpr':
        nm = NormGPR(X, y, theta)
    elif alg =='blr':
        nm = NormBLR(X, y, theta, configparam)
    elif alg == 'rfa':
        nm = NormRFA(X, y, theta, configparam)
    elif alg == 'hbr':
        nm = NormHBR(X, y, configparam)
    elif alg == 'np':
        nm = NormNP(X, y, configparam)
    else:
        raise(ValueError, "Algorithm " + alg + " not known.")
        
    return nm