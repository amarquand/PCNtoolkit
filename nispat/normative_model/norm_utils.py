from norm_blr import NormBLR
from norm_gpr import NormGPR
from norm_rfa import NormRFA
from norm_hlr import NormHLR

def norm_init(X, y=None, theta=None, alg='gpr', configparam=None):
    if alg == 'gpr':
        nm = NormGPR(X, y, theta)
    elif alg =='blr':
        nm = NormBLR(X, y, theta, configparam)
    elif alg == 'rfa':
        nm = NormRFA(X, y, theta, configparam)
    elif alg == 'hlr':
        nm = NormHLR(X, y, configparam = {'age':0, 'site':1, 'gender':2})
    else:
        raise(ValueError, "Algorithm " + alg + " not known.")
        
    return nm