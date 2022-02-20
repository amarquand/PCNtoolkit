try:  # run as a package if installed
    from pcntoolkit.normative_model.norm_blr import NormBLR
    from pcntoolkit.normative_model.norm_gpr import NormGPR
    from pcntoolkit.normative_model.norm_rfa import NormRFA
    from pcntoolkit.normative_model.norm_hbr import NormHBR
    from pcntoolkit.normative_model.norm_np import NormNP
except:
    from norm_blr import NormBLR
    from norm_gpr import NormGPR
    from norm_rfa import NormRFA
    from norm_hbr import NormHBR
    from norm_np import NormNP

def norm_init(X, y=None, theta=None, alg='gpr', **kwargs):
    if alg == 'gpr':
        nm = NormGPR(X=X, y=y, theta=theta, **kwargs)
    elif alg =='blr':
        nm = NormBLR(X=X, y=y, theta=theta, **kwargs)
    elif alg == 'rfa':
        nm = NormRFA(X=X, y=y, theta=theta, **kwargs)
    elif alg == 'hbr':
        nm = NormHBR(**kwargs)
    elif alg == 'np':
        nm = NormNP(X=X, y=y, **kwargs)
    else:
        raise(ValueError, "Algorithm " + alg + " not known.")
        
    return nm