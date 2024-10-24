
# Third-party imports
from pytensor.scalar.basic import BinaryScalarOp, ScalarOp, upgrade_to_float
from pytensor.gradient import grad_not_implemented

import scipy.special as spp

class KnuOp(BinaryScalarOp):
    """
    Modified Bessel function of the second kind, pytensor wrapper for scipy.special.kv
    """
    nfunc_spec = ("scipy.special.kv", 2, 1)
    
    @staticmethod
    def st_impl(p, x):
        return spp.kv(p, x)

    def impl(self, p, x):
        return KnuOp.st_impl(p, x)
    
    def grad(self, inputs, grads):
        dp = 1e-16
        (p, x) = inputs
        (gz,) = grads
        dfdp = (knuop(p + dp, x) - knuop(p - dp, x)) / (2*dp)
        gni = grad_not_implemented(self, 0, 'x', ' Grad with respect to x is not implemented for KvOp')
        # dfdx = spp.kvp(p, x) #TODO Implement this
        return [
            gz * dfdp,
            gni
        ]


knuop = KnuOp(upgrade_to_float, name='knuop')