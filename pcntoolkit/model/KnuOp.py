# Third-party imports
import scipy.special as spp
from pytensor.scalar.basic import BinaryScalarOp, upgrade_to_float


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
        dfdp = (knuop(p + dp, x) - knuop(p - dp, x)) / (2 * dp)
        return [gz * dfdp, gz * knupop(p, x)]


class KnuPrimeOp(BinaryScalarOp):
    """
    Derivative of the modified Bessel function of the second kind.
    """

    nfunc_spec = ("scipy.special.kvp", 2, 1)

    @staticmethod
    def st_impl(p, x):
        return spp.kvp(p, x)

    def impl(self, p, x):
        return KnuPrimeOp.st_impl(p, x)

    def grad(self, inputs, grads):
        dp = 1e-16
        (p, x) = inputs
        (gz,) = grads
        dfdp = (knupop(p + dp, x) - knupop(p - dp, x)) / (2 * dp)
        dfdx = -knuop(p, x) - knupop(p, x) / x
        return [gz * dfdp, gz * dfdx]


knuop = KnuOp(upgrade_to_float, name="knuop")
knupop = KnuPrimeOp(upgrade_to_float, name="knupop")
