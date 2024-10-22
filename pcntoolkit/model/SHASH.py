"""
@author: Stijn de Boer (AuguB)
See: Jones et al. (2009), Sinh-Arcsinh distributions.
"""

# Standard library imports
from typing import Union, List, Optional

# Third-party imports
import pymc as pm
from pymc import floatX
from pymc.distributions import Continuous

import pytensor as pt
import pytensor.tensor as ptt
from pytensor.graph.op import Op
from pytensor.graph import Apply
from pytensor.gradient import grad_not_implemented
from pytensor.tensor.random.basic import normal
from pytensor.tensor.random.op import RandomVariable

import numpy as np
import scipy.special as spp
import matplotlib.pyplot as plt

CONST1 = np.exp(0.25) / np.power(8.0 * np.pi, 0.5)
CONST2 = -np.log(2 * np.pi) / 2

class KOp(Op):
    """
    Modified Bessel function of the second kind, pytensor wrapper for scipy.special.kv
    """
    __props__ = ()

    def make_node(self, p, x):
        p = pt.tensor.as_tensor_variable(p)
        x = pt.tensor.as_tensor_variable(x)
        return Apply(self, [p, x], [p.type()])

    def perform(self, node, inputs_storage, output_storage):
        output_storage[0][0] = spp.kv(inputs_storage[0], inputs_storage[1])

    def grad(self, inputs, output_grads):
        # Approximation of the derivative. This should suffice for using NUTS
        dp = 1e-16
        p = inputs[0]
        x = inputs[1]
        grad = (self(p + dp, x) - self(p - dp, x)) / (2*dp)
        return [
            output_grads[0] * grad,
            grad_not_implemented("KOp", 2, "x", "")

        ]

def S(x, epsilon, delta):
    """
    :param epsilon:
    :param delta:
    :param x:
    :return: The sinharcsinh transformation of x
    """
    return np.sinh(np.arcsinh(x) * delta - epsilon)


def S_inv(x, epsilon, delta):
    return np.sinh((np.arcsinh(x) + epsilon) / delta)


def C(x, epsilon, delta):
    """
    :param epsilon:
    :param delta:
    :param x:
    :return: the cosharcsinh transformation of x
    Be aware that this is sqrt(1+S(x)^2), so you may save some compute if you can re-use the result from S.
    """
    return np.cosh(np.arcsinh(x) * delta - epsilon)


def m(epsilon, delta, r):
    """
    :param epsilon:
    :param delta:
    :param r:
    :return:  The r'th uncentered moment of the SHASH distribution parameterized by epsilon and delta. Given by Jones et al.
    The first four moments are given in closed form.
    """
    if r == 1:
        return np.sinh(epsilon / delta) * P(1 / delta)
    elif r == 2:
        return (np.cosh(2 * epsilon / delta) * P(2 / delta) - 1) / 2
    elif r == 3:
        return (
            np.sinh(3 * epsilon / delta) * P(3 / delta)
            - 3 * np.sinh(epsilon / delta) * P(1 / delta)
        ) / 4
    elif r == 4:
        return (
            np.cosh(4 * epsilon / delta) * P(4 / delta)
            - 4 * np.cosh(2 * epsilon / delta) * P(2 / delta)
            + 3
        ) / 8

def m1(epsilon, delta):
    return np.sinh(epsilon / delta) * P(1 / delta)

def m2(epsilon, delta):
    return (np.cosh(2 * epsilon / delta) * P(2 / delta) - 1) / 2

def m3(epsilon, delta):
    return (
        np.sinh(3 * epsilon / delta) * P(3 / delta)
        - 3 * np.sinh(epsilon / delta) * P(1 / delta)
    ) / 4

def numpy_P(q):
    """
    The P function as given in Jones et al.
    :param q:
    :return:
    """
    frac = CONST1
    K1 = spp.kv((q + 1) / 2, 0.25)
    K2 = spp.kv((q - 1) / 2, 0.25)
    a = (K1 + K2) * frac
    return a

# Instance of the KOp
my_K = KOp()

def P(q):
    """
    The P function as given in Jones et al.
    :param q:
    :return:
    """
    K1 = my_K((q + 1) / 2, 0.25)
    K2 = my_K((q - 1) / 2, 0.25)
    a = (K1 + K2) * CONST1
    return a


##### SHASH Distributions #####

class SHASH(RandomVariable):
    name = "shash"
    signature = "(),()->()"
    dtype = "floatX"
    _print_name = ("SHASH", "\\operatorname{SHASH}")

    @classmethod
    def rng_fn(cls, rng, epsilon, delta, size=None) -> np.ndarray:
        return np.sinh(
            (np.arcsinh(rng.normal(loc=0, scale=1, size=size)) + epsilon) / delta
        )


shash = SHASH()


class SHASH(Continuous):
    rv_op = shash
    """
    SHASH described by Jones et al., based on a standard normal distribution.
    """

    @classmethod
    def dist(cls, epsilon, delta, **kwargs):
        epsilon = ptt.as_tensor_variable(floatX(epsilon))
        delta = ptt.as_tensor_variable(floatX(delta))
        return super().dist([epsilon, delta], **kwargs)

    def logp(value, epsilon, delta):
        this_S = S(value, epsilon, delta)
        this_S_sqr = ptt.sqr(this_S)
        this_C_sqr = 1 + this_S_sqr
        frac2 = (
            ptt.log(delta) + ptt.log(this_C_sqr) /
            2 - ptt.log(1 + ptt.sqr(value)) / 2
        )
        exp = -this_S_sqr / 2
        return CONST2 + frac2 + exp


class SHASHoRV(RandomVariable):
    name = "shasho"
    signature = "(),(),(),()->()"
    dtype = "floatX"
    _print_name = ("SHASHo", "\\operatorname{SHASHo}")

    @classmethod
    def rng_fn(cls, rng, mu, sigma, epsilon, delta, size=None) -> np.ndarray:
        s = rng.normal(size=size)
        return np.sinh((np.arcsinh(s) + epsilon) / delta) * sigma + mu


shasho = SHASHoRV()


class SHASHo(Continuous):
    rv_op = shasho
    """
    This is the transformation where the location and scale parameters have simply been applied as an linear transformation directly on the original distribution.
    """

    @classmethod
    def dist(cls, mu, sigma, epsilon, delta, **kwargs):
        mu = ptt.as_tensor_variable(floatX(mu))
        sigma = ptt.as_tensor_variable(floatX(sigma))
        epsilon = ptt.as_tensor_variable(floatX(epsilon))
        delta = ptt.as_tensor_variable(floatX(delta))
        return super().dist([mu, sigma, epsilon, delta], **kwargs)

    def logp(value, mu, sigma, epsilon, delta):
        remapped_value = (value - mu) / sigma
        this_S = S(remapped_value, epsilon, delta)
        this_S_sqr = ptt.sqr(this_S)
        this_C_sqr = 1 + this_S_sqr
        frac2 = (
            ptt.log(delta)
            + ptt.log(this_C_sqr) / 2
            - ptt.log(1 + ptt.sqr(remapped_value)) / 2
        )
        exp = -this_S_sqr / 2
        return CONST2 + frac2 + exp - ptt.log(sigma)


class SHASHo2RV(RandomVariable):
    name = "shasho2"
    signature = "(),(),(),()->()"
    dtype = "floatX"
    _print_name = ("SHASHo2", "\\operatorname{SHASHo2}")

    @classmethod
    def rng_fn(cls, rng, mu, sigma, epsilon, delta, size=None) -> np.ndarray:
        s = rng.normal(size=size)
        sigma_d = sigma / delta
        return np.sinh((np.arcsinh(s) + epsilon) / delta) * sigma_d + mu


shasho2 = SHASHo2RV()


class SHASHo2(Continuous):
    rv_op = shasho2
    """
    This is the reparameterization where we apply the transformation provided in section 4.3 in Jones et al.
    """

    @classmethod
    def dist(cls, mu, sigma, epsilon, delta, **kwargs):
        mu = ptt.as_tensor_variable(floatX(mu))
        sigma = ptt.as_tensor_variable(floatX(sigma))
        epsilon = ptt.as_tensor_variable(floatX(epsilon))
        delta = ptt.as_tensor_variable(floatX(delta))
        return super().dist([mu, sigma, epsilon, delta], **kwargs)

    def logp(value, mu, sigma, epsilon, delta):
        sigma_d = sigma / delta
        remapped_value = (value - mu) / sigma_d
        this_S = S(remapped_value, epsilon, delta)
        this_S_sqr = ptt.sqr(this_S)
        this_C_sqr = 1 + this_S_sqr
        frac2 = (
            ptt.log(delta)
            + ptt.log(this_C_sqr) / 2
            - ptt.log(1 + ptt.sqr(remapped_value)) / 2
        )
        exp = -this_S_sqr / 2
        return CONST2 + frac2 + exp - ptt.log(sigma_d)




class SHASHbRV(RandomVariable):
    name = "shashb"
    signature = "(),(),(),()->()"
    dtype = "floatX"
    _print_name = ("SHASHb", "\\operatorname{SHASHb}")

    @classmethod
    def rng_fn(
        cls,
        rng: np.random.RandomState,
        mu: Union[np.ndarray, float],
        sigma: Union[np.ndarray, float],
        epsilon: Union[np.ndarray, float],
        delta: Union[np.ndarray, float],
        size: Optional[Union[List[int], int]],
    ) -> np.ndarray:
        s = rng.normal(size=size)
        mean = np.sinh(epsilon / delta) * numpy_P(1 / delta)
        var = ((np.cosh(2 * epsilon / delta) * numpy_P(2 / delta) - 1) / 2) - mean**2
        out = (
            (np.sinh((np.arcsinh(s) + epsilon) / delta) - mean) / np.sqrt(var)
        ) * sigma + mu
        return out


shashb = SHASHbRV()


class SHASHb(Continuous):
    rv_op = shashb
    """
    This is the reparameterization where the location and scale parameters been applied as an linear transformation on the shash distribution which was corrected for mean and variance.
    """

    @classmethod
    def dist(cls, mu, sigma, epsilon, delta, **kwargs):
        mu = ptt.as_tensor_variable(floatX(mu))
        sigma = ptt.as_tensor_variable(floatX(sigma))
        epsilon = ptt.as_tensor_variable(floatX(epsilon))
        delta = ptt.as_tensor_variable(floatX(delta))
        return super().dist([mu, sigma, epsilon, delta], **kwargs)

    def logp(value, mu, sigma, epsilon, delta):
        mean = m1(epsilon, delta)
        var = m2(epsilon, delta) - mean**2
        remapped_value = ((value - mu) / sigma) * np.sqrt(var) + mean
        this_S = S(remapped_value, epsilon, delta)
        this_S_sqr = np.square(this_S)
        this_C_sqr = 1 + this_S_sqr
        frac2 = (
            np.log(delta)
            + np.log(this_C_sqr) / 2
            - np.log(1 + np.square(remapped_value)) / 2
        )
        exp = -this_S_sqr / 2
        return CONST2 + frac2 + exp + np.log(var) / 2 - np.log(sigma)
