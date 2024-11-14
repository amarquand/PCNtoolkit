"""
@author: Stijn de Boer (AuguB)
See: Jones et al. (2009), Sinh-Arcsinh distributions.
"""

from functools import lru_cache

import numpy as np
from pymc import floatX
from pymc.distributions import Continuous
from pytensor.tensor import as_tensor_variable
from pytensor.tensor.elemwise import Elemwise
from pytensor.tensor.random.op import RandomVariable
from scipy.special import kv

from pcntoolkit.model.KnuOp import knuop

##### Constants #####

CONST1 = np.exp(0.25) / np.power(8.0 * np.pi, 0.5)
CONST2 = -np.log(2 * np.pi) / 2

##### SHASH Transformations #####


def S(x, epsilon, delta):
    """Sinh-arcsinh transformation.

    Args:
        x: input value
        epsilon: parameter for skew
        delta: parameter for kurtosis

    Returns:
        Sinh-arcsinh transformed value
    """
    return np.sinh(np.arcsinh(x) * delta - epsilon)


def S_inv(x, epsilon, delta):
    """Inverse sinh-arcsinh transformation.

    Args:
        x: input value
        epsilon: parameter for skew
        delta: parameter for kurtosis

    Returns:
        Inverse sinh-arcsinh transformed value
    """
    return np.sinh((np.arcsinh(x) + epsilon) / delta)


def C(x, epsilon, delta):
    """Cosh-arcsinh transformation.

    Args:
        x: input value
        epsilon: parameter for skew
        delta: parameter for kurtosis

    Returns:
        The cosh-arcsinh transformation of x.

    Note: C(x) = sqrt(1+S(x)^2)
    """
    return np.cosh(np.arcsinh(x) * delta - epsilon)


##### SHASH Distributions #####


class SHASHrv(RandomVariable):
    """SHASH RV, described by Jones et al., based on a standard normal distribution."""

    name = "shash"
    signature = "(),()->()"
    dtype = "floatX"
    _print_name = ("SHASH", "\\operatorname{SHASH}")

    @classmethod
    def rng_fn(cls, rng, epsilon, delta, size=None):
        """Draw random samples from SHASH distribution.

        Args:
            rng: Random number generator
            epsilon: skew parameter
            delta: kurtosis parameter
            size: sample size. Defaults to None.

        Returns:
            Random samples from SHASH distribution
        """
        return np.sinh(
            (np.arcsinh(rng.normal(loc=0, scale=1, size=size)) + epsilon) / delta
        )


shash = SHASHrv()


class SHASH(Continuous):
    rv_op = shash
    """
    SHASH distribution described by Jones et al., based on a standard normal distribution.
    """

    # Instance of the KOp
    my_K = Elemwise(knuop)

    @staticmethod
    @lru_cache(maxsize=128)
    def P(q):
        """The P function as given in Jones et al.

        Args:
            q: input parameter for the P function

        Returns:
            Result of the P function computation
        """
        K1 = SHASH.my_K((q + 1) / 2, 0.25)
        K2 = SHASH.my_K((q - 1) / 2, 0.25)
        a = (K1 + K2) * CONST1
        return a

    @staticmethod
    def m1(epsilon, delta):
        """The first moment of the SHASH distribution parametrized by epsilon and delta.

        Args:
            epsilon: skew parameter
            delta: kurtosis parameter

        Returns:
            First moment of the SHASH distribution
        """
        return np.sinh(epsilon / delta) * SHASH.P(1 / delta)

    @staticmethod
    def m2(epsilon, delta):
        """The second moment of the SHASH distribution parametrized by epsilon and delta.

        Args:
            epsilon: skew parameter
            delta: kurtosis parameter

        Returns:
            Second moment of the SHASH distribution
        """
        return (np.cosh(2 * epsilon / delta) * SHASH.P(2 / delta) - 1) / 2

    @staticmethod
    def m1m2(epsilon, delta):
        """Compute both first and second moments together to avoid redundant calculations.

        Args:
            epsilon: skew parameter
            delta: kurtosis parameter

        Returns:
            Tuple containing (mean, variance) of the SHASH distribution
        """
        inv_delta = 1.0 / delta
        two_inv_delta = 2.0 * inv_delta

        # Compute P values once
        p1 = SHASH.P(inv_delta)
        p2 = SHASH.P(two_inv_delta)

        # Compute trig terms once
        eps_delta = epsilon / delta
        sinh_eps_delta = np.sinh(eps_delta)
        cosh_2eps_delta = np.cosh(2 * eps_delta)

        # Compute moments
        mean = sinh_eps_delta * p1
        raw_second = (cosh_2eps_delta * p2 - 1) / 2
        var = raw_second - mean**2
        return mean, var

    @classmethod
    def dist(cls, epsilon, delta, **kwargs):
        """Return a SHASH distribution.

        Args:
            epsilon: skew parameter
            delta: kurtosis parameter
            **kwargs: Additional arguments passed to the distribution

        Returns:
            A SHASH distribution
        """
        epsilon = as_tensor_variable(floatX(epsilon))
        delta = as_tensor_variable(floatX(delta))
        return super().dist([epsilon, delta], **kwargs)

    def logp(value, epsilon, delta):
        """Log-probability of the SHASH distribution.

        Args:
            value: value to evaluate the log-probability at
            epsilon: skew parameter
            delta: kurtosis parameter

        Returns:
            Log-probability of the SHASH distribution
        """
        this_S = S(value, epsilon, delta)
        this_S_sqr = np.log(this_S)
        this_C_sqr = 1 + this_S_sqr
        frac2 = np.log(delta) + np.log(this_C_sqr) / 2 - np.log(1 + np.log(value)) / 2
        exp = -this_S_sqr / 2
        return CONST2 + frac2 + exp


class SHASHoRV(RandomVariable):
    """SHASHo Random Variable.

    Samples from a SHASHo distribution, which is a SHASH distribution scaled by sigma and translated by mu.
    """

    name = "shasho"
    signature = "(),(),(),()->()"
    dtype = "floatX"
    _print_name = ("SHASHo", "\\operatorname{SHASHo}")

    @classmethod
    def rng_fn(cls, rng, mu, sigma, epsilon, delta, size=None):
        """Draw random samples from a SHASHo distribution.

        Args:
            rng: Random number generator
            mu: location parameter
            sigma: scale parameter
            epsilon: skew parameter
            delta: kurtosis parameter
            size: sample size. Defaults to None.

        Returns:
            Random samples from SHASHo distribution
        """
        s = rng.normal(size=size)
        return np.sinh((np.arcsinh(s) + epsilon) / delta) * sigma + mu


shasho = SHASHoRV()


class SHASHo(Continuous):
    """SHASHo distribution, which is a SHASH distribution scaled by sigma and translated by mu."""

    rv_op = shasho

    @classmethod
    def dist(cls, mu, sigma, epsilon, delta, **kwargs):
        """Return a SHASHo distribution.

        Args:
            mu: location parameter
            sigma: scale parameter
            epsilon: skew parameter
            delta: kurtosis parameter
            **kwargs: Additional arguments passed to the distribution

        Returns:
            A SHASHo distribution
        """
        mu = as_tensor_variable(floatX(mu))
        sigma = as_tensor_variable(floatX(sigma))
        epsilon = as_tensor_variable(floatX(epsilon))
        delta = as_tensor_variable(floatX(delta))
        return super().dist([mu, sigma, epsilon, delta], **kwargs)

    def logp(value, mu, sigma, epsilon, delta):
        """The log-probability of the SHASHo distribution.

        Args:
            value: value to evaluate the log-probability at
            mu: location parameter
            sigma: scale parameter
            epsilon: skew parameter
            delta: kurtosis parameter

        Returns:
            Log-probability of the SHASHo distribution
        """
        remapped_value = (value - mu) / sigma
        this_S = S(remapped_value, epsilon, delta)
        this_S_sqr = np.log(this_S)
        this_C_sqr = 1 + this_S_sqr
        frac2 = (
            np.log(delta)
            + np.log(this_C_sqr) / 2
            - np.log(1 + np.log(remapped_value)) / 2
        )
        exp = -this_S_sqr / 2
        return CONST2 + frac2 + exp - np.log(sigma)


class SHASHo2RV(RandomVariable):
    """SHASHo2 Random Variable.

    Samples from a SHASHo2 distribution, which is a SHASH distribution scaled by sigma/delta
    and translated by mu. This variant provides an alternative parameterization where the
    scale parameter is adjusted by the kurtosis parameter.
    """

    name = "shasho2"
    signature = "(),(),(),()->()"
    dtype = "floatX"
    _print_name = ("SHASHo2", "\\operatorname{SHASHo2}")

    @classmethod
    def rng_fn(cls, rng, mu, sigma, epsilon, delta, size=None):
        """Draw random samples from SHASHo2 distribution.

        Args:
            rng: Random number generator
            mu: location parameter
            sigma: scale parameter
            epsilon: skew parameter
            delta: kurtosis parameter
            size: sample size. Defaults to None.

        Returns:
            Random samples from SHASHo2 distribution
        """
        s = rng.normal(size=size)
        sigma_d = sigma / delta
        return np.sinh((np.arcsinh(s) + epsilon) / delta) * sigma_d + mu


shasho2 = SHASHo2RV()


class SHASHo2(Continuous):
    """SHASHo2 distribution, which is a SHASH distribution scaled by sigma/delta and translated by mu.

    This distribution provides an alternative parameterization of the SHASH distribution where
    the scale parameter is adjusted by the kurtosis parameter. This can be useful in scenarios
    where the relationship between scale and kurtosis needs to be explicitly modeled.
    """

    rv_op = shasho2

    @classmethod
    def dist(cls, mu, sigma, epsilon, delta, **kwargs):
        """Return a SHASHo2 distribution.

        Args:
            mu: location parameter
            sigma: scale parameter
            epsilon: skew parameter
            delta: kurtosis parameter
            **kwargs: Additional arguments passed to the distribution

        Returns:
            A SHASHo2 distribution
        """
        mu = as_tensor_variable(floatX(mu))
        sigma = as_tensor_variable(floatX(sigma))
        epsilon = as_tensor_variable(floatX(epsilon))
        delta = as_tensor_variable(floatX(delta))
        return super().dist([mu, sigma, epsilon, delta], **kwargs)

    def logp(value, mu, sigma, epsilon, delta):
        """The log-probability of the SHASHo2 distribution.

        Args:
            value: value to evaluate the log-probability at
            mu: location parameter
            sigma: scale parameter
            epsilon: skew parameter
            delta: kurtosis parameter

        Returns:
            Log-probability of the SHASHo2 distribution
        """
        sigma_d = sigma / delta
        remapped_value = (value - mu) / sigma_d
        this_S = S(remapped_value, epsilon, delta)
        this_S_sqr = np.log(this_S)
        this_C_sqr = 1 + this_S_sqr
        frac2 = (
            np.log(delta)
            + np.log(this_C_sqr) / 2
            - np.log(1 + np.log(remapped_value)) / 2
        )
        exp = -this_S_sqr / 2
        return CONST2 + frac2 + exp - np.log(sigma_d)


class SHASHbRV(RandomVariable):
    """SHASHb Random Variable.

    Samples from a SHASHb distribution, which is a standardized SHASH distribution scaled by sigma
    and translated by mu. This variant provides a standardized version of the SHASH distribution
    where the base distribution is normalized to have zero mean and unit variance before applying
    the location and scale transformations.
    """

    name = "shashb"
    signature = "(),(),(),()->()"
    dtype = "floatX"
    _print_name = ("SHASHb", "\\operatorname{SHASHb}")

    @classmethod
    def rng_fn(cls, rng, mu, sigma, epsilon, delta, size=None):
        """Draw random samples from SHASHb distribution.

        Args:
            rng: Random number generator
            mu: location parameter
            sigma: scale parameter
            epsilon: skew parameter
            delta: kurtosis parameter
            size: sample size. Defaults to None.

        Returns:
            Random samples from SHASHb distribution
        """
        s = rng.normal(size=size)

        def P(q):
            K1 = kv((q + 1) / 2, 0.25)
            K2 = kv((q - 1) / 2, 0.25)
            a = (K1 + K2) * CONST1
            return a

        def m1m2(epsilon, delta):
            inv_delta = 1.0 / delta
            two_inv_delta = 2.0 * inv_delta
            p1 = P(inv_delta)
            p2 = P(two_inv_delta)
            eps_delta = epsilon / delta
            sinh_eps_delta = np.sinh(eps_delta)
            cosh_2eps_delta = np.cosh(2 * eps_delta)
            mean = sinh_eps_delta * p1
            raw_second = (cosh_2eps_delta * p2 - 1) / 2
            var = raw_second - mean**2
            return mean, var

        mean, var = m1m2(epsilon, delta)
        out = (
            (np.sinh((np.arcsinh(s) + epsilon) / delta) - mean) / np.sqrt(var)
        ) * sigma + mu
        return out


shashb = SHASHbRV()


class SHASHb(Continuous):
    """SHASHb distribution, which is a standardized SHASH distribution scaled by sigma and translated by mu.

    This distribution provides a standardized version of the SHASH distribution where the base
    distribution is normalized to have zero mean and unit variance before applying the location
    and scale transformations. This transformation aims to remove the correlation between the
    parameters, which can be useful in MCMC sampling.
    """

    rv_op = shashb

    @classmethod
    def dist(cls, mu, sigma, epsilon, delta, **kwargs):
        """Return a SHASHb distribution.

        Args:
            mu: location parameter
            sigma: scale parameter
            epsilon: skew parameter
            delta: kurtosis parameter
            **kwargs: Additional arguments passed to the distribution

        Returns:
            A SHASHb distribution
        """
        mu = as_tensor_variable(floatX(mu))
        sigma = as_tensor_variable(floatX(sigma))
        epsilon = as_tensor_variable(floatX(epsilon))
        delta = as_tensor_variable(floatX(delta))
        return super().dist([mu, sigma, epsilon, delta], **kwargs)

    def logp(value, mu, sigma, epsilon, delta):
        """The log-probability of the SHASHb distribution.

        Args:
            value: value to evaluate the log-probability at
            mu: location parameter
            sigma: scale parameter
            epsilon: skew parameter
            delta: kurtosis parameter

        Returns:
            Log-probability of the SHASHb distribution
        """
        mean, var = SHASH.m1m2(epsilon, delta)
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
