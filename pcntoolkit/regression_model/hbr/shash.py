"""Sinh-Arcsinh (SHASH) Distribution Implementation Module.

This module implements the Sinh-Arcsinh (SHASH) distribution and its variants as described in
Jones and Pewsey (2009) [1]_. The SHASH distribution is a flexible distribution family that can
model skewness and kurtosis through separate parameters.

The module provides:

1. Basic SHASH transformations (S, S_inv, C)

2. SHASH distribution (base implementation)

3. SHASHo distribution (location-scale variant)

4. SHASHo2 distribution (alternative parameterization)

5. SHASHb distribution (standardized variant)


References
----------
.. [1] Jones, M. C., & Pewsey, A. (2009). Sinh-arcsinh distributions. Biometrika, 96(4), 761-780.
       https://doi.org/10.1093/biomet/asp053

Notes
-----
The implementation uses PyMC and PyTensor for probabilistic programming capabilities.
All distributions support random sampling and log-probability calculations.
"""

from functools import lru_cache
from typing import Any, Optional, Tuple, Union

import numpy as np
from numpy.random import Generator
from numpy.typing import ArrayLike, NDArray
from pymc import floatX  # type: ignore
from pymc.distributions import Continuous  # type: ignore
from pytensor import tensor as pt
from pytensor.graph.basic import Variable
from pytensor.tensor import as_tensor_variable  # type: ignore
from pytensor.tensor.elemwise import Elemwise
from pytensor.tensor.random.op import RandomVariable  # type: ignore
from scipy.special import kv  # type: ignore

from pcntoolkit.regression_model.hbr.KnuOp import knuop

# pylint: disable=arguments-differ


##### Constants #####

CONST1 = np.exp(0.25) / np.power(8.0 * np.pi, 0.5)
"""Constant used in P function calculations."""

CONST2 = -np.log(2 * np.pi) / 2
"""Constant used in log-probability calculations."""

##### SHASH Transformations #####


def S(x: ArrayLike, epsilon: float, delta: float) -> NDArray[np.float64]:
    """Apply the Sinh-arcsinh transformation.

    This transformation allows for flexible modeling of skewness and kurtosis.

    Parameters
    ----------
    x : array_like
        Input values to transform
    epsilon : float
        Skewness parameter. Positive values give positive skewness
    delta : float
        Kurtosis parameter. Values < 1 give heavier tails than normal

    Returns
    -------
    NDArray[np.float64]
        Transformed values

    Examples
    --------
    >>> S(0.0, epsilon=0.0, delta=1.0)
    0.0
    >>> S(1.0, epsilon=0.5, delta=2.0)  # Positive skew, lighter tails
    1.32460...
    """
    return np.sinh(np.arcsinh(x) * delta - epsilon)


def S_inv(x: ArrayLike, epsilon: float, delta: float) -> NDArray[np.float64]:
    """Apply the inverse sinh-arcsinh transformation.

    Parameters
    ----------
    x : array_like
        Input values to transform
    epsilon : float
        Skewness parameter controlling asymmetry
    delta : float
        Kurtosis parameter controlling tail weight

    Returns
    -------
    NDArray[np.float64]
        Inverse transformed values
    """
    return np.sinh((np.arcsinh(x) + epsilon) / delta)


def C(x: ArrayLike, epsilon: float, delta: float) -> NDArray[np.float64]:
    """Apply the cosh-arcsinh transformation.

    Parameters
    ----------
    x : array_like
        Input values to transform
    epsilon : float
        Skewness parameter controlling asymmetry
    delta : float
        Kurtosis parameter controlling tail weight

    Returns
    -------
    NDArray[np.float64]
        Transformed values
    """
    return np.cosh(np.arcsinh(x) * delta - epsilon)


##### SHASH Distributions #####


class SHASHrv(RandomVariable):
    """Random variable class for the base SHASH distribution.

    This class implements sampling from the basic SHASH distribution
    without location and scale parameters.

    Notes
    -----
    The base SHASH distribution is obtained by applying the sinh-arcsinh
    transformation to a standard normal distribution.
    """

    name = "shash"
    signature = "(),()->()"
    dtype = "floatX"
    _print_name = ("SHASH", "\\operatorname{SHASH}")

    @classmethod
    def rng_fn(
        cls,
        rng: Generator,
        epsilon: float,
        delta: float,
        size: Optional[Union[int, Tuple[int, ...]]] = None,
    ) -> NDArray[np.float64]:
        """Generate random samples from the base SHASH distribution.

        Parameters
        ----------
        rng : Generator
            NumPy random number generator
        epsilon : float
            Skewness parameter controlling asymmetry
        delta : float
            Kurtosis parameter controlling tail weight
        size : int or tuple of ints, optional
            Output shape. Default is None.

        Returns
        -------
        NDArray[np.float64]
            Array of random samples from the distribution
        """
        return np.sinh(
            (np.arcsinh(rng.normal(loc=0, scale=1, size=size)) + epsilon) / delta
        )


shash = SHASHrv()


class SHASH(Continuous):
    """Sinh-arcsinh distribution based on standard normal.

    A flexible distribution family that extends the normal distribution by adding
    skewness and kurtosis parameters while maintaining many desirable properties.

    Parameters
    ----------
    epsilon : float
        Skewness parameter controlling asymmetry
    delta : float
        Kurtosis parameter controlling tail weight
    """

    rv_op = shash
    my_K = Elemwise(knuop)

    @staticmethod
    @lru_cache(maxsize=128)
    def P(q: float) -> float:
        """The P function as given in Jones et al.

        Parameters
        ----------
        q : float
            Input parameter for the P function

        Returns
        -------
        float
            Result of the P function computation
        """
        K1 = SHASH.my_K((q + 1) / 2, 0.25)
        K2 = SHASH.my_K((q - 1) / 2, 0.25)
        a: Variable[Any, Any] = (K1 + K2) * CONST1  # type: ignore
        return a  # type: ignore

    @staticmethod
    def m1(epsilon: float, delta: float) -> float:
        """The first moment of the SHASH distribution parametrized by epsilon and delta.

        Parameters
        ----------
        epsilon : float
            Skewness parameter controlling asymmetry
        delta : float
            Kurtosis parameter controlling tail weight

        Returns
        -------
        float
            First moment of the SHASH distribution
        """
        return np.sinh(epsilon / delta) * SHASH.P(1 / delta)

    @staticmethod
    def m2(epsilon: float, delta: float) -> float:
        """The second moment of the SHASH distribution parametrized by epsilon and delta.

        Parameters
        ----------
        epsilon : float
            Skewness parameter controlling asymmetry
        delta : float
            Kurtosis parameter controlling tail weight

        Returns
        -------
        float
            Second moment of the SHASH distribution
        """
        return (np.cosh(2 * epsilon / delta) * SHASH.P(2 / delta) - 1) / 2

    @staticmethod
    def m1m2(epsilon: float, delta: float) -> Tuple[float, float]:
        """Compute both first and second moments of the SHASH distribution.

        This method efficiently calculates both moments together to avoid redundant
        computations of the P function.

        Parameters
        ----------
        epsilon : float
            Skewness parameter controlling distribution asymmetry
        delta : float
            Kurtosis parameter controlling tail weight

        Returns
        -------
        mean : float
            First moment (mean) of the distribution
        var : float
            Second central moment (variance) of the distribution

        Notes
        -----
        This method is more efficient than calling m1() and m2() separately
        as it reuses intermediate calculations.
        """
        inv_delta = 1.0 / delta
        two_inv_delta = 2.0 * inv_delta
        p1 = SHASH.P(inv_delta)
        p2 = SHASH.P(two_inv_delta)
        eps_delta = epsilon / delta
        sinh_eps_delta = np.sinh(eps_delta)
        cosh_2eps_delta = np.cosh(2 * eps_delta)
        mean = sinh_eps_delta * p1
        raw_second = (cosh_2eps_delta * p2 - 1) / 2
        var = raw_second - mean**2
        return mean, var

    @classmethod
    def dist(cls, epsilon: pt.TensorLike, delta: pt.TensorLike, **kwargs: Any) -> Any:
        """Create a SHASH distribution with given parameters.

        Parameters
        ----------
        epsilon : TensorLike
            Skewness parameter controlling distribution asymmetry
        delta : TensorLike
            Kurtosis parameter controlling tail weight
        **kwargs : dict
            Additional arguments passed to the distribution constructor

        Returns
        -------
        SHASH
            A SHASH distribution instance
        """
        epsilon = as_tensor_variable(floatX(epsilon))
        delta = as_tensor_variable(floatX(delta))
        return super().dist([epsilon, delta], **kwargs)

    def logp(value: ArrayLike, epsilon: float, delta: float) -> float:  # type: ignore
        """Calculate the log probability density of the SHASH distribution.

        Parameters
        ----------
        value : array_like
            Points at which to evaluate the log probability density
        epsilon : float
            Skewness parameter controlling distribution asymmetry
        delta : float
            Kurtosis parameter controlling tail weight

        Returns
        -------
        float
            Log probability density at the specified points
        """
        this_S = S(value, epsilon, delta)
        this_S_sqr = np.square(this_S)
        this_C_sqr = 1 + this_S_sqr
        frac2 = (
            np.log(delta) + np.log(this_C_sqr) / 2 - np.log(1 + np.square(value)) / 2
        )
        exp = -this_S_sqr / 2
        return CONST2 + frac2 + exp


class SHASHoRV(RandomVariable):
    """Random variable class for the location-scale SHASH distribution.

    This class implements sampling from a SHASH distribution that has been
    transformed to include location (mu) and scale (sigma) parameters.

    Notes
    -----
    The transformation is y = sigma * x + mu, where x follows the base SHASH
    distribution with parameters epsilon and delta.
    """

    name = "shasho"
    signature = "(),(),(),()->()"
    dtype = "floatX"
    _print_name = ("SHASHo", "\\operatorname{SHASHo}")

    @classmethod
    def rng_fn(
        cls,
        rng: Generator,
        mu: pt.TensorLike,
        sigma: pt.TensorLike,
        epsilon: pt.TensorLike,
        delta: pt.TensorLike,
        size: Optional[Union[int, Tuple[int, ...]]] = None,
    ) -> NDArray[np.float64]:
        """Generate random samples from the location-scale SHASH distribution.

        Parameters
        ----------
        rng : Generator
            NumPy random number generator
        mu : TensorLike
            Location parameter (mean)
        sigma : TensorLike
            Scale parameter (standard deviation)
        epsilon : TensorLike
            Skewness parameter
        delta : TensorLike
            Kurtosis parameter
        size : int or tuple of ints, optional
            Output shape. Default is None.

        Returns
        -------
        NDArray[np.float64]
            Array of random samples from the distribution
        """
        s = rng.normal(size=size)
        return np.sinh((np.arcsinh(s) + epsilon) / delta) * sigma + mu  # type: ignore


shasho = SHASHoRV()


class SHASHo(Continuous):
    """Location-scale variant of the SHASH distribution.

    This distribution extends the base SHASH distribution by adding
    location (mu) and scale (sigma) parameters.

    Parameters
    ----------
    mu : float
        Location parameter (mean)
    sigma : float
        Scale parameter (standard deviation)
    epsilon : float
        Skewness parameter controlling asymmetry
    delta : float
        Kurtosis parameter controlling tail weight

    Notes
    -----
    The distribution is obtained by applying the transformation
    Y = mu + sigma * X where X follows the base SHASH distribution.
    """

    rv_op = shasho

    @classmethod
    def dist(
        cls,
        mu: pt.TensorLike,
        sigma: pt.TensorLike,
        epsilon: pt.TensorLike,
        delta: pt.TensorLike,
        **kwargs: Any,
    ) -> Any:
        """Create a SHASHo distribution with given parameters.

        Parameters
        ----------
        mu : TensorLike
            Location parameter (mean)
        sigma : TensorLike
            Scale parameter (standard deviation)
        epsilon : TensorLike
            Skewness parameter controlling asymmetry
        delta : TensorLike
            Kurtosis parameter controlling tail weight
        **kwargs : dict
            Additional arguments passed to the distribution constructor

        Returns
        -------
        SHASHo
            A location-scale SHASH distribution instance
        """
        mu = as_tensor_variable(floatX(mu))
        sigma = as_tensor_variable(floatX(sigma))
        epsilon = as_tensor_variable(floatX(epsilon))
        delta = as_tensor_variable(floatX(delta))
        return super().dist([mu, sigma, epsilon, delta], **kwargs)

    def logp(
        value: ArrayLike, mu: float, sigma: float, epsilon: float, delta: float  # type: ignore
    ) -> float: 
        """Calculate the log probability density of the SHASHo distribution.

        Parameters
        ----------
        value : array_like
            Points at which to evaluate the log probability density
        mu : float
            Location parameter (mean)
        sigma : float
            Scale parameter (standard deviation)
        epsilon : float
            Skewness parameter controlling asymmetry
        delta : float
            Kurtosis parameter controlling tail weight

        Returns
        -------
        float
            Log probability density at the specified points
        """
        remapped_value = (value - mu) / sigma  # type: ignore
        this_S = S(remapped_value, epsilon, delta)
        this_S_sqr = np.square(this_S)
        this_C_sqr = 1 + this_S_sqr
        frac2 = (
            np.log(delta)
            + np.log(this_C_sqr) / 2
            - np.log(1 + np.square(remapped_value)) / 2
        )
        exp = -this_S_sqr / 2
        return CONST2 + frac2 + exp - np.log(sigma)


class SHASHo2RV(RandomVariable):
    """Random variable class for the alternative parameterization of SHASH distribution.

    This class implements sampling from a SHASH distribution where the scale parameter
    is adjusted by the kurtosis parameter (sigma/delta).

    Notes
    -----
    The transformation is y = (sigma/delta) * x + mu, where x follows the base SHASH
    distribution with parameters epsilon and delta.
    """

    name = "shasho2"
    signature = "(),(),(),()->()"
    dtype = "floatX"
    _print_name = ("SHASHo2", "\\operatorname{SHASHo2}")

    @classmethod
    def rng_fn(
        cls,
        rng: Generator,
        mu: pt.TensorLike,
        sigma: pt.TensorLike,
        epsilon: pt.TensorLike,
        delta: pt.TensorLike,
        size: Optional[Union[int, Tuple[int, ...]]] = None,
    ) -> NDArray[np.float64]:
        """Generate random samples from the alternative parameterization SHASH distribution.

        Parameters
        ----------
        rng : Generator
            NumPy random number generator
        mu : TensorLike
            Location parameter (mean)
        sigma : TensorLike
            Scale parameter (before delta adjustment)
        epsilon : TensorLike
            Skewness parameter
        delta : TensorLike
            Kurtosis parameter
        size : int or tuple of ints, optional
            Output shape. Default is None.

        Returns
        -------
        NDArray[np.float64]
            Array of random samples from the distribution

        """
        s = rng.normal(size=size)
        sigma_d = sigma / delta  # type: ignore
        return np.sinh((np.arcsinh(s) + epsilon) / delta) * sigma_d + mu  # type: ignore


shasho2 = SHASHo2RV()


class SHASHo2(Continuous):
    """Alternative parameterization of the SHASH distribution.

    This distribution extends the base SHASH distribution by adding location (mu)
    and an adjusted scale parameter (sigma/delta).

    Parameters
    ----------
    mu : float
        Location parameter (mean)
    sigma : float
        Scale parameter (before delta adjustment)
    epsilon : float
        Skewness parameter controlling asymmetry
    delta : float
        Kurtosis parameter controlling tail weight

    Notes
    -----
    The distribution is obtained by applying the transformation
    Y = mu + (sigma/delta) * X where X follows the base SHASH distribution.
    """

    rv_op = shasho2

    @classmethod
    def dist(
        cls,
        mu: pt.TensorLike,
        sigma: pt.TensorLike,
        epsilon: pt.TensorLike,
        delta: pt.TensorLike,
        **kwargs: Any,
    ) -> Any:
        """Create a SHASHo2 distribution with given parameters.

        Parameters
        ----------
        mu : TensorLike
            Location parameter (mean)
        sigma : TensorLike
            Scale parameter (before delta adjustment)
        epsilon : TensorLike
            Skewness parameter controlling asymmetry
        delta : TensorLike
            Kurtosis parameter controlling tail weight
        **kwargs : dict
            Additional arguments passed to the distribution constructor

        Returns
        -------
        SHASHo2
            An alternative parameterization SHASH distribution instance
        """
        mu = as_tensor_variable(floatX(mu))
        sigma = as_tensor_variable(floatX(sigma))
        epsilon = as_tensor_variable(floatX(epsilon))
        delta = as_tensor_variable(floatX(delta))
        return super().dist([mu, sigma, epsilon, delta], **kwargs)

    def logp(
        value: ArrayLike, mu: float, sigma: float, epsilon: float, delta: float  # type: ignore
    ) -> float:
        """Calculate the log probability density of the SHASHo2 distribution.

        Parameters
        ----------
        value : array_like
            Points at which to evaluate the log probability density
        mu : float
            Location parameter (mean)
        sigma : float
            Scale parameter (before delta adjustment)
        epsilon : float
            Skewness parameter controlling asymmetry
        delta : float
            Kurtosis parameter controlling tail weight

        Returns
        -------
        float
            Log probability density at the specified points

        Notes
        -----
        The implementation follows Jones et al. (2009) equation (2.2)
        with additional location-scale transformation where scale is
        adjusted by the kurtosis parameter.
        """
        sigma_d = sigma / delta
        remapped_value = (value - mu) / sigma_d  # type: ignore
        this_S = S(remapped_value, epsilon, delta)
        this_S_sqr = np.square(this_S)
        this_C_sqr = 1 + this_S_sqr
        frac2 = (
            np.log(delta)
            + np.log(this_C_sqr) / 2
            - np.log(1 + np.square(remapped_value)) / 2
        )
        exp = -this_S_sqr / 2
        return CONST2 + frac2 + exp - np.log(sigma_d)


class SHASHbRV(RandomVariable):
    """Random variable class for the standardized SHASH distribution.

    This class implements sampling from a SHASH distribution that has been
    standardized to have zero mean and unit variance before applying
    location and scale transformations.
    """

    name = "shashb"
    signature = "(),(),(),()->()"
    dtype = "floatX"
    _print_name = ("SHASHb", "\\operatorname{SHASHb}")

    @classmethod
    def rng_fn(
        cls,
        rng: Generator,
        mu: float,
        sigma: float,
        epsilon: float,
        delta: float,
        size: Optional[Union[int, Tuple[int, ...]]] = None,
    ) -> NDArray[np.float64]:
        """Generate random samples from standardized SHASH distribution.

        Parameters
        ----------
        rng : Generator
            NumPy random number generator
        mu : float
            Location parameter (mean)
        sigma : float
            Scale parameter (standard deviation)
        epsilon : float
            Skewness parameter
        delta : float
            Kurtosis parameter
        size : int or tuple of ints, optional
            Output shape. Default is None.

        Returns
        -------
        NDArray[np.float64]
            Array of random samples from the distribution
        """
        s = rng.normal(size=size)

        def P(q: float) -> float:
            """Helper function to compute P function.

            Parameters
            ----------
            q : float
                Input parameter

            Returns
            -------
            float
                P function value
            """
            K1 = kv((q + 1) / 2, 0.25)
            K2 = kv((q - 1) / 2, 0.25)
            a = (K1 + K2) * CONST1
            return a

        def m1m2(epsilon: float, delta: float) -> Tuple[float, float]:
            """Helper function to compute moments.

            Parameters
            ----------
            epsilon : float
                Skewness parameter
            delta : float
                Kurtosis parameter

            Returns
            -------
            Tuple[float, float]
                Mean and variance
            """
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
        ) * sigma + mu  # type: ignore
        return out


shashb = SHASHbRV()


class SHASHb(Continuous):
    """Standardized variant of the SHASH distribution.

    This distribution extends the base SHASH distribution by standardizing it
    to have zero mean and unit variance before applying location and scale
    transformations.

    Parameters
    ----------
    mu : float
        Location parameter (mean)
    sigma : float
        Scale parameter (standard deviation)
    epsilon : float
        Skewness parameter controlling asymmetry
    delta : float
        Kurtosis parameter controlling tail weight

    Notes
    -----
    The distribution is obtained by:
    1. Starting with base SHASH distribution
    2. Standardizing to zero mean and unit variance
    3. Applying Y = mu + sigma * X transformation

    This standardization can improve numerical stability and parameter
    interpretability in some applications.
    """

    rv_op = shashb

    @classmethod
    def dist(
        cls,
        mu: pt.TensorLike,
        sigma: pt.TensorLike,
        epsilon: pt.TensorLike,
        delta: pt.TensorLike,
        **kwargs: Any,
    ) -> Any:
        """Create a SHASHb distribution with given parameters.

        Parameters
        ----------
        mu : TensorLike
            Location parameter (mean)
        sigma : TensorLike
            Scale parameter (standard deviation)
        epsilon : TensorLike
            Skewness parameter controlling asymmetry
        delta : TensorLike
            Kurtosis parameter controlling tail weight
        **kwargs : dict
            Additional arguments passed to the distribution constructor

        Returns
        -------
        SHASHb
            A standardized SHASH distribution instance
        """
        mu = as_tensor_variable(floatX(mu))
        sigma = as_tensor_variable(floatX(sigma))
        epsilon = as_tensor_variable(floatX(epsilon))
        delta = as_tensor_variable(floatX(delta))
        return super().dist([mu, sigma, epsilon, delta], **kwargs)

    def logp(
        value: ArrayLike, mu: float, sigma: float, epsilon: float, delta: float  # type: ignore
    ) -> float:
        """Calculate the log probability density of the SHASHb distribution.

        Parameters
        ----------
        value : array_like
            Points at which to evaluate the log probability density
        mu : float
            Location parameter (mean)
        sigma : float
            Scale parameter (standard deviation)
        epsilon : float
            Skewness parameter controlling asymmetry
        delta : float
            Kurtosis parameter controlling tail weight

        Returns
        -------
        float
            Log probability density at the specified points
        """
        mean, var = SHASH.m1m2(epsilon, delta)
        remapped_value = ((value - mu) / sigma) * np.sqrt(var) + mean  # type: ignore
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
