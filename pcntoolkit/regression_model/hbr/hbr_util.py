"""Utility functions for Heteroscedastic Bayesian Regression (HBR) models.

This module provides mathematical utility functions for implementing various likelihood
models in HBR, particularly focusing on the SHASH (Sinh-Arcsinh) family of distributions
and normal distributions. It includes functions for computing z-scores, centiles,
and various transformations needed for these likelihood models.

The module implements three variants of the SHASH distribution:
- SHASHo: Original SHASH implementation
- SHASHo2: Modified SHASH with delta-scaled sigma
- SHASHb: SHASH with bias correction
As well as the standard Normal distribution.

Functions
---------
S_inv : Inverse sinh transformation
K : Modified Bessel function computation for unique values
P : P function implementation from Jones et al.
m : Uncentered moment calculation
centile : Centile computation for different likelihood models
zscore : Z-score computation for different likelihood models

References
----------
.. [1] Jones, M. C., & Pewsey, A. (2009). Sinh-arcsinh distributions.
       Biometrika, 96(4), 761-780.
"""

from typing import Literal

import numpy as np
import scipy.special as spp  # type: ignore
import scipy.stats
from numpy.typing import NDArray

from pcntoolkit.util.output import Errors, Output


def S(x: NDArray[np.float64], e: NDArray[np.float64], d: NDArray[np.float64]) -> NDArray[np.float64]:
    """Sinh transformation.

    Parameters
    ----------
    x : NDArray[np.float64]
        Input array to transform
    e : NDArray[np.float64]
        Epsilon parameter for the transformation
    d : NDArray[np.float64]
        Delta parameter for the transformation

    Returns
    -------
    NDArray[np.float64]
        Transformed array using sinh function
    """
    return np.sinh(np.arcsinh(x) * d - e)


def S_inv(x: NDArray[np.float64], e: NDArray[np.float64], d: NDArray[np.float64]) -> NDArray[np.float64]:
    """Inverse sinh arcsinh transformation.

    Parameters
    ----------
    x : NDArray[np.float64]
        Input array to transform
    e : NDArray[np.float64]
        Epsilon parameter for the transformation
    d : NDArray[np.float64]
        Delta parameter for the transformation

    Returns
    -------
    NDArray[np.float64]
        Transformed array using inverse sinh function
    """
    return np.sinh((np.arcsinh(x) + e) / d)


def K(p: NDArray[np.float64], x: float) -> NDArray[np.float64]:
    """Compute modified Bessel function of the second kind for unique values.

    Computes the values of spp.kv(p,x) for only the unique values of p to improve efficiency.

    Parameters
    ----------
    p : NDArray[np.float64]
        Array of values for which to compute the Bessel function
    x : float
        Second parameter of the Bessel function

    Returns
    -------
    NDArray[np.float64]
        Modified Bessel function values reshaped to match input shape
    """
    ps, idxs = np.unique(p, return_inverse=True)
    return spp.kv(ps, x)[idxs].reshape(p.shape)


def P(q: NDArray[np.float64]) -> NDArray[np.float64]:
    """Compute the P function as given in Jones et al.

    Parameters
    ----------
    q : NDArray[np.float64]
        Input array for P function calculation

    Returns
    -------
    NDArray[np.float64]
        Result of P function computation
    """
    frac = np.exp(1 / 4) / np.sqrt(8 * np.pi)
    K1 = K((q + 1) / 2, 1 / 4)
    K2 = K((q - 1) / 2, 1 / 4)
    a = (K1 + K2) * frac
    return a


def m(epsilon: NDArray[np.float64], delta: NDArray[np.float64], r: int) -> NDArray[np.float64]:
    """Calculate the r'th uncentered moment as given in Jones et al.

    Parameters
    ----------
    epsilon : NDArray[np.float64]
        Epsilon parameter array
    delta : NDArray[np.float64]
        Delta parameter array
    r : int
        Order of the moment to calculate

    Returns
    -------
    NDArray[np.float64]
        The r'th uncentered moment
    """
    frac1 = 1 / np.power(2, r)
    acc = 0
    for i in range(r + 1):
        combs = spp.comb(r, i)
        flip = np.power(-1, i)
        ex = np.exp((r - 2 * i) * epsilon / delta)
        p = P((r - 2 * i) / delta)
        acc += combs * flip * ex * p
    return frac1 * acc


def centile(
    likelihood: Literal["SHASHo", "SHASHo2", "SHASHb", "Normal", "beta"],
    *args,
    **kwargs,
) -> NDArray[np.float64]:
    """Compute centiles for different likelihood models.

    Parameters
    ----------
    likelihood : {"SHASHo", "SHASHo2", "SHASHb", "Normal"}
        The likelihood model to use
    *args, **kwargs
        Arguments and keyword arguments for the likelihood model

    Returns
    -------
    NDArray[np.float64]
        Computed quantiles
    """
    zs = kwargs.get("zs", 0)
    match likelihood:
        case "SHASHo":
            mu, sigma, epsilon, delta = args
            quantiles = S_inv(zs, epsilon, delta) * sigma + mu
        case "SHASHo2":
            mu, sigma, epsilon, delta = args
            sigma_d = sigma / delta
            quantiles = S_inv(zs, epsilon, delta) * sigma_d + mu
        case "SHASHb":
            mu, sigma, epsilon, delta = args
            true_mu = m(epsilon, delta, 1)
            true_sigma = np.sqrt((m(epsilon, delta, 2) - true_mu**2))
            SHASH_c = (S_inv(zs, epsilon, delta) - true_mu) / true_sigma
            quantiles = SHASH_c * sigma + mu
        case "Normal":
            mu, sigma = args
            quantiles = zs * sigma + mu
        case "beta":
            alpha, beta = args
            # First map the zs to the normal distribution
            cdf_norm = scipy.stats.norm.cdf(zs)
            # Then map the normal distribution to the beta distribution
            quantiles = scipy.stats.beta.ppf(cdf_norm, alpha, beta)
        case _:
            raise Output.error(Errors.ERROR_UNKNOWN_LIKELIHOOD, likelihood=likelihood)
    return quantiles


def zscore(
    likelihood: Literal["SHASHo", "SHASHo2", "SHASHb", "Normal", "beta"],
    *args,
    **kwargs,
) -> NDArray[np.float64]:
    """Compute z-scores for different likelihood models.

    Parameters
    ----------
    likelihood : {"SHASHo", "SHASHo2", "SHASHb", "Normal", "beta"}
        The likelihood model to use
    *args, **kwargs
        Arguments and keyword arguments for the likelihood model

    Returns
    -------
    NDArray[np.float64]
        Computed z-scores

    Raises
    ------
    ValueError
        If likelihood is not one of the supported types
    """
    y = kwargs.get("y", None)
    if y is None:
        raise Output.error(Errors.ERROR_HBR_Y_NOT_PROVIDED)
    match likelihood:
        case "SHASHo":
            mu, sigma, epsilon, delta = args
            SHASH = (y - mu) / sigma
            Z = np.sinh(np.arcsinh(SHASH) * delta - epsilon)
        case "SHASHo2":
            mu, sigma, epsilon, delta = args
            sigma_d = sigma / delta
            SHASH = (y - mu) / sigma_d
            Z = np.sinh(np.arcsinh(SHASH) * delta - epsilon)
        case "SHASHb":
            mu, sigma, epsilon, delta = args
            true_mu = m(epsilon, delta, 1)
            true_sigma = np.sqrt((m(epsilon, delta, 2) - true_mu**2))
            SHASH_c = (y - mu) / sigma
            SHASH = SHASH_c * true_sigma + true_mu
            Z = np.sinh(np.arcsinh(SHASH) * delta - epsilon)
        case "Normal":
            mu, sigma = args
            Z = (y - mu) / sigma
        case "beta":
            alpha, beta = args
            cdf = scipy.stats.beta.cdf(y, alpha, beta)
            Z = scipy.stats.norm.ppf(cdf)
        case _:
            raise Output.error(Errors.ERROR_UNKNOWN_LIKELIHOOD, likelihood=likelihood)
    return Z
