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
from numpy.typing import NDArray


def S(
    x: NDArray[np.float64], e: NDArray[np.float64], d: NDArray[np.float64]
) -> NDArray[np.float64]:
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


def S_inv(
    x: NDArray[np.float64], e: NDArray[np.float64], d: NDArray[np.float64]
) -> NDArray[np.float64]:
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


def m(
    epsilon: NDArray[np.float64], delta: NDArray[np.float64], r: int
) -> NDArray[np.float64]:
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
    likelihood: Literal["SHASHo", "SHASHo2", "SHASHb", "Normal"],
    mu: NDArray[np.float64],
    sigma: NDArray[np.float64],
    zs: NDArray[np.float64],
    epsilon: NDArray[np.float64] = None,  # type: ignore
    delta: NDArray[np.float64] = None,  # type: ignore
) -> NDArray[np.float64]:
    """Compute centiles for different likelihood models.

    Parameters
    ----------
    likelihood : {"SHASHo", "SHASHo2", "SHASHb", "Normal"}
        The likelihood model to use
    mu : NDArray[np.float64]
        Mean parameter array
    sigma : NDArray[np.float64]
        Standard deviation parameter array
    epsilon : NDArray[np.float64] or None, optional
        Epsilon parameter for SHASH models
    delta : NDArray[np.float64] or None, optional
        Delta parameter for SHASH models
    zs : NDArray[np.float64] or float, optional
        Z-scores for quantile computation, default 0

    Returns
    -------
    NDArray[np.float64]
        Computed quantiles
    """
    if zs is None:
        zs = 0

    if likelihood == "SHASHo":
        quantiles = S_inv(zs, epsilon, delta) * sigma + mu
    elif likelihood == "SHASHo2":
        sigma_d = sigma / delta
        quantiles = S_inv(zs, epsilon, delta) * sigma_d + mu
    elif likelihood == "SHASHb":
        true_mu = m(epsilon, delta, 1)
        true_sigma = np.sqrt((m(epsilon, delta, 2) - true_mu**2))
        SHASH_c = (S_inv(zs, epsilon, delta) - true_mu) / true_sigma
        quantiles = SHASH_c * sigma + mu
    elif likelihood == "Normal":
        quantiles = zs * sigma + mu
    else:
        raise ValueError("Unsupported likelihood")
    return quantiles


def zscore(
    likelihood: Literal["SHASHo", "SHASHo2", "SHASHb", "Normal"],
    mu: NDArray[np.float64],
    sigma: NDArray[np.float64],
    y: NDArray[np.float64],
    epsilon: NDArray[np.float64] = None,  # type: ignore
    delta: NDArray[np.float64] = None,  # type: ignore
) -> NDArray[np.float64]:
    """Compute z-scores for different likelihood models.

    Parameters
    ----------
    likelihood : {"SHASHo", "SHASHo2", "SHASHb", "Normal"}
        The likelihood model to use
    mu : NDArray[np.float64]
        Mean parameter array
    sigma : NDArray[np.float64]
        Standard deviation parameter array
    epsilon : NDArray[np.float64] or None, optional
        Epsilon parameter for SHASH models
    delta : NDArray[np.float64] or None, optional
        Delta parameter for SHASH models
    y : NDArray[np.float64] or None, optional
        Observed values for z-score computation

    Returns
    -------
    NDArray[np.float64]
        Computed z-scores

    Raises
    ------
    ValueError
        If likelihood is not one of the supported types
    """
    if likelihood == "SHASHo":
        SHASH = (y - mu) / sigma
        Z = np.sinh(np.arcsinh(SHASH) * delta - epsilon)
    elif likelihood == "SHASHo2":
        sigma_d = sigma / delta
        SHASH = (y - mu) / sigma_d
        Z = np.sinh(np.arcsinh(SHASH) * delta - epsilon)
    elif likelihood == "SHASHb":
        true_mu = m(epsilon, delta, 1)
        true_sigma = np.sqrt((m(epsilon, delta, 2) - true_mu**2))
        SHASH_c = (y - mu) / sigma
        SHASH = SHASH_c * true_sigma + true_mu
        Z = np.sinh(np.arcsinh(SHASH) * delta - epsilon)
    elif likelihood == "Normal":
        Z = (y - mu) / sigma
    else:
        raise ValueError("Unsupported likelihood")
    return Z
