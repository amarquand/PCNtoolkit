import numpy as np
import scipy.special as spp


def S_inv(x, e, d):
    return np.sinh((np.arcsinh(x) + e) / d)


def K(p, x):
    """
    Computes the values of spp.kv(p,x) for only the unique values of p
    """

    ps, idxs = np.unique(p, return_inverse=True)
    return spp.kv(ps, x)[idxs].reshape(p.shape)


def P(q):
    """
    The P function as given in Jones et al.
    :param q:
    :return:
    """
    frac = np.exp(1 / 4) / np.sqrt(8 * np.pi)
    K1 = K((q + 1) / 2, 1 / 4)
    K2 = K((q - 1) / 2, 1 / 4)
    a = (K1 + K2) * frac
    return a


def m(epsilon, delta, r):
    """
    The r'th uncentered moment. Given by Jones et al.
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



def centile(likelihood, mu, sigma, epsilon=None, delta=None, zs=0):
    """Auxiliary function for computing centile"""
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
        exit("Unsupported likelihood")
    return quantiles

def zscore(likelihood, mu, sigma, epsilon=None, delta=None, y=None):
    """Auxiliary function for computing z-scores"""
    """Get the z-scores of Y, given likelihood parameters"""
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
        exit("Unsupported likelihood")
    return Z
