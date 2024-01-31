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
