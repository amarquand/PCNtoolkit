from __future__ import print_function

import numpy as np
from scipy import stats

# -----------------
# Utility functions
# -----------------


def squared_dist(x, z=None):
    """ compute sum((x-z) ** 2) for all vectors in a 2d array"""

    # do some basic checks
    if z is None:
        z = x
    if len(x.shape) == 1:
        x = x[:, np.newaxis]
    if len(z.shape) == 1:
        z = z[:, np.newaxis]

    nx, dx = x.shape
    nz, dz = z.shape
    if dx != dz:
        raise ValueError("""
                Cannot compute distance: vectors have different length""")

    # mean centre for numerical stability
    m = np.mean(np.vstack((np.mean(x, axis=0), np.mean(z, axis=0))), axis=0)
    x = x - m
    z = z - m

    xx = np.tile(np.sum((x*x), axis=1)[:, np.newaxis], (1, nz))
    zz = np.tile(np.sum((z*z), axis=1), (nx, 1))

    dist = (xx - 2*x.dot(z.T) + zz)

    return dist
    

def compute_pearsonr(A, B):
    """ Manually computes the Pearson correlation between two matrices.

        Basic usage::

        compute_pearsonr(A, B)

        where

        :param A: an N * M data array
        :param cov: an N * M array

        :returns Rho: N dimensional vector of correlation coefficients
        :returns ys2: N dimensional vector of p-values

        Notes::

        This function is useful when M is large and only the diagonal entries
        of the resulting correlation matrix are of interest. This function
        does not compute the full correlation matrix as an intermediate step"""

    N = A.shape[1]

    # first mean centre
    Am = A - np.mean(A, axis=0)
    Bm = B - np.mean(B, axis=0)
    # then normalize
    An = Am / np.sqrt(np.sum(Am**2, axis=0))
    Bn = Bm / np.sqrt(np.sum(Bm**2, axis=0))
    del(Am, Bm)

    Rho = np.sum(An * Bn, axis=0)
    del(An, Bn)

    # Fisher r-to-z
    Zr = (np.arctanh(Rho) - np.arctanh(0)) * np.sqrt(N - 3)
    N = stats.norm()
    pRho = 1-N.cdf(Zr)

    return Rho, pRho


class CustomCV:
    """ Custom cross-validation approach. This function does not do much, it
        merely provides a wrapper designed to be compatible with
        scikit-learn (e.g. sklearn.model_selection...)

        :param train: a list of indices of training splits (each itself a list)
        :param test: a list of indices of test splits (each itself a list)

        :returns tr: Indices for training set
        :returns te: Indices for test set """

    def __init__(self, train, test, X=None, y=None):
        self.train = train
        self.test = test
        self.n_splits = len(train)
        if X is not None:
            self.N = X.shape[0]
        else:
            self.N = None

    def split(self, X, y=None):
        if self.N is None:
            self.N = X.shape[0]

        for i in range(0, self.n_splits):
            tr = self.train[i]
            te = self.test[i]
            yield tr, te
