from __future__ import print_function

import numpy as np
from scipy import optimize
from numpy.linalg import solve, LinAlgError
from numpy.linalg import cholesky as chol

# -----------------
# Utility functions
# -----------------


def _sqDist(x, z=None):
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

# --------------------
# Covariance functions
# --------------------


def covLin(theta, x, z=None, i=None):

    if theta[0] is not None:
        print("hyperparameter specified but not required. Ignoring...")

    if z is None:
        z = x

    if i is None:
        K = x.dot(z.T)
        return K
    elif i == 0:
        return np.asarray(0)
    else:
        raise ValueError("Invalid covariance function parameter")


def covSqExp(theta, x, z=None, i=None):
    """ Ordinary squared exponential covariance function.
        The hyperparameters are:
            theta = ( log(ell), log(sf2) )
        where ell is a lengthscale parameter and sf2 is the signal variance
    """

    ell = np.exp(theta[0])
    sf2 = np.exp(2*theta[1])

    if z is None:
        z = x

    R = _sqDist(x/ell, z/ell)

    if i is None:  # return covariance
        K = sf2*np.exp(-R/2)
        return K
    elif i == 0:   # return derivative of lengthscale parameter
        dK = sf2*np.exp(-R/2)*R
        return dK
    elif i == 1:   # return derivative of signal variance parameter
        dK = 2*sf2*np.exp(-R/2)
        return dK
    else:
        raise ValueError("Invalid covariance function parameter")


def covSqExpARD(theta, x, z=None, i=None):
    """ Squared exponential covariance function with ARD
        The hyperparameters are:
            theta = ( log(ell_1, ..., log_ell_D), log(sf2) )
        where ell_i are lengthscale parameters and sf2 is the signal variance
    """

    D = x.shape[1]
    ell = np.exp(theta[0:D])
    sf2 = np.exp(2*theta[D])

    if z is None:
        z = x

    R = _sqDist(x.dot(np.diag(1./ell)), x.dot(np.diag(1./ell)))

    K = sf2*np.exp(-R/2)
    if i is None:  # return covariance
        return K
    elif i < D:    # return derivative of lengthscale parameter
        dK = K*_sqDist(x[:, i]/ell[i], z[:, i]/ell[i])
        return dK
    elif i == D:   # return derivative of signal variance parameter
        dK = 2*K
        return dK
    else:
        raise ValueError("Invalid covariance function parameter")


class GPR:
    """Gaussian process regression

    Estimation and prediction of Gaussian process regression models

    Basic usage::

        G = GPR()
        hyp = B.estimate(hyp0, cov, X, y)
        ys,ys2 = B.predict(hyp, cov, X, y, Xs)

    where the variables are

    :param hyp: vector of hyperparmaters.
    :param cov: covariance function
    :param X: N x D data array
    :param y: 1D Array of targets (length N)
    :param Xs: Nte x D array of test cases
    :param hyp0: starting estimates for hyperparameter optimisation

    :returns ys: predictive mean
    :returns ys2: predictive variance

    The hyperparameters are::

        hyp = ( log(sn2), (cov function params) )  # hyp is a list or array

    The implementation and notation  follows Rasmussen and Williams (2006).
    As in the gpml toolbox, these parameters are estimated using conjugate
    gradient optimisation of the marginal likelihood. Note that there is no
    explicit mean function, thus the gpr routines are limited to modelling
    zero-mean processes.

    Reference:
    C. Rasmussen and C. Williams (2006) Gaussian Processes for Machine Learning

    Written by A. Marquand
    """

    def __init__(self, hyp=None, cov=None, X=None, y=None, n_iter=1000,
                 tol=1e-3, verbose=False):

        self.hyp = np.nan
        self.nlZ = np.nan
        self.tol = tol          # not used at present
        self.n_iter = n_iter    # not used at present
        self.verbose = verbose

        if (hyp is not None) and (X is not None) and (y is not None):
            self.post(hyp, cov, X, y)

    def _updatepost(self, hyp, cov):

        hypeq = np.asarray(hyp == self.hyp)
        if hypeq.all() and hasattr(self, 'alpha') and \
           (hasattr(self, 'cov') and cov == self.cov):
            return False
        else:
            return True

    def post(self, hyp, cov, X, y):
        """ Generic function to compute posterior distribution.
        """

        if len(X.shape) == 1:
            X = X[:, np.newaxis]
        self.N, self.D = X.shape

        if not self._updatepost(hyp, cov):
            print("hyperparameters have not changed, using exising posterior")
            return

        # hyperparameters
        sn2 = np.exp(2*hyp[0])       # noise variance
        theta = hyp[1:]            # (generic) covariance hyperparameters

        if self.verbose:
            print("estimating posterior ... | hyp=", hyp)

        self.K = cov(theta, X)
        self.L = chol(self.K + sn2*np.eye(self.N))
        self.alpha = solve(self.L.T, solve(self.L, y))
        self.hyp = hyp
        self.cov = cov

    def loglik(self, hyp, cov, X, y):
        """ Function to compute compute log (marginal) likelihood """

        # load or recompute posterior
        if self._updatepost(hyp, cov):
            try:
                self.post(hyp, cov, X, y)
            except (ValueError, LinAlgError):
                print("Warning: Estimation of posterior distribution failed")
                self.nlZ = 1/np.finfo(float).eps
                return self.nlZ

        self.nlZ = 0.5*y.T.dot(self.alpha) + sum(np.log(np.diag(self.L))) + \
                   0.5*self.N*np.log(2*np.pi)

        # make sure the output is finite to stop the minimizer getting upset
        if not np.isfinite(self.nlZ):
            self.nlZ = 1/np.finfo(float).eps

        if self.verbose:
            print("nlZ= ", self.nlZ, " | hyp=", hyp)

        return self.nlZ

    def dloglik(self, hyp, cov, X, y):
        """ Function to compute derivatives """

        # hyperparameters
        sn2 = np.exp(2*hyp[0])       # noise variance
        theta = hyp[1:]            # (generic) covariance hyperparameters

        # load posterior and prior covariance
        if self._updatepost(hyp, cov):
            try:
                self.post(hyp, cov, X, y)
            except (ValueError, LinAlgError):
                print("Warning: Estimation of posterior distribution failed")
                dnlZ = np.sign(self.dnlZ) / np.finfo(float).eps
                return dnlZ

        # compute Q = alpha*alpha' - inv(K)
        Q = np.outer(self.alpha, self.alpha) - \
            solve(self.L.T, solve(self.L, np.eye(self.N)))

        # initialise derivatives
        self.dnlZ = np.zeros(len(hyp))

        # noise variance
        self.dnlZ[0] = -sn2*np.trace(Q)

        # covariance parameter(s)
        for par in range(0, len(theta)):
            # compute -0.5*trace(Q.dot(dK/d[theta_i])) efficiently
            self.dnlZ[par+1] = -0.5*np.sum(np.sum(Q*cov(theta, X, i=par).T))

        # make sure the gradient is finite to stop the minimizer getting upset
        if not all(np.isfinite(self.dnlZ)):
            bad = np.where(np.logical_not(np.isfinite(self.dnlZ)))
            for b in bad:
                self.dnlZ[b] = np.sign(self.dnlZ[b]) / np.finfo(float).eps

        if self.verbose:
            print("dnlZ= ", self.dnlZ, " | hyp=", hyp)

        return self.dnlZ

    # model estimation (optimization)
    def estimate(self, hyp0, cov, X, y, optimizer='cg'):
        """ Function to estimate the model """

        if optimizer.lower() == 'cg':  # conjugate gradients
            out = optimize.fmin_cg(self.loglik, hyp0, self.dloglik,
                                   (cov, X, y), disp=True, gtol=self.tol,
                                   maxiter=self.n_iter, full_output=1)

        elif optimizer.lower() == 'powell':  # Powell's method
            out = optimize.fmin_powell(self.loglik, hyp0, (cov, X, y),
                                       full_output=1)
        else:
            raise ValueError("unknown optimizer")

        self.hyp = out[0]
        self.nlZ = out[1]
        self.optimizer = optimizer

        return self.hyp

    def predict(self, hyp, X, y, Xs):
        """ Function to make predictions from the model """

        if self._updatepost(hyp, self.cov):
            self.post(hyp, self.cov, X, y)

        # hyperparameters
        sn2 = np.exp(2*hyp[0])     # noise variance
        theta = hyp[1:]            # (generic) covariance hyperparameters

        Ks = self.cov(theta, Xs, X)
        kss = self.cov(theta, Xs)

        # predictive mean
        ymu = Ks.dot(self.alpha)

        # predictive variance (for a noisy test input)
        v = solve(self.L.T, Ks.T)
        ys2 = kss - v.T.dot(self.alpha) + sn2

        return ymu, ys2
