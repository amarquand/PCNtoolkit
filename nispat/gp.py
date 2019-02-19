from __future__ import print_function
from __future__ import division

import os
import sys
import numpy as np
from scipy import optimize
from numpy.linalg import solve, LinAlgError
from numpy.linalg import cholesky as chol
from six import with_metaclass
from abc import ABCMeta, abstractmethod


try:  # Run as a package if installed    
    from nispat.utils import squared_dist
except ImportError:
    pass

    path = os.path.abspath(os.path.dirname(__file__))
    if path not in sys.path:
        sys.path.append(path)
    del path
    
    from utils import squared_dist

# --------------------
# Covariance functions
# --------------------


class CovBase(with_metaclass(ABCMeta)):
    """ Base class for covariance functions.

        All covariance functions must define the following methods::

            CovFunction.get_n_params()
            CovFunction.cov()
            CovFunction.xcov()
            CovFunction.dcov()
    """

    def __init__(self, x=None):
        self.n_params = np.nan

    def get_n_params(self):
        """ Report the number of parameters required """

        assert not np.isnan(self.n_params), \
            "Covariance function not initialised"

        return self.n_params

    @abstractmethod
    def cov(self, theta, x, z=None):
        """ Return the full covariance (or cross-covariance if z is given) """

    @abstractmethod
    def dcov(self, theta, x, i):
        """ Return the derivative of the covariance function with respect to
            the i-th hyperparameter """


class CovLin(CovBase):
    """ Linear covariance function (no hyperparameters)
    """

    def __init__(self, x=None):
        self.n_params = 0
        self.first_call = False

    def cov(self, theta, x, z=None):
        if not self.first_call and not theta and theta is not None:
            self.first_call = True
            if len(theta) > 0 and theta[0] is not None:
                print("CovLin: ignoring unnecessary hyperparameter ...")

        if z is None:
            z = x

        K = x.dot(z.T)
        return K

    def dcov(self, theta, x, i):
        raise ValueError("Invalid covariance function parameter")


class CovSqExp(CovBase):
    """ Ordinary squared exponential covariance function.
        The hyperparameters are::

            theta = ( log(ell), log(sf) )

        where ell is a lengthscale parameter and sf2 is the signal variance
    """

    def __init__(self, x=None):
        self.n_params = 2

    def cov(self, theta, x, z=None):
        self.ell = np.exp(theta[0])
        self.sf2 = np.exp(2*theta[1])

        if z is None:
            z = x

        R = squared_dist(x/self.ell, z/self.ell)
        K = self.sf2 * np.exp(-R/2)
        return K

    def dcov(self, theta, x, i):
        self.ell = np.exp(theta[0])
        self.sf2 = np.exp(2*theta[1])

        R = squared_dist(x/self.ell, x/self.ell)

        if i == 0:   # return derivative of lengthscale parameter
            dK = self.sf2 * np.exp(-R/2) * R
            return dK
        elif i == 1:   # return derivative of signal variance parameter
            dK = 2*self.sf2 * np.exp(-R/2)
            return dK
        else:
            raise ValueError("Invalid covariance function parameter")


class CovSqExpARD(CovBase):
    """ Squared exponential covariance function with ARD
        The hyperparameters are::

            theta = (log(ell_1, ..., log_ell_D), log(sf))

        where ell_i are lengthscale parameters and sf2 is the signal variance
    """

    def __init__(self, x=None):
        if x is None:
            raise ValueError("N x D data matrix must be supplied as input")
        if len(x.shape) == 1:
            self.D = 1
        else:
            self.D = x.shape[1]
        self.n_params = self.D + 1

    def cov(self, theta, x, z=None):
        self.ell = np.exp(theta[0:self.D])
        self.sf2 = np.exp(2*theta[self.D])

        if z is None:
            z = x

        R = squared_dist(x.dot(np.diag(1./self.ell)),
                         z.dot(np.diag(1./self.ell)))
        K = self.sf2*np.exp(-R/2)
        return K

    def dcov(self, theta, x, i):
        K = self.cov(theta, x)
        if i < self.D:    # return derivative of lengthscale parameter
            dK = K * squared_dist(x[:, i]/self.ell[i], x[:, i]/self.ell[i])
            return dK
        elif i == self.D:   # return derivative of signal variance parameter
            dK = 2*K
            return dK
        else:
            raise ValueError("Invalid covariance function parameter")


class CovSum(CovBase):
    """ Sum of covariance functions. These are passed in as a cell array and
        intialised automatically. For example::

            C = CovSum(x,(CovLin, CovSqExpARD))
            C = CovSum.cov(x, )

        The hyperparameters are::

            theta = ( log(ell_1, ..., log_ell_D), log(sf2) )

        where ell_i are lengthscale parameters and sf2 is the signal variance
    """

    def __init__(self, x=None, covfuncnames=None):
        if x is None:
            raise ValueError("N x D data matrix must be supplied as input")
        if covfuncnames is None:
            raise ValueError("A list of covariance functions is required")
        self.covfuncs = []
        self.n_params = 0
        for cname in covfuncnames:
            covfunc = eval(cname + '(x)')
            self.n_params += covfunc.get_n_params()
            self.covfuncs.append(covfunc)

        if len(x.shape) == 1:
            self.N = len(x)
            self.D = 1
        else:
            self.N, self.D = x.shape

    def cov(self, theta, x, z=None):
        theta_offset = 0
        for ci, covfunc in enumerate(self.covfuncs):
            try: 
                n_params_c = covfunc.get_n_params()
                theta_c = [theta[c] for c in
                           range(theta_offset, theta_offset + n_params_c)]
                theta_offset += n_params_c                
            except Exception as e:
                print(e)

            if ci == 0:
                K = covfunc.cov(theta_c, x, z)
            else:
                K += covfunc.cov(theta_c, x, z)
        return K

    def dcov(self, theta, x, i):
        theta_offset = 0
        for covfunc in self.covfuncs:
            n_params_c = covfunc.get_n_params()
            theta_c = [theta[c] for c in
                       range(theta_offset, theta_offset + n_params_c)]
            theta_offset += n_params_c

            if theta_c:  # does the variable have any hyperparameters?
                if 'dK' not in locals():
                    dK = covfunc.dcov(theta_c, x, i)
                else:
                    dK += covfunc.dcov(theta_c, x, i)
        return dK

# -----------------------
# Gaussian process models
# -----------------------


class GPR:
    """Gaussian process regression

    Estimation and prediction of Gaussian process regression models

    Basic usage::

        G = GPR()
        hyp = B.estimate(hyp0, cov, X, y)
        ys, ys2 = B.predict(hyp, cov, X, y, Xs)

    where the variables are

    :param hyp: vector of hyperparmaters
    :param cov: covariance function
    :param X: N x D data array
    :param y: 1D Array of targets (length N)
    :param Xs: Nte x D array of test cases
    :param hyp0: starting estimates for hyperparameter optimisation

    :returns: * ys - predictive mean
              * ys2 - predictive variance

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

    def __init__(self, hyp=None, covfunc=None, X=None, y=None, n_iter=100,
                 tol=1e-3, verbose=False):

        self.hyp = np.nan
        self.nlZ = np.nan
        self.tol = tol          # not used at present
        self.n_iter = n_iter
        self.verbose = verbose

        if (hyp is not None) and (X is not None) and (y is not None):
            self.post(hyp, covfunc, X, y)

    def _updatepost(self, hyp, covfunc):

        hypeq = np.asarray(hyp == self.hyp)
        if hypeq.all() and hasattr(self, 'alpha') and \
           (hasattr(self, 'covfunc') and covfunc == self.covfunc):
            return False
        else:
            return True

    def post(self, hyp, covfunc, X, y):
        """ Generic function to compute posterior distribution.
        """
        
        if len(hyp.shape) > 1: # force 1d hyperparameter array
            hyp = hyp.flatten()

        if len(X.shape) == 1:
            X = X[:, np.newaxis]
        self.N, self.D = X.shape

        # hyperparameters
        sn2 = np.exp(2*hyp[0])       # noise variance
        theta = hyp[1:]            # (generic) covariance hyperparameters

        if self.verbose:
            print("estimating posterior ... | hyp=", hyp)

        self.K = covfunc.cov(theta, X)
        self.L = chol(self.K + sn2*np.eye(self.N))
        self.alpha = solve(self.L.T, solve(self.L, y))
        self.hyp = hyp
        self.covfunc = covfunc

    def loglik(self, hyp, covfunc, X, y):
        """ Function to compute compute log (marginal) likelihood
        """

        # load or recompute posterior
        if self.verbose:
            print("computing likelihood ... | hyp=", hyp)
            
        if len(hyp.shape) > 1: # force 1d hyperparameter array
            hyp = hyp.flatten()
        if self._updatepost(hyp, covfunc):
            try:
                self.post(hyp, covfunc, X, y)
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

    def dloglik(self, hyp, covfunc, X, y):
        """ Function to compute derivatives
        """

        if len(hyp.shape) > 1: # force 1d hyperparameter array
            hyp = hyp.flatten()
            
        # hyperparameters
        sn2 = np.exp(2*hyp[0])       # noise variance
        theta = hyp[1:]            # (generic) covariance hyperparameters

        # load posterior and prior covariance
        if self._updatepost(hyp, covfunc):
            try:
                self.post(hyp, covfunc, X, y)
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
            dK = covfunc.dcov(theta, X, i=par)
            self.dnlZ[par+1] = -0.5*np.sum(np.sum(Q*dK.T))

        # make sure the gradient is finite to stop the minimizer getting upset
        if not all(np.isfinite(self.dnlZ)):
            bad = np.where(np.logical_not(np.isfinite(self.dnlZ)))
            for b in bad:
                self.dnlZ[b] = np.sign(self.dnlZ[b]) / np.finfo(float).eps

        if self.verbose:
            print("dnlZ= ", self.dnlZ, " | hyp=", hyp)

        return self.dnlZ

    # model estimation (optimization)
    def estimate(self, hyp0, covfunc, X, y, optimizer='cg'):
        """ Function to estimate the model
        """
        if len(X.shape) == 1:
            X = X[:, np.newaxis]

        self.hyp0 = hyp0
        
        if optimizer.lower() == 'cg':  # conjugate gradients
            out = optimize.fmin_cg(self.loglik, hyp0, self.dloglik,
                                   (covfunc, X, y), disp=True, gtol=self.tol,
                                   maxiter=self.n_iter, full_output=1)

        elif optimizer.lower() == 'powell':  # Powell's method
            out = optimize.fmin_powell(self.loglik, hyp0, (covfunc, X, y),
                                       full_output=1)
        else:
            raise ValueError("unknown optimizer")

        # Always return a 1d array. The optimizer sometimes changes dimesnions
        if len(out[0].shape) > 1:
            self.hyp = out[0].flatten()
        else:
            self.hyp = out[0]
        self.nlZ = out[1]
        self.optimizer = optimizer

        return self.hyp

    def predict(self, hyp, X, y, Xs):
        """ Function to make predictions from the model
        """
        if len(hyp.shape) > 1: # force 1d hyperparameter array
            hyp = hyp.flatten()
        
        # ensure X and Xs are multi-dimensional arrays
        if len(Xs.shape) == 1:
            Xs = Xs[:, np.newaxis]
        if len(X.shape) == 1:
            X = X[:, np.newaxis]
            
        # reestimate posterior (avoids numerical problems with optimizer)
        self.post(hyp, self.covfunc, X, y)
        
        # hyperparameters
        sn2 = np.exp(2*hyp[0])     # noise variance
        theta = hyp[1:]            # (generic) covariance hyperparameters

        Ks = self.covfunc.cov(theta, Xs, X)
        kss = self.covfunc.cov(theta, Xs)

        # predictive mean
        ymu = Ks.dot(self.alpha)

        # predictive variance (for a noisy test input)
        v = solve(self.L, Ks.T)
        ys2 = kss - v.T.dot(v) + sn2

        return ymu, ys2
