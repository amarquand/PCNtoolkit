from __future__ import print_function
from __future__ import division

import os
import sys
import numpy as np
from scipy import optimize
from numpy.linalg import solve, LinAlgError
from numpy.linalg import cholesky as chol
    
from bayesreg import BLR

# ----------------------------
# Random Feature Approximation
# ----------------------------


class GPRRFA:
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

    
        XO = X.dot(Omega)
        Phi = np.sqrt(sn2_est/Nf)*np.c_[np.cos(XO),np.sin(XO)]
        
        XsO = Xs.dot(Omega)
        Phis = np.sqrt(sn2_est/Nf)*np.c_[np.cos(XsO),np.sin(XsO)]

        hyp_blr = np.asarray([np.log(1/sn2_est), np.log(1/sf2_est)])
        B = BLR(hyp_blr, Phi, y)
        yhat_blr, s2_blr = B.predict(hyp_blr, Phi, y, Phis)
        #s2_blr = np.diag(s2_blr)



        if self.verbose:
            print("dnlZ= ", self.dnlZ, " | hyp=", hyp)

        return self.dnlZ

    # model estimation (optimization)
    def estimate(self, hyp0, covfunc, X, y, optimizer='cg'):
        """ Function to estimate the model
        """
        if len(X.shape) == 1:
            X = X[:, np.newaxis]

        if optimizer.lower() == 'cg':  # conjugate gradients
            out = optimize.fmin_cg(self.loglik, hyp0, self.dloglik,
                                   (covfunc, X, y), disp=True, gtol=self.tol,
                                   maxiter=self.n_iter, full_output=1)

        elif optimizer.lower() == 'powell':  # Powell's method
            out = optimize.fmin_powell(self.loglik, hyp0, (covfunc, X, y),
                                       full_output=1)
        else:
            raise ValueError("unknown optimizer")

        self.hyp = out[0]
        self.nlZ = out[1]
        self.optimizer = optimizer

        return self.hyp

    def predict(self, hyp, X, y, Xs):
        """ Function to make predictions from the model
        """
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
