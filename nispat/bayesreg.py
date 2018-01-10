from __future__ import print_function
from __future__ import division

import numpy as np
from scipy import optimize , linalg
from scipy.linalg import LinAlgError


class BLR:
    """Bayesian linear regression

    Estimation and prediction of Bayesian linear regression models

    Basic usage::

        B = BLR()
        hyp = B.estimate(hyp0, X, y)
        ys,s2 = B.predict(hyp, X, y, Xs)

    where the variables are

    :param hyp: vector of hyperparmaters.
    :param X: N x D data array
    :param y: 1D Array of targets (length N)
    :param Xs: Nte x D array of test cases
    :param hyp0: starting estimates for hyperparameter optimisation

    :returns: * ys - predictive mean
              * s2 - predictive variance

    The hyperparameters are::

        hyp = ( log(beta), log(alpha) )  # hyp is a list or numpy array

    The implementation and notation mostly follows Bishop (2006).
    The hyperparameter beta is the noise precision and alpha is the precision
    over lengthscale parameters. This can be either a scalar variable (a
    common lengthscale for all input variables), or a vector of length D (a
    different lengthscale for each input variable, derived using an automatic
    relevance determination formulation). These are estimated using conjugate
    gradient optimisation of the marginal likelihood.

    Reference:
    Bishop (2006) Pattern Recognition and Machine Learning, Springer

    Written by A. Marquand
    """

    def __init__(self, hyp=None, X=None, y=None,
                 n_iter=100, tol=1e-3, verbose=False):

        self.hyp = np.nan
        self.nlZ = np.nan
        self.tol = tol          # not used at present
        self.n_iter = n_iter
        self.verbose = verbose

        if (hyp is not None) and (X is not None) and (y is not None):
            self.post(hyp, X, y)

    def post(self, hyp, X, y):
        """ Generic function to compute posterior distribution.

            This function will save the posterior mean and precision matrix as
            self.m and self.A and will also update internal parameters (e.g.
            N, D and the prior covariance (Sigma) and precision (iSigma).
        """

        N = X.shape[0]
        if len(X.shape) == 1:
            D = 1
        else:
            D = X.shape[1]

        if (hyp == self.hyp).all() and hasattr(self, 'N'):
            print("hyperparameters have not changed, exiting")
            return

        # hyperparameters
        beta = np.exp(hyp[0])    # noise precision
        alpha = np.exp(hyp[1:])  # precision for the coefficients

        if self.verbose:
            print("estimating posterior ... | hyp=", hyp)

        # prior variance
        if len(alpha) == 1 or len(alpha) == D:
            self.Sigma = np.diag(np.ones(D))/alpha
            self.iSigma = np.diag(np.ones(D))*alpha
        else:
            raise ValueError("hyperparameter vector has invalid length")

        # compute posterior precision and mean
        self.A = beta*X.T.dot(X) + self.iSigma
        self.m = beta*linalg.solve(self.A, X.T, check_finite=False).dot(y)

        # save stuff
        self.N = N
        self.D = D
        self.hyp = hyp

    def loglik(self, hyp, X, y):
        """ Function to compute compute log (marginal) likelihood """

        # hyperparameters (only beta needed)
        beta = np.exp(hyp[0])  # noise precision

        # load posterior and prior covariance
        if (hyp != self.hyp).all() or not(hasattr(self, 'A')):
            try:
                self.post(hyp, X, y)
            except ValueError:
                print("Warning: Estimation of posterior distribution failed")
                nlZ = 1/np.finfo(float).eps
                return nlZ

        try:
            # compute the log determinants in a numerically stable way
            logdetA = 2*sum(np.log(np.diag(np.linalg.cholesky(self.A))))
        except (ValueError, LinAlgError):
            print("Warning: Estimation of posterior distribution failed")
            nlZ = 1/np.finfo(float).eps
            return nlZ

        logdetSigma = sum(np.log(np.diag(self.Sigma)))  # Sigma is diagonal

        # compute negative marginal log likelihood
        nlZ = -0.5 * (self.N*np.log(beta) - self.N*np.log(2*np.pi) -
                      logdetSigma -
                      beta*(y-X.dot(self.m)).T.dot(y-X.dot(self.m)) -
                      self.m.T.dot(self.iSigma).dot(self.m) -
                      logdetA
                      )

        # make sure the output is finite to stop the minimizer getting upset
        if not np.isfinite(nlZ):
            nlZ = 1/np.finfo(float).eps

        if self.verbose:
            print("nlZ= ", nlZ, " | hyp=", hyp)

        self.nlZ = nlZ
        return nlZ

    def dloglik(self, hyp, X, y):
        """ Function to compute derivatives """

        # hyperparameters
        beta = np.exp(hyp[0])
        alpha = np.exp(hyp[1:])

        # load posterior and prior covariance
        if (hyp != self.hyp).all() or not(hasattr(self, 'A')):
            try:
                self.post(hyp, X, y)
            except ValueError:
                print("Warning: Estimation of posterior distribution failed")
                dnlZ = np.sign(self.dnlZ) / np.finfo(float).eps
                return dnlZ

        # useful quantities
        XX = X.T.dot(X)
        S = np.linalg.inv(self.A)  # posterior covariance
        Q = S.dot(X.T)
        # Q = linalg.solve(self.A, X.T)
        b = (np.eye(self.D) - beta*Q.dot(X)).dot(Q).dot(y)

        # initialise derivatives
        dnlZ = np.zeros(hyp.shape)

        # noise precision
        dnlZ[0] = - (self.N / (2 * beta) - 0.5 * y.dot(y) +
                     y.dot(X).dot(self.m) +
                     beta * y.T.dot(X).dot(b) -
                     0.5 * self.m.T.dot(XX).dot(self.m) -
                     beta * b.T.dot(self.iSigma).dot(self.m) -
                     0.5 * np.trace(Q.dot(X))
                     ) * beta

        # scaling parameter(s)
        for i in range(0, len(alpha)):
            # are we using ARD?
            if len(alpha) == self.D:
                dSigma = np.zeros((self.D, self.D))
                dSigma[i, i] = -alpha[i] ** -2
                diSigma = np.zeros((self.D, self.D))
                diSigma[i, i] = 1
            else:
                dSigma = -alpha[i] ** -2*np.eye(self.D)
                diSigma = np.eye(self.D)

            F = diSigma
            c = -beta*S.dot(F).dot(S).dot(X.T).dot(y)

            dnlZ[i+1] = -(-0.5 * np.trace(self.iSigma.dot(dSigma)) +
                          beta * y.T.dot(X).dot(c) -
                          beta * c.T.dot(XX).dot(self.m) -
                          c.T.dot(self.iSigma).dot(self.m) -
                          0.5 * self.m.T.dot(F).dot(self.m) -
                          0.5*np.trace(linalg.solve(self.A, F))
                          ) * alpha[i]

        # make sure the gradient is finite to stop the minimizer getting upset
        if not all(np.isfinite(dnlZ)):
            bad = np.where(np.logical_not(np.isfinite(dnlZ)))
            for b in bad:
                dnlZ[b] = np.sign(self.dnlZ[b]) / np.finfo(float).eps

        if self.verbose:
            print("dnlZ= ", dnlZ, " | hyp=", hyp)

        self.dnlZ = dnlZ
        return dnlZ

    # model estimation (optimization)
    def estimate(self, hyp0, X, y, optimizer='cg'):
        """ Function to estimate the model """

        if optimizer.lower() == 'cg':  # conjugate gradients
            out = optimize.fmin_cg(self.loglik, hyp0, self.dloglik, (X, y),
                                   disp=True, gtol=self.tol,
                                   maxiter=self.n_iter, full_output=1)

        elif optimizer.lower() == 'powell':  # Powell's method
            out = optimize.fmin_powell(self.loglik, hyp0, (X, y),
                                       full_output=1)
        else:
            raise ValueError("unknown optimizer")

        self.hyp = out[0]
        self.nlZ = out[1]
        self.optimizer = optimizer

        return self.hyp

    def predict(self, hyp, X, y, Xs):
        """ Function to make predictions from the model """

        if (hyp != self.hyp).all() or not(hasattr(self, 'A')):
            self.post(hyp, X, y)

        # hyperparameters
        beta = np.exp(hyp[0])

        ys = Xs.dot(self.m)
        # compute xs.dot(S).dot(xs.T) avoiding computing off-diagonal entries
        s2 = 1/beta + np.sum(Xs*linalg.solve(self.A, Xs.T).T, axis=1)

        return ys, s2
