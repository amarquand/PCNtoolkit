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
                 n_iter=100, tol=1e-3, verbose=False, 
                 var_groups=None, warp=None):

        self.hyp = np.nan
        self.nlZ = np.nan
        self.tol = tol          # not used at present
        self.n_iter = n_iter
        self.verbose = verbose
        self.var_groups = var_groups 
        if self.var_groups is not None:
            self.var_ids = set(self.var_groups)
            self.var_ids = sorted(list(self.var_ids))

        # set up warped likelihood
        if warp is None:
            self.warp = None
            self.n_warp_param = 0
        else:
            self.warp = warp
            self.n_warp_param = warp.get_n_params()
        self.gamma = None
    
    def _parse_hyps(self, hyp, X):

        N = X.shape[0]
        
        # hyperparameters
        if self.var_groups is None:
            beta = np.asarray([np.exp(hyp[0])])               # noise precision 
            self.Lambda_n = np.diag(np.ones(N)*beta)
            self.Sigma_n = np.diag(np.ones(N)/beta)
        else:
            beta = np.exp(hyp[0:len(self.var_ids)])
            beta_all = np.ones(N)
            for v in range(len(self.var_ids)):
                beta_all[self.var_groups == self.var_ids[v]] = beta[v]
            self.Lambda_n = np.diag(beta_all)
            self.Sigma_n = np.diag(1/beta_all)
        
        # parameters for warping the likelhood function
        n_lik_param = len(beta)
        if self.warp is not None:
            gamma = hyp[n_lik_param:(n_lik_param + self.n_warp_param)]
            n_lik_param += self.n_warp_param
        else:
            gamma = None

        # precision for the coefficients
        if isinstance(beta, list) or type(beta) is np.ndarray:
            alpha = np.exp(hyp[n_lik_param:])
        else:
            alpha = np.exp(hyp[1:])

        return beta, alpha, gamma
        
    def post(self, hyp, X, y):
        """ Generic function to compute posterior distribution.

            This function will save the posterior mean and precision matrix as
            self.m and self.A and will also update internal parameters (e.g.
            N, D and the prior covariance (Sigma_a) and precision (Lambda_a).
        """

        N = X.shape[0]
        if len(X.shape) == 1:
            D = 1
        else:
            D = X.shape[1]

        if (hyp == self.hyp).all() and hasattr(self, 'N'):
            print("hyperparameters have not changed, exiting")
            return
        
        beta, alpha, gamma = self._parse_hyps(hyp, X)

        if self.verbose:
            print("estimating posterior ... | hyp=", hyp)

        # prior variance
        if len(alpha) == 1 or len(alpha) == D:
            self.Sigma_a = np.diag(np.ones(D))/alpha
            self.Lambda_a = np.diag(np.ones(D))*alpha
        else:
            raise ValueError("hyperparameter vector has invalid length")

        # compute posterior precision and mean
        self.A = X.T.dot(self.Lambda_n).dot(X) + self.Lambda_a
        self.m = linalg.solve(self.A, X.T, 
                              check_finite=False).dot(self.Lambda_n).dot(y)
        #self.m = linalg.lstsq(self.A, X.T, 
        #                      check_finite=False)[0].dot(self.Lambda_n).dot(y)

        # save stuff
        self.N = N
        self.D = D
        self.hyp = hyp

    def loglik(self, hyp, X, y):
        """ Function to compute compute log (marginal) likelihood """

        # hyperparameters (alpha not needed)
        beta, alpha, gamma = self._parse_hyps(hyp, X) 

        # warp the likelihood?
        if self.warp is not None:
            if self.verbose:
                print('warping input...')
            y_unwarped = y
            y = self.warp.f(y, gamma)
            
        # load posterior and prior covariance
        if (hyp != self.hyp).any() or not(hasattr(self, 'A')):
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

        logdetSigma_a = sum(np.log(np.diag(self.Sigma_a))) # diagonal
        logdetSigma_n = sum(np.log(np.diag(self.Sigma_n)))

        # compute negative marginal log likelihood
        nlZ = -0.5 * (-self.N*np.log(2*np.pi) -
                      logdetSigma_n -
                      logdetSigma_a -
                      (y-X.dot(self.m)).T.dot(self.Lambda_n).dot(y-X.dot(self.m)) -
                      self.m.T.dot(self.Lambda_a).dot(self.m) -
                      logdetA
                      )
        
        if self.warp is not None:
            # add in the Jacobian 
            nlZ = nlZ - sum(np.log(self.warp.df(y_unwarped, gamma)))

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
        beta, alpha, gamma = self._parse_hyps(hyp, X)
        
        if self.warp is not None:
            raise ValueError('optimization with derivatives is not yet ' + \
                             'supported for warped liklihood')

        # load posterior and prior covariance
        if (hyp != self.hyp).any() or not(hasattr(self, 'A')):
            try:
                self.post(hyp, X, y)
            except ValueError:
                print("Warning: Estimation of posterior distribution failed")
                dnlZ = np.sign(self.dnlZ) / np.finfo(float).eps
                return dnlZ

        # precompute re-used quantities to maximise speed 
        # todo: revise implementation to use Cholesky throughout 
        #       that would remove the need to explicitly compute the inverse
        S = np.linalg.inv(self.A)                       # posterior covariance
        SX = S.dot(X.T)
        XLn = X.T.dot(self.Lambda_n)
        XLny = XLn.dot(y)
        SXLny = S.dot(XLny)       
        XLnXm = XLn.dot(X).dot(self.m)
      
        # initialise derivatives
        dnlZ = np.zeros(hyp.shape)
 
        # noise precision parameter(s)
        for i in range(0, len(beta)):
            # first compute derivative of Lambda_n with respect to beta
            dL_n_vec = np.zeros(self.N)
            if self.var_groups is None:
                dL_n_vec = np.ones(self.N)
            else:    
                dL_n_vec[np.where(self.var_groups == self.var_ids[i])[0]] = 1
            dLambda_n = np.diag(dL_n_vec)
            
            # compute quantities used multiple times
            XdLnX = X.T.dot(dLambda_n).dot(X)
            dA = XdLnX
            
            # derivative of posterior parameters with respect to beta
            b = -S.dot(dA).dot(SXLny) + SX.dot(dLambda_n).dot(y)
            
            # compute np.trace(self.Sigma_n.dot(dLambda_n)) efficiently
            trSigma_ndLambda_n = sum(np.diag(self.Sigma_n)*np.diag(dLambda_n))
            
            dnlZ[i] = - (0.5 * trSigma_ndLambda_n - 
                         0.5 * y.dot(dLambda_n).dot(y) +
                         y.dot(dLambda_n).dot(X).dot(self.m) +
                         y.T.dot(self.Lambda_n).dot(X).dot(b) -
                         0.5 * self.m.T.dot(XdLnX).dot(self.m) -    
                         b.T.dot(XLnXm) -
                         b.T.dot(self.Lambda_a).dot(self.m) -
                         0.5 * np.trace(S.dot(dA))
                         ) * beta[i]

        # scaling parameter(s)
        for i in range(0, len(alpha)):
            # first compute derivatives with respect to alpha
            if len(alpha) == self.D:  # are we using ARD?
                dLambda_a = np.zeros((self.D, self.D))
                dLambda_a[i, i] = 1
            else:
                dLambda_a = np.eye(self.D)

            F = dLambda_a
            c = -S.dot(F).dot(SXLny)
            
            # compute np.trace(self.Sigma_a.dot(dLambda_a)) efficiently
            trSigma_adLambda_a = sum(np.diag(self.Sigma_a)*np.diag(dLambda_a))
            
            dnlZ[i+len(beta)] = -(0.5* trSigma_adLambda_a +
                                  XLny.T.dot(c) -
                                  c.T.dot(XLnXm) -
                                  c.T.dot(self.Lambda_a).dot(self.m) -
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

    def predict(self, hyp, X, y, Xs, var_groups_test=None):
        """ Function to make predictions from the model """

        if X is None or y is None:
            # set hyperparameters. we can use an array of zeros because 
            beta, alpha, gamma = self._parse_hyps(hyp, np.zeros((self.N, 1)))
        else:
            
            # set hyperparameters
            beta, alpha, gamma = self._parse_hyps(hyp, X)
            
            # do we need to re-estimate the posterior?
            if (hyp != self.hyp).any() or not(hasattr(self, 'A')):
                # warp the likelihood?
                if self.warp is not None:
                    if self.verbose:
                        print('warping input...')
                    y = self.warp.f(y, gamma) 
                self.post(hyp, X, y)

        N_test = Xs.shape[0]

        ys = Xs.dot(self.m)
        
        if self.var_groups is None:
            s2n = 1/beta
        else:
            if len(var_groups_test) != N_test:
                raise(ValueError, 'Invalid variance groups for test')
            # separate variance groups
            s2n = np.ones(N_test)
            for v in range(len(self.var_ids)):
                s2n[var_groups_test == self.var_ids[v]] = 1/beta[v]
        
        # compute xs.dot(S).dot(xs.T) avoiding computing off-diagonal entries
        s2 = s2n + np.sum(Xs*linalg.solve(self.A, Xs.T).T, axis=1)
        
        return ys, s2
