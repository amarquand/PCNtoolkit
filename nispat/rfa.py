from __future__ import print_function
from __future__ import division

import numpy as np
import torch

class GPRRFA:
    """Random Feature Approximation for Gaussian Process Regression

    Estimation and prediction of Bayesian linear regression models

    Basic usage::

        R = GPRRFA()
        hyp = R.estimate(hyp0, X, y)
        ys,s2 = R.predict(hyp, X, y, Xs)

    where the variables are

    :param hyp: vector of hyperparmaters.
    :param X: N x D data array
    :param y: 1D Array of targets (length N)
    :param Xs: Nte x D array of test cases
    :param hyp0: starting estimates for hyperparameter optimisation

    :returns: * ys - predictive mean
              * s2 - predictive variance

    The hyperparameters are::

        hyp = [ log(sn), log(ell), log(sf) ]  # hyp is a numpy array

    where sn^2 is the noise variance, ell are lengthscale parameters and 
    sf^2 is the signal variance. This provides an approximation to the
    covariance function: 
        
        k(x,z) = x'*z + sn2*exp(0.5*(x-z)'*Lambda*(x-z))
    
    where Lambda = diag((ell_1^2, ... ell_D^2))

    Written by A. Marquand
    """

    def __init__(self, hyp=None, X=None, y=None, n_feat=None,
                 n_iter=100, tol=1e-3, verbose=False):

        self.hyp = np.nan
        self.nlZ = np.nan
        self.tol = tol          # not used at present
        self.Nf = n_feat
        self.n_iter = n_iter
        self.verbose = verbose
        self._n_restarts = 5

        if (hyp is not None) and (X is not None) and (y is not None):
            self.post(hyp, X, y)

    def _numpy2torch(self, X, y=None, hyp=None):

        if type(X) is torch.Tensor:
           pass
        elif type(X) is np.ndarray:
           X = torch.from_numpy(X)
        else:
           raise(ValueError, 'Unknown data type (X)')
        X = X.double()
        
        if y is not None:
            if type(y) is torch.Tensor:
                pass
            elif type(y) is np.ndarray:
                y = torch.from_numpy(y)
            else:
                raise(ValueError, 'Unknown data type (y)')
            
            if len(y.shape) == 1:
                y.resize_(y.shape[0],1)
            y = y.double()
        
        if hyp is not None:
            if type(hyp) is torch.Tensor:
                pass
            else:
                hyp = torch.tensor(hyp, requires_grad=True)
        
        return X, y, hyp
    
    def get_n_params(self, X):
        
        return X.shape[1] + 2
        
    def post(self, hyp, X, y):
        """ Generic function to compute posterior distribution.

            This function will save the posterior mean and precision matrix as
            self.m and self.A and will also update internal parameters (e.g.
            N, D and the prior covariance (Sigma) and precision (iSigma).
        """
     
        # make sure all variables are the right type
        X, y, hyp = self._numpy2torch(X, y, hyp)
        
        self.N, self.Dx = X.shape
        
        # ensure the number of features is specified (use 75% as a default)
        if self.Nf is None:
            self.Nf = int(0.75 * self.N)
        
        self.Omega = torch.zeros((self.Dx, self.Nf), dtype=torch.double)
        for f in range(self.Nf):
            self.Omega[:,f] = torch.exp(hyp[1:-1]) * \
            torch.randn((self.Dx, 1), dtype=torch.double).squeeze()

        XO = torch.mm(X, self.Omega) 
        self.Phi = torch.exp(hyp[-1])/np.sqrt(self.Nf) *  \
                   torch.cat((torch.cos(XO), torch.sin(XO)), 1)
        
        # concatenate linear weights 
        self.Phi = torch.cat((self.Phi, X), 1)
        self.D = self.Phi.shape[1]

        if self.verbose:
            print("estimating posterior ... | hyp=", hyp)
        
        self.A = torch.mm(torch.t(self.Phi), self.Phi) / torch.exp(2*hyp[0]) + \
                 torch.eye(self.D, dtype=torch.double)
        self.m = torch.mm(torch.gesv(torch.t(self.Phi), self.A)[0], y) / \
                 torch.exp(2*hyp[0])

        # save hyperparameters
        self.hyp = hyp
        
        # update optimizer iteration count
        if hasattr(self,'_iterations'):
            self._iterations += 1

    def loglik(self, hyp, X, y):
        """ Function to compute compute log (marginal) likelihood """
        X, y, hyp = self._numpy2torch(X, y, hyp)

        # always recompute the posterior
        self.post(hyp, X, y)

        #logdetA = 2*torch.sum(torch.log(torch.diag(torch.cholesky(self.A))))
        try:
            # compute the log determinants in a numerically stable way
            logdetA = 2*torch.sum(torch.log(torch.diag(torch.cholesky(self.A))))
        except Exception as e:
            print("Warning: Estimation of posterior distribution failed")
            print(e)
            #nlZ = torch.tensor(1/np.finfo(float).eps)
            nlZ = torch.tensor(np.nan)
            self._optim_failed = True
            return nlZ
        
        # compute negative marginal log likelihood
        nlZ = -0.5 * (self.N*torch.log(1/torch.exp(2*hyp[0])) - 
                      self.N*np.log(2*np.pi) -
                      torch.mm(torch.t(y - torch.mm(self.Phi,self.m)),
                               (y - torch.mm(self.Phi,self.m))) / 
                      torch.exp(2*hyp[0]) -
                      torch.mm(torch.t(self.m), self.m) - logdetA)

        if self.verbose:
            print("nlZ= ", nlZ, " | hyp=", hyp)

        # save marginal likelihood
        self.nlZ = nlZ
        return nlZ

    def dloglik(self, hyp, X, y):
        """ Function to compute derivatives """

        print("derivatives not available")

        return

    def estimate(self, hyp0, X, y, optimizer='lbfgs'):
        """ Function to estimate the model """
        
        if type(hyp0) is torch.Tensor:
            hyp = hyp0
            hyp0.requires_grad_()
        else:
            hyp = torch.tensor(hyp0, requires_grad=True) 
        # save the starting values
        self.hyp0 = hyp
        
        if optimizer.lower() == 'lbfgs':
            opt = torch.optim.LBFGS([hyp])
        else:
            raise(ValueError, "Optimizer " + " not implemented")
        self._iterations = 0
        
        def closure():
            opt.zero_grad()
            nlZ = self.loglik(hyp, X, y)
            if not torch.isnan(nlZ):
                nlZ.backward()
            return nlZ
        
        for r in range(self._n_restarts):
            self._optim_failed = False
            
            nlZ = opt.step(closure)
            
            if self._optim_failed:
                print("optimization failed. retrying (", r+1, "of", 
                      self._n_restarts,")")
                hyp = torch.randn_like(hyp, requires_grad=True)
                self.hyp0 = hyp
            else:
                print("Optimzation complete after", self._iterations, 
                      "evaluations. Function value =", 
                      nlZ.detach().numpy().squeeze())
                break

        return self.hyp.detach().numpy()

    def predict(self, hyp, X, y, Xs):
        """ Function to make predictions from the model """

        X, y, hyp = self._numpy2torch(X, y, hyp)
        Xs, *_ = self._numpy2torch(Xs)

        if (hyp != self.hyp).all() or not(hasattr(self, 'A')):
            self.post(hyp, X, y)
        
        # generate prediction tensors
        XsO = torch.mm(Xs, self.Omega) 
        Phis = torch.exp(hyp[-1])/np.sqrt(self.Nf) * \
               torch.cat((torch.cos(XsO), torch.sin(XsO)), 1)
        # add linear component
        Phis = torch.cat((Phis, Xs), 1)
        
        ys = torch.mm(Phis, self.m)

        # compute diag(Phis*(Phis'\A)) avoiding computing off-diagonal entries
        s2 = torch.exp(2*hyp[0]) + \
                torch.sum(Phis * torch.t(torch.gesv(torch.t(Phis), self.A)[0]), 1)

        # return output as numpy arrays
        return ys.detach().numpy().squeeze(), s2.detach().numpy().squeeze()
