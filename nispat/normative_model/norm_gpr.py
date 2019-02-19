from __future__ import print_function
from __future__ import division

import os
import sys
import numpy as np

try:  # run as a package if installed
    from nispat.gp import GPR, CovSum
    from nispat.gp.normative_model.normbase import NormBase
except ImportError:
    pass

    path = os.path.abspath(os.path.dirname(__file__))
    if path not in sys.path:
        sys.path.append(path)
    del path

    from gp import GPR, CovSum
    from norm_base import NormBase

class NormGPR(NormBase):
    """ Classical GPR-based normative modelling approach
    """

    def __init__(self, X=None, y=None, theta=None):
        self.covfunc = CovSum(X, ('CovLin', 'CovSqExpARD'))
        self.theta0 = np.zeros(self.covfunc.get_n_params() + 1)
        self.theta = self.theta0
        
        print("Initialising GPR")
        if (theta is not None) and (X is not None) and (y is not None):
            self.gpr = GPR(theta, self.covfunc, X, y)
            self._n_params = self.covfunc.get_n_params() + 1
        else:
            self.gpr = GPR()
            
    @property
    def n_params(self):
        if not hasattr(self,'_n_params'):
             self._n_params = self.covfunc.get_n_params() + 1
    
        return self._n_params
    
    @property
    def neg_log_lik(self):
        return self.gpr.nlZ

    def estimate(self, X, y, theta=None):
        if theta is None:
            theta = self.theta0
            self.gpr = GPR(theta, self.covfunc, X, y)
        self.theta = self.gpr.estimate(theta, self.covfunc, X, y)
        
        return self.theta

    def predict(self, X, y, Xs, theta=None):
        if theta is None:
            theta = self.theta
        yhat, s2 = self.gpr.predict(theta, X, y, Xs)
        
        # only return the marginal variances
        if len(s2.shape) == 2:
            s2 = np.diag(s2)
        
        return yhat, s2