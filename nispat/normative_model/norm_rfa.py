from __future__ import print_function
from __future__ import division

import os
import sys
import numpy as np

try:  # run as a package if installed
    from nispat.normative_model.normbase import NormBase
    from nispat.rfa import GPRRFA 
except ImportError:
    pass

    path = os.path.abspath(os.path.dirname(__file__))
    if path not in sys.path:
        sys.path.append(path)
    del path

    from rfa import GPRRFA
    from norm_base import NormBase

class NormRFA(NormBase):
    """ Classical GPR-based normative modelling approach
    """

    def __init__(self, X, y=None, theta=None, n_feat=None):
                        
        if (X is not None):
            if n_feat is None:
                print("initialising RFA")
            else:
                print("initialising RFA with", n_feat, "random features")
            self.gprrfa = GPRRFA(theta, X, n_feat=n_feat)
            self._n_params = self.gprrfa.get_n_params(X)
        else:
            raise(ValueError, 'please specify covariates')
            return
        
        if theta is None:
            self.theta0 = np.zeros(self._n_params)
        else:
            if len(theta) == self._n_params:
                self.theta0 = theta
            else:
                raise(ValueError, 'hyperparameter vector has incorrect size')
       
        self.theta = self.theta0
            
    @property
    def n_params(self):
           
        return self._n_params
    
    @property
    def neg_log_lik(self):
        return self.gprrfa.nlZ

    def estimate(self, X, y, theta=None):
        if theta is None:
            theta = self.theta0
        self.gprrfa = GPRRFA(theta, X, y)
        self.theta = self.gprrfa.estimate(theta, X, y)
        
        return self.theta

    def predict(self, X, y, Xs, theta=None):
        if theta is None:
            theta = self.theta
        yhat, s2 = self.gprrfa.predict(theta, X, y, Xs)
        
        return yhat, s2