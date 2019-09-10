from __future__ import print_function
from __future__ import division

import os
import sys
import numpy as np

try:  # run as a package if installed
    from nispat.bayesreg import BLR
    from nispat.normative_model.normbase import NormBase
    from nispat.utils import create_poly_basis
except ImportError:
    pass

    path = os.path.abspath(os.path.dirname(__file__))
    if path not in sys.path:
        sys.path.append(path)
    del path

    from bayesreg import BLR
    from norm_base import NormBase
    from utils import create_poly_basis

class NormBLR(NormBase):
    """ Normative modelling based on Bayesian Linear Regression
    """     
            
    def __init__(self, X=None, y=None, theta=None, model_order=3):
        if X is None:
            raise(ValueError, "Data matrix must be specified")

        if len(X.shape) == 1:
            self.D = 1
        else:
            self.D = X.shape[1]
            
        # Force a default value and check datatype
        if model_order is None:
            model_order = 3
        elif type(model_order) is not int:
            model_order = int(model_order)
        
        self._n_params = 1 + self.D * model_order
        self._model_order = model_order
        
        print("initialising BLR ( order", model_order, ")")
        if (theta is None) or (len(theta) != self._n_params):
            print("Using default hyperparameters")
            self.theta0 = np.zeros(self._n_params)
        else:
            self.theta0 = theta
        self.theta = self.theta0
        
        if (theta is not None) and (y is not None):
            self.Phi = create_poly_basis(X, self._model_order)
            self.gpr = BLR(theta, self.Phi, y)
        else:
            self.gpr = BLR()    
            
    @property
    def n_params(self):
        return self._n_params
    
    @property
    def neg_log_lik(self):
        return self.blr.nlZ

    def estimate(self, X, y, theta=None):
        if not hasattr(self,'Phi'):
            self.Phi = create_poly_basis(X, self._model_order)
        if len(y.shape) > 1:
            y = y.ravel()
            
        if theta is None:
            theta = self.theta0
            self.blr = BLR(theta, self.Phi, y)

        self.theta = self.blr.estimate(theta, self.Phi, y)
        
        return self.theta

    def predict(self, X, y, Xs, theta=None):
        if theta is None:
            theta = self.theta

        Phis = create_poly_basis(Xs, self._model_order)
        yhat, s2 = self.blr.predict(theta, self.Phi, y, Phis)
        
        return yhat, s2