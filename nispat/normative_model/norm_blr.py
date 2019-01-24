from __future__ import print_function
from __future__ import division

import os
import sys
import numpy as np

try:  # run as a package if installed
    from nispat.bayesreg import BLR
    from nispat.gp.normative_model.normbase import NormBase
except ImportError:
    pass

    path = os.path.abspath(os.path.dirname(__file__))
    if path not in sys.path:
        sys.path.append(path)
    del path

    from bayesreg import BLR
    from norm_base import NormBase

class NormBLR(NormBase):
    """ Normative modelling based on Bayesian Linear Regression
    """     
    def _create_poly_basis(self, X, dimpoly): 
        if len(X.shape) == 1:
            X = X[:, np.newaxis]
        Phi = np.zeros((X.shape[0], self.D*dimpoly))
        colid = np.arange(0, self.D)
        for d in range(1, dimpoly+1):
            Phi[:, colid] = X ** d
            colid += self.D
        
        return Phi
            
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
            self.Phi = self._create_poly_basis(X, self._model_order)
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
            self.Phi = self._create_poly_basis(X, self._model_order)
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

        Phis = self._create_poly_basis(Xs, self._model_order)
        yhat, s2 = self.blr.predict(theta, self.Phi, y, Phis)
        
        return yhat, s2