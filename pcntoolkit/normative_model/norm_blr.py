from __future__ import print_function
from __future__ import division

import os
import sys
import numpy as np
import pandas as pd

try:  # run as a package if installed
    from pcntoolkit.bayesreg import BLR
    from pcntoolkit.normative_model.normbase import NormBase
    from pcntoolkit.utils import create_poly_basis
except ImportError:
    pass

    path = os.path.abspath(os.path.dirname(__file__))
    if path not in sys.path:
        sys.path.append(path)
    del path

    from bayesreg import BLR
    from norm_base import NormBase
    from utils import create_poly_basis, WarpBoxCox, \
                      WarpAffine, WarpCompose, WarpSinArcsinh

class NormBLR(NormBase):
    """ Normative modelling based on Bayesian Linear Regression
    """     
            
    def __init__(self,  **kwargs): #X=None, y=None, theta=None,
        X = kwargs.pop('X', None)
        y = kwargs.pop('y', None)
        theta = kwargs.pop('theta', None)
        self.optim_alg = kwargs.pop('optimizer','powell')

        if X is None:
            raise(ValueError, "Data matrix must be specified")

        if len(X.shape) == 1:
            self.D = 1
        else:
            self.D = X.shape[1]
        
        # Parse model order
        if kwargs is None:
            model_order = 1
        elif 'configparam' in kwargs:
            model_order = kwargs.pop('configparam')
        elif 'model_order' in kwargs: 
            model_order = kwargs.pop('model_order')
        else:
            model_order = 1
            
        # Force a default value and check datatype
        if model_order is None:
            model_order = 1
        if type(model_order) is not int:
            model_order = int(model_order)
        
        # configure variance groups (e.g. site specific variance)
        if 'var_groups' in kwargs:
            var_groups_file = kwargs.pop('var_groups')
            if var_groups_file.endswith('.pkl'):
                self.var_groups = pd.read_pickle(var_groups_file)
            else:
                self.var_groups = np.loadtxt(var_groups_file)
            var_ids = set(self.var_groups)
            var_ids = sorted(list(var_ids))
            n_beta = len(var_ids)
        else:
            self.var_groups = None
            n_beta = 1
        
        # are we using ARD?
        if 'use_ard' in kwargs: 
            self.use_ard = kwargs.pop('use_ard')
        else:
            self.use_ard = False
        if self.use_ard:
            n_alpha = self.D * model_order
        else:
            n_alpha = 1
        
        # Configure warped likelihood
        if 'warp' in kwargs:
            warp_str = kwargs.pop('warp')
            if warp_str is None:
                self.warp = None
                n_gamma = 0
            else:
                # set up warp
                exec('self.warp =' + warp_str + '()')
                n_gamma = self.warp.get_n_params()
        else:
            self.warp = None
            n_gamma = 0

        self._n_params = n_alpha + n_beta + n_gamma
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
            self.blr = BLR(theta, self.Phi, y)
        else:
            self.blr = BLR()    
            
    @property
    def n_params(self):
        return self._n_params
    
    @property
    def neg_log_lik(self):
        return self.blr.nlZ

    def estimate(self, X, y, **kwargs):
        theta = kwargs.pop('theta', None)
        
        if not hasattr(self,'Phi'):
            self.Phi = create_poly_basis(X, self._model_order)
        if len(y.shape) > 1:
            y = y.ravel()
            
        if theta is None:
            theta = self.theta0
            self.blr = BLR(theta, self.Phi, y, 
                           var_groups=self.var_groups,
                           warp=self.warp)

        self.theta = self.blr.estimate(theta, self.Phi, y, 
                                       optimizer=self.optim_alg)
        
        return self

    def predict(self, Xs, X=None, y=None, **kwargs):
        theta = kwargs.pop('theta', None)
        
        if theta is None:
            theta = self.theta

        Phis = create_poly_basis(Xs, self._model_order)
        
        if 'var_groups_test' in kwargs:
            var_groups_test_file = kwargs.pop('var_groups_test')
            if var_groups_test_file.endswith('.pkl'):
                var_groups_te = pd.read_pickle(var_groups_test_file)
            else:
                var_groups_te = np.loadtxt(var_groups_test_file)
        else:
            var_groups_te = None
            
        yhat, s2 = self.blr.predict(theta, self.Phi, y, Phis, 
                                    var_groups_test=var_groups_te)
        
        return yhat, s2
    