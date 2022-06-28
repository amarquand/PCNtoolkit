from __future__ import print_function
from __future__ import division

import os
import sys
import numpy as np
import pandas as pd
from ast import literal_eval

try:  # run as a package if installed
    from pcntoolkit.model.bayesreg import BLR
    from pcntoolkit.normative_model.norm_base import NormBase
    from pcntoolkit.dataio import fileio
    from pcntoolkit.util.utils import create_poly_basis, WarpBoxCox, \
                                  WarpAffine, WarpCompose, WarpSinArcsinh
except ImportError:
    pass

    path = os.path.abspath(os.path.dirname(__file__))
    if path not in sys.path:
        sys.path.append(path)
    del path

    from model.bayesreg import BLR
    from norm_base import NormBase
    from dataio import fileio
    from util.utils import create_poly_basis, WarpBoxCox, \
                      WarpAffine, WarpCompose, WarpSinArcsinh

class NormBLR(NormBase):
    """ Normative modelling based on Bayesian Linear Regression
    """     
            
    def __init__(self,  **kwargs):
        X = kwargs.pop('X', None)
        y = kwargs.pop('y', None)
        theta = kwargs.pop('theta', None)
        if isinstance(theta, str):
            theta = np.array(literal_eval(theta))
        self.optim_alg = kwargs.get('optimizer','powell')

        if X is None:
            raise(ValueError, "Data matrix must be specified")

        if len(X.shape) == 1:
            self.D = 1
        else:
            self.D = X.shape[1]
        
        # Parse model order
        if kwargs is None:
            model_order = 1
        elif 'configparam' in kwargs: # deprecated syntax
            model_order = kwargs.pop('configparam')
        elif 'model_order' in kwargs: 
            model_order = kwargs.pop('model_order')
        else:
            model_order = 1
            
        # Force a default model order and check datatype
        if model_order is None:
            model_order = 1
        if type(model_order) is not int:
            model_order = int(model_order)
        
        # configure heteroskedastic noise
        if 'varcovfile' in kwargs:
            var_cov_file = kwargs.get('varcovfile')
            if var_cov_file.endswith('.pkl'):
                self.var_covariates = pd.read_pickle(var_cov_file)
            else:
                self.var_covariates = np.loadtxt(var_cov_file)
            if len(self.var_covariates.shape) == 1:
                self.var_covariates = self.var_covariates[:, np.newaxis]
            n_beta = self.var_covariates.shape[1]
            self.var_groups = None
        elif 'vargroupfile' in kwargs:
            # configure variance groups (e.g. site specific variance)
            var_groups_file = kwargs.pop('vargroupfile')
            if var_groups_file.endswith('.pkl'):
                self.var_groups = pd.read_pickle(var_groups_file)
            else:
                self.var_groups = np.loadtxt(var_groups_file)
            var_ids = set(self.var_groups)
            var_ids = sorted(list(var_ids))
            n_beta = len(var_ids)
        else:
            self.var_groups = None
            self.var_covariates = None
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
        
        print("configuring BLR ( order", model_order, ")")
        if (theta is None) or (len(theta) != self._n_params):
            print("Using default hyperparameters")
            theta0 = np.zeros(self._n_params)
        else:
            self.theta0 = theta
        self.theta = self.theta0
        
        # initialise the BLR object if the required parameters are present
        if (theta is not None) and (y is not None):
            Phi = create_poly_basis(X, self._model_order)
            self.blr = BLR(theta=theta, X=Phi, y=y, 
                           warp=self.warp, **kwargs)
        else:
            self.blr = BLR(**kwargs)    
            
    @property
    def n_params(self):
        return self._n_params
    
    @property
    def neg_log_lik(self):
        return self.blr.nlZ

    def estimate(self, X, y, **kwargs):
        theta = kwargs.pop('theta', None)
        if isinstance(theta, str):
            theta = np.array(literal_eval(theta))
            
        # remove warp string to prevent it being passed to the blr object
        kwargs.pop('warp',None) 
        
        Phi = create_poly_basis(X, self._model_order)
        if len(y.shape) > 1:
            y = y.ravel()
            
        if theta is None:
            theta = self.theta0           
            
            # (re-)initialize BLR object because parameters were not specified
            self.blr = BLR(theta=theta, X=Phi, y=y, 
                           var_groups=self.var_groups, 
                           warp=self.warp, **kwargs)

        self.theta = self.blr.estimate(theta, Phi, y, 
                                       var_covariates=self.var_covariates, **kwargs)
        
        return self

    def predict(self, Xs, X=None, y=None, **kwargs):      
        
        theta = self.theta # always use the estimated coefficients
        # remove from kwargs to avoid downstream problems
        kwargs.pop('theta', None)
        
        

        Phis = create_poly_basis(Xs, self._model_order)
        
        if X is None:
            Phi =None
        else:
            Phi = create_poly_basis(X, self._model_order)
        
        # process variance groups for the test data
        if 'testvargroupfile' in kwargs:
            var_groups_test_file = kwargs.pop('testvargroupfile')
            if var_groups_test_file.endswith('.pkl'):
                var_groups_te = pd.read_pickle(var_groups_test_file)
            else:
                var_groups_te = np.loadtxt(var_groups_test_file)
        else:
            var_groups_te = None
        
        # process test variance covariates
        if 'testvarcovfile' in kwargs:
            var_cov_test_file = kwargs.get('testvarcovfile')
            if var_cov_test_file.endswith('.pkl'):
                var_cov_te = pd.read_pickle(var_cov_test_file)
            else:
                var_cov_te = np.loadtxt(var_cov_test_file)
        else:
            var_cov_te = None
        
        # do we want to adjust the responses?
        if 'adaptrespfile' in kwargs:
            y_adapt = fileio.load(kwargs.pop('adaptrespfile'))
            if len(y_adapt.shape) == 1:
                y_adapt = y_adapt[:, np.newaxis]
        else:
            y_adapt = None
        
        if 'adaptcovfile' in kwargs:
            X_adapt = fileio.load(kwargs.pop('adaptcovfile'))
            Phi_adapt = create_poly_basis(X_adapt, self._model_order)
        else:
            Phi_adapt = None
        
        if 'adaptvargroupfile' in kwargs: 
            var_groups_adapt_file = kwargs.pop('adaptvargroupfile')
            if var_groups_adapt_file.endswith('.pkl'):
                var_groups_ad = pd.read_pickle(var_groups_adapt_file)
            else:
                var_groups_ad = np.loadtxt(var_groups_adapt_file)
        else:
            var_groups_ad = None
            
        
        if y_adapt is None:
            yhat, s2 = self.blr.predict(theta, Phi, y, Phis, 
                                        var_groups_test=var_groups_te,
                                        var_covariates_test=var_cov_te, 
                                        **kwargs)
        else:
            yhat, s2 = self.blr.predict_and_adjust(theta, Phi_adapt, y_adapt, Phis, 
                                                   var_groups_test=var_groups_te,
                                                   var_groups_adapt=var_groups_ad, 
                                                   **kwargs)
        
        return yhat, s2
    