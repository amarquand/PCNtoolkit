#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 25 17:01:24 2019

@author: seykia
"""

from __future__ import print_function
from __future__ import division

import os
import sys
import numpy as np
import pickle

try:  # run as a package if installed
    from nispat.normative_model.normbase import NormBase
    from nispat.hbr import HBR 
except ImportError:
    pass

    path = os.path.abspath(os.path.dirname(__file__))
    if path not in sys.path:
        sys.path.append(path)
    del path

    from hbr import HBR
    from norm_base import NormBase

class NormHBR(NormBase):
    """ Classical GPR-based normative modelling approach
    """

    def __init__(self, X, y=None, configparam=None):
        self.configparam = configparam
        
        with open(configparam, 'rb') as handle:
             data = pickle.load(handle)
        confounds = data['confounds']
        self.type = data['model_type']
        
        if (X is not None):
            self.hbr = HBR(np.squeeze(X), 
                           np.squeeze(confounds['train'][:, 0]), 
                           np.squeeze(confounds['train'][:, 1]), 
                           np.squeeze(y), self.type)
        else:
            raise(ValueError, 'please specify covariates')
            return
        
    @property
    def n_params(self):
        return 1
    
    @property
    def neg_log_lik(self):
        return -1
    
    def estimate(self, X, y=None):
        self.hbr.estimate()
        return None
        
    def predict(self, X, y, Xs, theta=None): 
        with open(self.configparam, 'rb') as handle:
             data = pickle.load(handle)
             
        confounds = data['confounds']
        yhat, s2 = self.hbr.predict(np.squeeze(Xs), 
                                    np.squeeze(confounds['test'][:, 0]), 
                                    np.squeeze(confounds['test'][:, 1]))
        

        return yhat, s2