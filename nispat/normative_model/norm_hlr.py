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
    from nispat.hlr import HLR 
except ImportError:
    pass

    path = os.path.abspath(os.path.dirname(__file__))
    if path not in sys.path:
        sys.path.append(path)
    del path

    from hlr import HLR
    from norm_base import NormBase

class NormHLR(NormBase):
    """ Classical GPR-based normative modelling approach
    """

    def __init__(self, X, y=None, configparam=None):
        #self.cov_ind = configparam      

        with open(configparam, 'rb') as handle:
             data = pickle.load(handle)
        self.configparam = data
        
        if (X is not None):
            self.hlr = HLR(np.squeeze(X), 
                           np.squeeze(self.configparam['train'][:, 0]), 
                           np.squeeze(self.configparam['train'][:, 1]), np.squeeze(y))
        else:
            raise(ValueError, 'please specify covariates')
            return
        
    @property
    def n_params(self):
        return 1
    
    def estimate(self, X, y=None):
        self.hlr.estimate()
        return None
        
    def predict(self, X, y, Xs, theta=None): 
        yhat, s2 = self.hlr.predict(np.squeeze(Xs), 
                                    np.squeeze(self.configparam['test'][:, 0]), 
                                    np.squeeze(self.configparam['test'][:, 1]))
        
#         yhat, s2 = self.hlr.predict(np.squeeze(Xs[:,self.cov_ind['age']]), 
#                                    np.squeeze(Xs[:,self.cov_ind['site']]), 
#                                    np.squeeze(Xs[:,self.cov_ind['gender']]))
        return yhat, s2