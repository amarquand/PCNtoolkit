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
        
        if configparam is not None:
            with open(configparam, 'rb') as handle:
                configs = pickle.load(handle)
        else:
            configs = dict()    
        
        #self.type = configs['model_type']
        
        if 'batch_effects_train' in configs:
            batch_effects_train = configs['batch_effects_train']
        else:
            batch_effects_train = np.zeros([X.shape[0],2])
            
        self.configs = dict()
        
        if 'model_type' in configs:
            self.configs['type'] = configs['model_type']
        else:
            self.configs['type'] = 'linear'
        
        if 'random_intercept' in configs:
            self.configs['random_intercept'] = configs['random_intercept']
        else:
            self.configs['random_intercept'] = True
        
        if 'random_slope' in configs:
            self.configs['random_slope'] = configs['random_slope']
        else:
            self.configs['random_slope'] = True
            
        if 'random_noise' in configs:
            self.configs['random_noise'] = configs['random_noise']
        else:
            self.configs['random_noise'] = True
                
        if 'hetero_noise' in configs:
            self.configs['hetero_noise'] = configs['hetero_noise']
        else:
            self.configs['hetero_noise'] = False
            
        if 'noise_model' in configs:
            self.configs['noise_model'] = configs['noise_model']
        else:
            self.configs['noise_model'] = 'linear'
        
        if 'nn_hidden_neuron_num' in configs:
            self.configs['nn_hidden_neuron_num'] = configs['nn_hidden_neuron_num']
        else:
            self.configs['nn_hidden_neuron_num'] = 2
        
        if 'new_site' in configs:
            self.configs['new_site'] = configs['new_site']
            if 'newsite_training_idx' in configs:
                self.configs['newsite_training_idx'] = configs['newsite_training_idx']
            else:
                self.configs['newsite_training_idx'] = np.ones([configs['batch_effects_test'].shape[0]])
        else:
            self.configs['new_site'] = False
            
        if y is not None:
            self.hbr = HBR(np.squeeze(X), 
                           np.squeeze(batch_effects_train[:, 0]), 
                           np.squeeze(batch_effects_train[:, 1]), 
                           np.squeeze(y), self.configs)
        
    @property
    def n_params(self):
        return 1
    
    @property
    def neg_log_lik(self):
        return -1
    
    def estimate(self, X, y=None):
        self.hbr.estimate()
        return self
    
    def predict(self, Xs, X=None, Y=None, theta=None): 
        with open(self.configparam, 'rb') as handle:
             configparam = pickle.load(handle)
             
        batch_effects_test = configparam['batch_effects_test']
        if 'prediction' in configparam:
            pred_type = configparam['prediction']
        else:
            pred_type = 'single'
            
        yhat, s2 = self.hbr.predict(np.squeeze(Xs), 
                                    np.squeeze(batch_effects_test[:, 0]), 
                                    np.squeeze(batch_effects_test[:, 1]), pred = pred_type)      
        return yhat, s2
    
    def estimate_on_new_sites(self, X, y):
        with open(self.configparam, 'rb') as handle:
             configparam = pickle.load(handle)
        newsite_training_idx = np.where(configparam['newsite_training_idx'] == 1)
        sites =  configparam['batch_effects_test'][newsite_training_idx,0].squeeze()
        gender =  configparam['batch_effects_test'][newsite_training_idx,1].squeeze()
        self.hbr.estimate_on_new_site(X[newsite_training_idx,].squeeze(), sites,
                                      gender, y[newsite_training_idx,].squeeze())
        return self
    
    def predict_on_new_sites(self, X): # For the limitations in normative.py, this predicts on all test data.
        with open(self.configparam, 'rb') as handle:
             configparam = pickle.load(handle)
        gender =  configparam['batch_effects_test'][:,1].squeeze()
        sites =  configparam['batch_effects_test'][:,0].squeeze()
        yhat, s2 = self.hbr.predict_on_new_site(X.squeeze(), sites, gender)
        return yhat, s2
    