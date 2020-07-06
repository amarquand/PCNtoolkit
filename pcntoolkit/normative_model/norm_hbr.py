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

try:  # run as a package if installed
    from pcntoolkit import fileio
    from pcntoolkit.normative_model.normbase import NormBase
    from pcntoolkit.hbr import HBR 
except ImportError:
    pass

    path = os.path.abspath(os.path.dirname(__file__))
    if path not in sys.path:
        sys.path.append(path)
    del path
    import fileio
    from hbr import HBR
    from norm_base import NormBase

class NormHBR(NormBase):
    """ Classical GPR-based normative modelling approach
    """

    def __init__(self, **kwargs):
        
        self.configs = dict()
        self.configs['trbefile'] = kwargs.pop('trbefile',None) 
        self.configs['tsbefile'] = kwargs.pop('tsbefile',None) 
        self.configs['type'] = kwargs.pop('model_type', 'linear')
        self.configs['skewed_likelihood'] = kwargs.pop('skewed_likelihood', 'False') == 'True'
        self.configs['pred_type'] = kwargs.pop('pred_type', 'single')
        self.configs['random_noise'] = kwargs.pop('random_noise', 'True') == 'True'
        self.configs['n_samples'] = int(kwargs.pop('n_samples', '1000'))
        self.configs['n_tuning'] = int(kwargs.pop('n_tuning', '500'))
        self.configs['n_chains'] = int(kwargs.pop('n_chains', '1'))
        self.configs['target_accept'] = float(kwargs.pop('target_accept', '0.8'))
        self.configs['init'] = kwargs.pop('init', 'jitter+adapt_diag')
        self.configs['cores'] = int(kwargs.pop('cores', '1'))
        
        if self.configs['type'] == 'bspline':
            self.configs['order'] = int(kwargs.pop('order', '3'))
            self.configs['nknots'] = int(kwargs.pop('nknots', '5'))
            self.configs['random_intercept'] = kwargs.pop('random_intercept', 'True') == 'True'
            self.configs['random_slope'] = kwargs.pop('random_slope', 'True') == 'True'
        elif self.configs['type'] == 'polynomial':
            self.configs['order'] = int(kwargs.pop('order', '3'))
            self.configs['random_intercept'] = kwargs.pop('random_intercept', 'True') == 'True'
            self.configs['random_slope'] = kwargs.pop('random_slope', 'True') == 'True'
        elif self.configs['type'] == 'nn':
            self.configs['nn_hidden_neuron_num'] = int(kwargs.pop('nn_hidden_neuron_num', '2'))
            self.configs['nn_hidden_layers_num'] = int(kwargs.pop('nn_hidden_layers_num', '2'))
            if self.configs['nn_hidden_layers_num'] > 2:
                raise ValueError("Using " + str(self.configs['nn_hidden_layers_num']) \
                                 + " layers was not implemented. The number of " \
                                 + " layers has to be less than 3.")
        elif self.configs['type'] == 'linear':
            self.configs['random_intercept'] = kwargs.pop('random_intercept', 'True') == 'True'
            self.configs['random_slope'] = kwargs.pop('random_slope', 'True') == 'True'
        else:
            raise ValueError("Unknown model type, please specify from 'linear', \
                             'polynomial', 'bspline', or 'nn'.")
            
        if self.configs['random_noise']:
            self.configs['hetero_noise'] = kwargs.pop('hetero_noise', 'False') == 'True'
            
        self.hbr = HBR(self.configs)
        
    @property
    def n_params(self):
        return 1
    
    @property
    def neg_log_lik(self):
        return -1
    
    
    def estimate(self, X, y, **kwargs):
        
        trbefile = kwargs.pop('trbefile', None) 
        if trbefile is not None:
            batch_effects_train = fileio.load(trbefile)
        else:
            print('Could not find batch-effects file! Initilizing all as zeros ...')
            batch_effects_train = np.zeros([X.shape[0],1])

        self.hbr.estimate(X, y, batch_effects_train)
        
        return self
    
    
    def predict(self, Xs, X=None, Y=None, **kwargs): 
        
        tsbefile = kwargs.pop('tsbefile', None) 
        if tsbefile is not None:
            batch_effects_test = fileio.load(tsbefile)
        else:
            print('Could not find batch-effects file! Initilizing all as zeros ...')
            batch_effects_test = np.zeros([Xs.shape[0],1])    
        
        pred_type = self.configs['pred_type']
        
        yhat, s2 = self.hbr.predict(Xs, batch_effects_test, pred = pred_type)     
        
        return yhat.squeeze(), s2.squeeze()
    
    
    def estimate_on_new_sites(self, X, y, batch_effects):
    
        self.hbr.estimate_on_new_site(X, y, batch_effects)
        return self
    
    
    def predict_on_new_sites(self, X, batch_effects):
    
        yhat, s2 = self.hbr.predict_on_new_site(X, batch_effects)
        return yhat, s2
    
    
    def extend(self, X, y, batch_effects, X_dummy, batch_effects_dummy, 
               samples=10, informative_prior=False):
        
        X_dummy, batch_effects_dummy, Y_dummy = self.hbr.generate(X_dummy, 
                                                batch_effects_dummy, samples)
        if informative_prior:
            self.hbr.estimate_on_new_sites(np.concatenate((X_dummy, X)), 
                         np.concatenate((Y_dummy, y)), 
                         np.concatenate((batch_effects_dummy, batch_effects)))
        else:
            self.hbr.estimate(np.concatenate((X_dummy, X)), 
                         np.concatenate((Y_dummy, y)), 
                         np.concatenate((batch_effects_dummy, batch_effects)))
        
        return self
    
    
    def generate(self, X, batch_effects, samples=10):
        
        X, batch_effects, generated_samples = self.hbr.generate(X, batch_effects, 
                                                                samples)
        return X, batch_effects, generated_samples
    