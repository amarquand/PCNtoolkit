#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 25 17:01:24 2019

@author: seykia
@author: augub
"""

from __future__ import print_function
from __future__ import division


import os
import warnings
import sys
import numpy as np
from ast import literal_eval as make_tuple

try:
    from pcntoolkit.dataio import fileio
    from pcntoolkit.normative_model.norm_base import NormBase
    from pcntoolkit.model.hbr import HBR
except ImportError:
    pass

    path = os.path.abspath(os.path.dirname(__file__))
    if path not in sys.path:
        sys.path.append(path)
    del path
    import dataio.fileio as fileio
    from model.hbr import HBR
    from norm_base import NormBase


class NormHBR(NormBase):
    
    """ HBR multi-batch normative modelling class
    
    :param X: [N×P] array of clinical covariates
    :param y: [N×1] array of neuroimaging measures
    :param trbefile: the address to the batch effects file for the training set.
        the batch effect array should be a [N×M] array where M is the number of
        the type of batch effects. For example when the site and gender is modeled
        as batch effects M=2. Each column in the batch effect array contains the 
        batch ID (starting from 0) for each sample. If not specified (default=None)
        then all samples assumed to be from the same batch (i.e., the batch effect 
                                                            is not modelled).
    :param tsbefile: Similar to trbefile for the test set.
    :param model_type: Specifies the type of the model from 'linear', 'plynomial',
        and 'bspline' (defauls is 'linear').
    :param likelihood: specifies the type of likelihood among 'Normal' 'SHASHb','SHASHo',
        and 'SHASHo2' (defauls is normal).
    :param sampler: specifies the type of PyMC sampler (Defauls is 'NUTS').
    :param n_samples: The number of samples to draw (Default is '1000'). Please 
        note that this parameter must be specified in a string fromat ('1000' and 
                                                                       not 1000).  
    :param n_tuning: String that specifies the number of iterations to adjust 
        the samplers's step sizes, scalings or similar (defauls is '500').
    :param n_chains: String that specifies the number of chains to sample. Defauls 
        is '1' for faster estimation, but note that sampling independent chains 
        is important for some convergence checks.
    :param cores: String that specifies the number of chains to run in parallel.
        (defauls is '1').
    :param Initialization method to use for auto-assigned NUTS samplers. The
        defauls is 'jitter+adapt_diag' that starts with a identity mass matrix 
        and then adapt a diagonal based on the variance of the tuning samples
        while adding a uniform jitter in [-1, 1] to the starting point in each chain.
    :param target_accept: String that of a float in [0, 1] that regulates the 
        step size such that we approximate this acceptance rate. The defauls is '0.8'
        but higher values like 0.9 or 0.95 often work better for problematic posteriors.
    :param order: String that defines the order of bspline or polynomial model.
        The defauls is '3'.
    :param nknots: String that defines the numbers of knots for the bspline model.
        The defauls is '5'. Higher values increase the model complexity with negative 
        effect on the spped of estimations.
    :param nn_hidden_layers_num: String the specifies the number of hidden layers
        in neural network model. It can be either '1' or '2'. The default is set to '2'.
    :param nn_hidden_neuron_num: String that specifies the number of neurons in 
        the hidden layers. The defauls is set to  '2'.
    
    """

    def __init__(self, **kwargs):

        self.configs = dict()
        self.configs['trbefile'] = kwargs.pop('trbefile', None)
        self.configs['tsbefile'] = kwargs.pop('tsbefile', None)
        self.configs['type'] = kwargs.pop('model_type', 'linear')
        self.configs['likelihood'] = kwargs.pop('likelihood', 'Normal')
        self.configs['sampler'] = kwargs.pop('sampler', 'NUTS')
        self.configs['n_samples'] = int(kwargs.pop('n_samples', '1000'))
        self.configs['n_tuning'] = int(kwargs.pop('n_tuning', '500'))
        self.configs['n_chains'] = int(kwargs.pop('n_chains', '1'))
        self.configs['cores'] = int(kwargs.pop('cores', '1'))
        self.configs['init'] = kwargs.pop('init', 'jitter+adapt_diag')
        self.configs['target_accept'] = float(kwargs.pop('target_accept', '0.8'))
        self.configs['freedom'] = int(kwargs.pop('freedom', '1'))
        self.configs['pred_type'] = kwargs.pop('pred_type', 'single') # ???
        self.configs['transferred'] = False # Specifies whether this model is transferred or not (is always False when initializing a new model).
        
        ### Get deprecated soon
        #self.configs['skewed_likelihood'] = kwargs.pop('skewed_likelihood', 'False') == 'True'
        #self.configs['random_noise'] = kwargs.pop('random_noise', 'True') == 'True'
        #self.configs['hetero_noise'] = kwargs.pop('hetero_noise', 'False') == 'True'
        ### End of deprecation section
        
        if self.configs['type'] == 'bspline':
            self.configs['order'] = int(kwargs.pop('order', '3'))
            self.configs['nknots'] = int(kwargs.pop('nknots', '5'))
        elif self.configs['type'] == 'polynomial':
            self.configs['order'] = int(kwargs.pop('order', '3'))
        elif self.configs['type'] == 'nn':
            self.configs['nn_hidden_layers_num'] = int(kwargs.pop('nn_hidden_layers_num', '2'))
            self.configs['nn_hidden_neuron_num'] = int(kwargs.pop('nn_hidden_neuron_num', '2'))
            if self.configs['nn_hidden_layers_num'] > 2 or self.configs['nn_hidden_layers_num'] <= 0:
                raise ValueError("Using " + str(self.configs['nn_hidden_layers_num']) \
                                 + " layers was not implemented. The number of " \
                                 + " layers has to be either 1 or 2.")
        elif self.configs['type'] == 'linear':
            pass
        else:
            raise ValueError("Unknown model type, please specify from 'linear', \
                             'polynomial', 'bspline', or 'nn'.")

        if self.configs['type'] in ['bspline', 'polynomial', 'linear']:
            
            self.configs['centered'] = kwargs.pop('centered', 'False') == 'True'
            self.configs['random_mu'] = kwargs.pop('random_mu', 'True') == 'True'
            ######## Deprecations (remove in later version)
            for c in ['mu_linear', 'linear_mu', 'random_intercept', 'random_slope']:
                if c in kwargs.keys():
                    print(f'The keyword {c} is deprecated. It is automatically replaced with random_mu.')
                    self.configs['random_mu'] = kwargs.pop(c, 'True') == 'True'
            ##### End Deprecations
            if self.configs['random_mu']:
                self.configs['random_mu_intercept'] = kwargs.pop('random_mu_intercept', 'True') == 'True'
                self.configs['random_mu_slope'] = kwargs.pop('random_mu_slope', 'True') == 'True'
            
            
            self.configs['random_sigma'] = kwargs.pop('random_sigma', 'True') == 'True'
            ######## Deprecations (remove in later version)
            for c in ['sigma_linear', 'linear_sigma', 'sigma_intercept', 'random_slope']:
                if c in kwargs.keys():
                    print(f'The keyword {c} is deprecated. It is automatically replaced with random_mu.')
                    self.configs['random_sigma'] = kwargs.pop(c, 'True') == 'True'
            ##### End Deprecations
            if self.configs['random_sigma']:
                self.configs['random_sigma_intercept'] = kwargs.pop('random_sigma_intercept', 'False') == 'True'
                self.configs['random_sigma_slope'] = kwargs.pop('random_sigma_slope', 'False') == 'True'
            
            
            for p in ['epsilon', 'delta']:
                self.configs[f'random_{p}'] = kwargs.pop(f'random_{p}', 'False') == 'True'

                ######## Deprecations (remove in later version)
                for c in [f'{p}_linear', f'linear_{p}','random_noise']:
                    if c in kwargs.keys():
                        print(f'The keywords {c} is deprecated. It is automatically replaced with \'random_{p}\'')
                        self.configs[f'random_{p}'] = kwargs.pop(c, 'True') == 'True'
                ##### End Deprecations 
                
                if self.configs[f'random_{p}']:
                    self.configs[f'random_{p}_intercept'] = kwargs.pop(f'random_{p}_intercept', 'True') == 'True'
                    self.configs[f'random_{p}_slope'] = kwargs.pop(f'random_{p}_slope', 'True') == 'True'
            
                #for c in ['centered','random']:
                #    self.configs[f'{c}_{p}'] = kwargs.pop(f'{c}_{p}', 'False') == 'True'
                #    for sp in ['slope','intercept']:
                #        self.configs[f'{c}_{sp}_{p}'] = kwargs.pop(f'{c}_{sp}_{p}', 'False') == 'True'

            ######## Deprecations (remove in later version)
            #if self.configs['linear_sigma']:
            #    if 'random_noise' in kwargs.keys():
            #        print("The keyword \'random_noise\' is deprecated. It is now automatically replaced with \'random_intercept_sigma\', because sigma is linear")
            #        self.configs['random_intercept_sigma'] = kwargs.pop('random_noise','False') == 'True'
            #elif 'random_noise' in kwargs.keys():
            #    print("The keyword \'random_noise\' is deprecated. It is now automatically replaced with \'random_sigma\', because sigma is fixed")
            #    self.configs['random_sigma'] = kwargs.pop('random_noise','False') == 'True'
            #if 'random_slope' in kwargs.keys():
            #    print("The keyword \'random_slope\' is deprecated. It is now automatically replaced with \'random_slope_mu\'")
            #    self.configs['random_slope_mu'] = kwargs.pop('random_slope','False') == 'True'
            ##### End Deprecations 


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
            batch_effects_train = np.zeros([X.shape[0], 1])

        self.hbr.estimate(X, y, batch_effects_train)

        return self

    def predict(self, Xs, X=None, Y=None, **kwargs):

        tsbefile = kwargs.pop('tsbefile', None)
        if tsbefile is not None:
            batch_effects_test = fileio.load(tsbefile)
        else:
            print('Could not find batch-effects file! Initilizing all as zeros ...')
            batch_effects_test = np.zeros([Xs.shape[0], 1])

        pred_type = self.configs['pred_type']

        if self.configs['transferred'] == False:
            yhat, s2 = self.hbr.predict(Xs, batch_effects_test, pred=pred_type)
        else:
            raise ValueError("This is a transferred model. Please use predict_on_new_sites function.")

        return yhat.squeeze(), s2.squeeze()

    def estimate_on_new_sites(self, X, y, batch_effects):
        self.hbr.estimate_on_new_site(X, y, batch_effects)
        self.configs['transferred'] = True
        return self

    def predict_on_new_sites(self, X, batch_effects):

        yhat, s2 = self.hbr.predict_on_new_site(X, batch_effects)
        return yhat, s2

    def extend(self, X, y, batch_effects, X_dummy_ranges=[[0.1, 0.9, 0.01]],
               merge_batch_dim=0, samples=10, informative_prior=False):

        X_dummy, batch_effects_dummy = self.hbr.create_dummy_inputs(X_dummy_ranges)

        X_dummy, batch_effects_dummy, Y_dummy = self.hbr.generate(X_dummy,
                                                                  batch_effects_dummy, samples)

        batch_effects[:, merge_batch_dim] = batch_effects[:, merge_batch_dim] + \
                                            np.max(batch_effects_dummy[:, merge_batch_dim]) + 1

        if informative_prior:
            self.hbr.adapt(np.concatenate((X_dummy, X)),
                           np.concatenate((Y_dummy, y)),
                           np.concatenate((batch_effects_dummy, batch_effects)))
        else:
            self.hbr.estimate(np.concatenate((X_dummy, X)),
                              np.concatenate((Y_dummy, y)),
                              np.concatenate((batch_effects_dummy, batch_effects)))

        return self

    def tune(self, X, y, batch_effects, X_dummy_ranges=[[0.1, 0.9, 0.01]],
             merge_batch_dim=0, samples=10, informative_prior=False):

        tune_ids = list(np.unique(batch_effects[:, merge_batch_dim]))

        X_dummy, batch_effects_dummy = self.hbr.create_dummy_inputs(X_dummy_ranges)

        for idx in tune_ids:
            X_dummy = X_dummy[batch_effects_dummy[:, merge_batch_dim] != idx, :]
            batch_effects_dummy = batch_effects_dummy[batch_effects_dummy[:, merge_batch_dim] != idx, :]

        X_dummy, batch_effects_dummy, Y_dummy = self.hbr.generate(X_dummy,
                                                                  batch_effects_dummy, samples)

        if informative_prior:
            self.hbr.adapt(np.concatenate((X_dummy, X)),
                           np.concatenate((Y_dummy, y)),
                           np.concatenate((batch_effects_dummy, batch_effects)))
        else:
            self.hbr.estimate(np.concatenate((X_dummy, X)),
                              np.concatenate((Y_dummy, y)),
                              np.concatenate((batch_effects_dummy, batch_effects)))

        return self

    def merge(self, nm, X_dummy_ranges=[[0.1, 0.9, 0.01]], merge_batch_dim=0,
              samples=10):

        X_dummy1, batch_effects_dummy1 = self.hbr.create_dummy_inputs(X_dummy_ranges)
        X_dummy2, batch_effects_dummy2 = nm.hbr.create_dummy_inputs(X_dummy_ranges)

        X_dummy1, batch_effects_dummy1, Y_dummy1 = self.hbr.generate(X_dummy1,
                                                                     batch_effects_dummy1, samples)
        X_dummy2, batch_effects_dummy2, Y_dummy2 = nm.hbr.generate(X_dummy2,
                                                                   batch_effects_dummy2, samples)

        batch_effects_dummy2[:, merge_batch_dim] = batch_effects_dummy2[:, merge_batch_dim] + \
                                                   np.max(batch_effects_dummy1[:, merge_batch_dim]) + 1

        self.hbr.estimate(np.concatenate((X_dummy1, X_dummy2)),
                          np.concatenate((Y_dummy1, Y_dummy2)),
                          np.concatenate((batch_effects_dummy1,
                                          batch_effects_dummy2)))

        return self

    def generate(self, X, batch_effects, samples=10):

        X, batch_effects, generated_samples = self.hbr.generate(X, batch_effects,
                                                                samples)
        return X, batch_effects, generated_samples
