#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 25 13:23:15 2019

@author: seykia
"""

from __future__ import print_function
from __future__ import division

import numpy as np
import pymc3 as pm
import theano
from itertools import product
from functools import reduce
from scipy import stats
import bspline
from bspline import splinelab



def bspline_fit(X, order, nknots):
    
    X = X.squeeze()
    if len(X.shape) > 1:
        raise ValueError('Bspline method only works for a single covariate.')
    
    knots = np.linspace(X.min(), X.max(), nknots)
    k = splinelab.augknt(knots, order)
    bsp_basis = bspline.Bspline(k, order)
    
    return bsp_basis

def bspline_transform(X, bsp_basis):
    
    X = X.squeeze()
    if len(X.shape) > 1:
        raise ValueError('Bspline method only works for a single covariate.')
        
    X_transformed = np.array([bsp_basis(i) for i in X])
    
    return X_transformed


def from_posterior(param, samples, distribution = None, half = False, freedom=100):
    
    if len(samples.shape)>1:
        shape = samples.shape[1:]
    else:
        shape = None
            
    if (distribution is None):
        smin, smax = np.min(samples), np.max(samples)
        width = smax - smin
        x = np.linspace(smin, smax, 1000)
        y = stats.gaussian_kde(samples)(x)
        if half:
            x = np.concatenate([x, [x[-1] + 0.05 * width]])
            y = np.concatenate([y, [0]])
        else:
            x = np.concatenate([[x[0] - 0.05 * width], x, [x[-1] + 0.05 * width]])
            y = np.concatenate([[0], y, [0]])
        return pm.distributions.Interpolated(param, x, y)
    elif (distribution=='normal'):
        temp = stats.norm.fit(samples)
        if shape is None:
            return pm.Normal(param, mu=temp[0], sigma=freedom*temp[1])
        else:
            return pm.Normal(param, mu=temp[0], sigma=freedom*temp[1], shape=shape)
    elif (distribution=='hnormal'):
        temp = stats.halfnorm.fit(samples)
        if shape is None:
            return pm.HalfNormal(param, sigma=freedom*temp[1])
        else:
            return pm.HalfNormal(param, sigma=freedom*temp[1], shape=shape)
    elif (distribution=='hcauchy'):
        temp = stats.halfcauchy.fit(samples)
        if shape is None:
            return pm.HalfCauchy(param, freedom*temp[1])
        else:
            return pm.HalfCauchy(param, freedom*temp[1], shape=shape)
    

def linear_hbr(X, y, batch_effects, batch_effects_size, configs, trace=None):
    
    feature_num = X.shape[1]
    y_shape = y.shape
    batch_effects_num = batch_effects.shape[1]
    all_idx = []
    for i in range(batch_effects_num):
        all_idx.append(np.int16(np.unique(batch_effects[:,i])))
    be_idx = list(product(*all_idx))
    
    X = theano.shared(X)
    y = theano.shared(y)
    
    with pm.Model() as model:
        # Priors
        if trace is not None: # Used for transferring the priors
            mu_prior_intercept = from_posterior('mu_prior_intercept', 
                                                    trace['mu_prior_intercept'], 
                                                    distribution='normal')
            sigma_prior_intercept = from_posterior('sigma_prior_intercept', 
                                                   trace['sigma_prior_intercept'], 
                                                   distribution='hcauchy')
            mu_prior_slope = from_posterior('mu_prior_slope', 
                                            trace['mu_prior_slope'], 
                                            distribution='normal')
            sigma_prior_slope = from_posterior('sigma_prior_slope', 
                                               trace['sigma_prior_slope'], 
                                               distribution='hcauchy')
        else:
            mu_prior_intercept = pm.Normal('mu_prior_intercept', mu=0., sigma=1e5)
            sigma_prior_intercept = pm.HalfCauchy('sigma_prior_intercept', 5)
            mu_prior_slope = pm.Normal('mu_prior_slope', mu=0., sigma=1e5, shape=(feature_num,))
            sigma_prior_slope = pm.HalfCauchy('sigma_prior_slope', 5, shape=(feature_num,))
        
        if configs['random_intercept']: 
            intercepts_offset = pm.Normal('intercepts_offset', mu=0, sd=1, 
                                          shape=(batch_effects_size))
        else:
            intercepts_offset = pm.Normal('intercepts_offset', mu=0, sd=1)
       
        intercepts = pm.Deterministic('intercepts', mu_prior_intercept + 
                                      intercepts_offset * sigma_prior_intercept)
        
        if configs['random_slope']:  # Random slopes
            slopes_offset = pm.Normal('slopes_offset', mu=0, sd=1, 
                                     shape=(batch_effects_size + [feature_num]))
        else:
            slopes_offset = pm.Normal('slopes_offset', mu=0, sd=1)
            
        slopes = pm.Deterministic('slopes', mu_prior_slope + 
                                          slopes_offset * sigma_prior_slope)
        
        y_hat = theano.tensor.zeros(y_shape)
        for be in be_idx:
            a = []
            for i, b in enumerate(be):
                a.append(batch_effects[:,i]==b) 
            idx = reduce(np.logical_and, a).nonzero()
            if idx[0].shape[0] != 0:
                if (not configs['random_intercept'] and not configs['random_slope']):
                    y_hat = theano.tensor.set_subtensor(y_hat[idx,0], intercepts + theano.tensor.dot(X[idx,:], 
                                     slopes))
                elif (configs['random_intercept'] and not configs['random_slope']):
                    y_hat = theano.tensor.set_subtensor(y_hat[idx,0], intercepts[be] + theano.tensor.dot(X[idx,:], 
                                     slopes))
                elif (not configs['random_intercept'] and configs['random_slope']):
                    y_hat = theano.tensor.set_subtensor(y_hat[idx,0], intercepts + theano.tensor.dot(X[idx,:], 
                                     slopes[be]))
                elif (configs['random_intercept'] and configs['random_slope']):        
                    y_hat = theano.tensor.set_subtensor(y_hat[idx,0], intercepts[be] + theano.tensor.dot(X[idx,:], 
                                     slopes[be]))
        
        if configs['random_noise']:
            if configs['hetero_noise']:
                # Priors
                if trace is not None: # Used for transferring the priors
                    mu_prior_intercept_noise = from_posterior('mu_prior_intercept_noise', 
                                                            trace['mu_prior_intercept_noise'], 
                                                            distribution='normal')
                    sigma_prior_intercept_noise = from_posterior('sigma_prior_intercept_noise', 
                                                           trace['sigma_prior_intercept_noise'], 
                                                           distribution='hcauchy')
                    mu_prior_slope_noise = from_posterior('mu_prior_slope_noise', 
                                                    trace['mu_prior_slope_noise'], 
                                                    distribution='normal')
                    sigma_prior_slope_noise = from_posterior('sigma_prior_slope_noise', 
                                                       trace['sigma_prior_slope_noise'], 
                                                       distribution='hcauchy')
                else:
                    mu_prior_intercept_noise = pm.HalfNormal('mu_prior_intercept_noise', 
                                                             sigma=1e5)
                    sigma_prior_intercept_noise = pm.HalfCauchy('sigma_prior_intercept_noise', 5)
                    mu_prior_slope_noise = pm.Normal('mu_prior_slope_noise',  mu=0., 
                                                     sigma=1e5, shape=(feature_num,))
                    sigma_prior_slope_noise = pm.HalfCauchy('sigma_prior_slope_noise', 
                                                            5, shape=(feature_num,))
                if configs['random_intercept']: 
                    intercepts_noise_offset = pm.HalfNormal('intercepts_noise_offset',
                                                        sd=1, shape=(batch_effects_size))
                else:
                    intercepts_noise_offset = pm.HalfNormal('intercepts_noise_offset',
                                                                    sd=1)
                    
                intercepts_noise = pm.Deterministic('intercepts_noise',
                                                    mu_prior_intercept_noise + 
                                          intercepts_noise_offset * sigma_prior_intercept_noise)
    
                if configs['random_slope']:
                    slopes_noise_offset = pm.Normal('slopes_noise_offset', mu=0, sd=1, 
                                     shape=(batch_effects_size + [feature_num]))
                else:
                    slopes_noise_offset = pm.Normal('slopes_noise_offset', mu=0, sd=1)
                    
                slopes_noise = pm.Deterministic('slopes_noise', mu_prior_slope_noise + 
                                          slopes_noise_offset * sigma_prior_slope_noise)
                
                sigma_noise = theano.tensor.zeros(y_shape)
                for be in be_idx:
                    a = []
                    for i, b in enumerate(be):
                        a.append(batch_effects[:,i]==b)        
                    idx = reduce(np.logical_and, a).nonzero()
                    if idx[0].shape[0]!=0:
                        if (not configs['random_intercept'] and not configs['random_slope']):
                            sigma_noise = theano.tensor.set_subtensor(sigma_noise[idx,0], 
                                                   intercepts_noise + theano.tensor.dot(X[idx,:], 
                                                                   slopes_noise))
                        elif (configs['random_intercept'] and not configs['random_slope']):
                            sigma_noise = theano.tensor.set_subtensor(sigma_noise[idx,0], 
                                                   intercepts_noise[be] + theano.tensor.dot(X[idx,:], 
                                                                   slopes_noise))
                        elif (not configs['random_intercept'] and configs['random_slope']):
                            sigma_noise = theano.tensor.set_subtensor(sigma_noise[idx,0], 
                                                   intercepts_noise + theano.tensor.dot(X[idx,:], 
                                                                   slopes_noise[be]))
                        elif (configs['random_intercept'] and configs['random_slope']):        
                            sigma_noise = theano.tensor.set_subtensor(sigma_noise[idx,0], 
                                                   intercepts_noise[be] + theano.tensor.dot(X[idx,:], 
                                                                   slopes_noise[be]))
                              
                sigma_y = pm.Deterministic('sigma_y', pm.math.log1pexp(sigma_noise) + 1e-5)
                        
            else:
                sigma_noise = pm.Uniform('sigma_noise', lower=0, upper=100, shape=(batch_effects_size))
                sigma_y = theano.tensor.zeros(y_shape)
                for be in be_idx:
                    a = []
                    for i, b in enumerate(be):
                        a.append(batch_effects[:,i]==b)             
                    idx = reduce(np.logical_and, a).nonzero()
                    if idx[0].shape[0]!=0:
                        sigma_y = theano.tensor.set_subtensor(sigma_y[idx,0], sigma_noise[be])
        
        else:
            sigma_noise = pm.Uniform('sigma_noise', lower=0, upper=100)
            sigma_y = theano.tensor.zeros(y_shape)
            for be in be_idx:
                a = []
                for i, b in enumerate(be):
                    a.append(batch_effects[:,i]==b)
                    
                idx = reduce(np.logical_and, a).nonzero()
                if idx[0].shape[0]!=0:
                    sigma_y = theano.tensor.set_subtensor(sigma_y[idx,0], sigma_noise)
            
        if configs['skewed_likelihood']:
            skewness = pm.Uniform('skewness', lower=-10, upper=10, shape=(batch_effects_size))
            alpha = theano.tensor.zeros(y_shape)
            for be in be_idx:
                a = []
                for i, b in enumerate(be):
                    a.append(batch_effects[:,i]==b)             
                idx = reduce(np.logical_and, a).nonzero()
                if idx[0].shape[0]!=0:
                    alpha = theano.tensor.set_subtensor(alpha[idx,0], skewness[be])
        else:
            alpha = 0
        
        y_like = pm.SkewNormal('y_like', mu=y_hat, sigma=sigma_y, alpha=alpha, observed=y)

    return model


def poly2_hbr(X, y, batch_effects, batch_effects_size, configs, trace=None):
    
    feature_num = X.shape[1]
    y_shape = y.shape
    batch_effects_num = batch_effects.shape[1]
    all_idx = []
    for i in range(batch_effects_num):
        all_idx.append(np.int16(np.unique(batch_effects[:,i])))
    be_idx = list(product(*all_idx))
    
    X = theano.shared(X)
    y = theano.shared(y)
    
    with pm.Model() as model:
        # Priors
        if trace is not None: # Used for transferring the priors
            mu_prior_intercept = from_posterior('mu_prior_intercept', 
                                                    trace['mu_prior_intercept'],
                                                    distribution='normal')
            sigma_prior_intercept = from_posterior('sigma_prior_intercept', 
                                                   trace['sigma_prior_intercept'], 
                                                   distribution='hcauchy')
            mu_prior_slope_1 = from_posterior('mu_prior_slope_1', 
                                              trace['mu_prior_slope_1'],
                                              distribution='normal')
            sigma_prior_slope_1 = from_posterior('sigma_prior_slope_1', 
                                                 trace['sigma_prior_slope_1'], 
                                                 distribution='hcauchy')
            mu_prior_slope_2 = from_posterior('mu_prior_slope_2', 
                                              trace['mu_prior_slope_2'],
                                              distribution='normal')
            sigma_prior_slope_2 = from_posterior('sigma_prior_slope_2', 
                                                 trace['sigma_prior_slope_2'], 
                                                 distribution='hcauchy')
        else:
            mu_prior_intercept = pm.Normal('mu_prior_intercept', mu=0., sigma=1e5)
            sigma_prior_intercept = pm.HalfCauchy('sigma_prior_intercept', 5)
            mu_prior_slope_1 = pm.Normal('mu_prior_slope_1', mu=0., sigma=1e5, shape=(feature_num,))
            sigma_prior_slope_1 = pm.HalfCauchy('sigma_prior_slope_1', 5, shape=(feature_num,))
            mu_prior_slope_2 = pm.Normal('mu_prior_slope_2', mu=0., sigma=1e5, shape=(feature_num,))
            sigma_prior_slope_2 = pm.HalfCauchy('sigma_prior_slope_2', 5, shape=(feature_num,))
    
        if configs['random_intercept']: 
            intercepts_offset = pm.Normal('intercepts_offset', mu=0, sd=1, 
                                          shape=(batch_effects_size))
        else:
            intercepts_offset = pm.Normal('intercepts_offset', mu=0, sd=1)
       
        intercepts = pm.Deterministic('intercepts', mu_prior_intercept + 
                                      intercepts_offset * sigma_prior_intercept)
        
        if configs['random_slope']:  # Random slopes
            slopes_offset1 = pm.Normal('slopes_offset1', mu=0, sd=1, 
                                     shape=(batch_effects_size + [feature_num]))
            slopes_offset2 = pm.Normal('slopes_offset2', mu=0, sd=1, 
                                     shape=(batch_effects_size + [feature_num]))
        else:
            slopes_offset1 = pm.Normal('slopes_offset1', mu=0, sd=1)
            slopes_offset2 = pm.Normal('slopes_offset2', mu=0, sd=1)
            
        slopes_1 = pm.Deterministic('slopes_1', mu_prior_slope_1 + 
                                          slopes_offset1 * sigma_prior_slope_1)
        slopes_2 = pm.Deterministic('slopes_2', mu_prior_slope_2 + 
                                          slopes_offset2 * sigma_prior_slope_2)
        
        y_hat = theano.tensor.zeros(y_shape)
        for be in be_idx:
            a = []
            for i, b in enumerate(be):
                a.append(batch_effects[:,i]==b) 
            idx = reduce(np.logical_and, a).nonzero()
            if idx[0].shape[0] != 0:
                if (not configs['random_intercept'] and not configs['random_slope']):
                    y_hat = theano.tensor.set_subtensor(y_hat[idx,0], intercepts + \
                                                        theano.tensor.dot(X[idx,:], slopes_1) + \
                                                        theano.tensor.dot(theano.tensor.sqr(X[idx,:]), slopes_2))
                elif (configs['random_intercept'] and not configs['random_slope']):
                    y_hat = theano.tensor.set_subtensor(y_hat[idx,0], intercepts[be] + \
                                                        theano.tensor.dot(X[idx,:], slopes_1) + \
                                                        theano.tensor.dot(theano.tensor.sqr(X[idx,:]), slopes_2))
                elif (not configs['random_intercept'] and configs['random_slope']):
                    y_hat = theano.tensor.set_subtensor(y_hat[idx,0], intercepts + \
                                                        theano.tensor.dot(X[idx,:], slopes_1[be]) + \
                                                        theano.tensor.dot(theano.tensor.sqr(X[idx,:]), slopes_2[be]))
                elif (configs['random_intercept'] and configs['random_slope']):        
                    y_hat = theano.tensor.set_subtensor(y_hat[idx,0], intercepts[be] + \
                                                        theano.tensor.dot(X[idx,:], slopes_1[be]) + \
                                                        theano.tensor.dot(theano.tensor.sqr(X[idx,:]), slopes_2[be]))
        
        if configs['random_noise']:
            if configs['hetero_noise']:
                # Priors
                if trace is not None: # Used for transferring the priors
                    mu_prior_intercept_noise = from_posterior('mu_prior_intercept_noise', 
                                                            trace['mu_prior_intercept_noise'],
                                                            distribution='normal')
                    sigma_prior_intercept_noise = from_posterior('sigma_prior_intercept_noise', 
                                                           trace['sigma_prior_intercept_noise'], 
                                                           distribution='hcauchy')
                    mu_prior_slope_1_noise = from_posterior('mu_prior_slope_1_noise', 
                                                      trace['mu_prior_slope_1_noise'],
                                                      distribution='normal')
                    sigma_prior_slope_1_noise = from_posterior('sigma_prior_slope_1_noise', 
                                                         trace['sigma_prior_slope_1_noise'], 
                                                         distribution='hcauchy')
                    mu_prior_slope_2_noise = from_posterior('mu_prior_slope_2_noise', 
                                                      trace['mu_prior_slope_2_noise'],
                                                      distribution='normal')
                    sigma_prior_slope_2_noise = from_posterior('sigma_prior_slope_2_noise', 
                                                         trace['sigma_prior_slope_2_noise'], 
                                                         distribution='hcauchy')
                else:
                    mu_prior_intercept_noise = pm.HalfNormal('mu_prior_intercept_noise', sigma=1e5)
                    sigma_prior_intercept_noise = pm.HalfCauchy('sigma_prior_intercept_noise', 5)
                    mu_prior_slope_1_noise = pm.Normal('mu_prior_slope_1_noise', mu=0., sigma=1e5, shape=(feature_num,))
                    sigma_prior_slope_1_noise = pm.HalfCauchy('sigma_prior_slope_1_noise', 5, shape=(feature_num,))
                    mu_prior_slope_2_noise = pm.Normal('mu_prior_slope_2_noise', mu=0., sigma=1e5, shape=(feature_num,))
                    sigma_prior_slope_2_noise = pm.HalfCauchy('sigma_prior_slope_2_noise', 5, shape=(feature_num,))
                    
                if configs['random_intercept']: 
                    intercepts_noise_offset = pm.HalfNormal('intercepts_noise_offset',
                                                        sd=1, shape=(batch_effects_size))
                else:
                    intercepts_noise_offset = pm.HalfNormal('intercepts_noise_offset',
                                                                    sd=1) 
                intercepts_noise = pm.Deterministic('intercepts_noise',
                                                    mu_prior_intercept_noise + 
                                          intercepts_noise_offset * sigma_prior_intercept_noise)
    
                if configs['random_slope']:
                    slopes_noise_offset1 = pm.Normal('slopes_noise_offset1', mu=0, sd=1, 
                                     shape=(batch_effects_size + [feature_num]))
                    slopes_noise_offset2 = pm.Normal('slopes_noise_offset2', mu=0, sd=1, 
                                     shape=(batch_effects_size + [feature_num]))
                else:
                    slopes_noise_offset1 = pm.Normal('slopes_noise_offset1', mu=0, sd=1)
                    slopes_noise_offset2 = pm.Normal('slopes_noise_offset2', mu=0, sd=1)
                    
                slopes_1_noise = pm.Deterministic('slopes_1_noise', mu_prior_slope_1_noise + 
                                                  slopes_noise_offset1 * sigma_prior_slope_1_noise)
                slopes_2_noise = pm.Deterministic('slopes_2_noise', mu_prior_slope_2_noise + 
                                                  slopes_noise_offset2 * sigma_prior_slope_2_noise)
                
                sigma_noise = theano.tensor.zeros(y_shape)
                for be in be_idx:
                    a = []
                    for i, b in enumerate(be):
                        a.append(batch_effects[:,i]==b)        
                    idx = reduce(np.logical_and, a).nonzero()
                    if idx[0].shape[0]!=0:
                        if (not configs['random_intercept'] and not configs['random_slope']):
                            sigma_noise = theano.tensor.set_subtensor(sigma_noise[idx,0], intercepts_noise + 
                                                   theano.tensor.dot(X[idx,:], slopes_1_noise) + \
                                                   theano.tensor.dot(theano.tensor.sqr(X[idx,:]), slopes_2_noise))
                        elif (configs['random_intercept'] and not configs['random_slope']):
                            sigma_noise = theano.tensor.set_subtensor(sigma_noise[idx,0], intercepts_noise[be] + 
                                                   theano.tensor.dot(X[idx,:], slopes_1_noise) + \
                                                   theano.tensor.dot(theano.tensor.sqr(X[idx,:]), slopes_2_noise))
                        elif (not configs['random_intercept'] and configs['random_slope']):
                            sigma_noise = theano.tensor.set_subtensor(sigma_noise[idx,0], intercepts_noise + 
                                                   theano.tensor.dot(X[idx,:], slopes_1_noise[be]) + \
                                                   theano.tensor.dot(theano.tensor.sqr(X[idx,:]), slopes_2_noise[be]))
                        elif (configs['random_intercept'] and configs['random_slope']):        
                            sigma_noise = theano.tensor.set_subtensor(sigma_noise[idx,0], intercepts_noise[be] + 
                                                   theano.tensor.dot(X[idx,:], slopes_1_noise[be]) + \
                                                   theano.tensor.dot(theano.tensor.sqr(X[idx,:]), slopes_2_noise[be]))
                              
                #sigma_y = pm.Deterministic('sigma_y', pm.math.log(1 + pm.math.exp(sigma_noise))+1e-3)
                sigma_y = pm.Deterministic('sigma_y', pm.math.log1pexp(sigma_noise) + 1e-5)
                        
            else:
                sigma_noise = pm.Uniform('sigma_noise', lower=0, upper=100, shape=(batch_effects_size))
                sigma_y = theano.tensor.zeros(y_shape)
                for be in be_idx:
                    a = []
                    for i, b in enumerate(be):
                        a.append(batch_effects[:,i]==b)             
                    idx = reduce(np.logical_and, a).nonzero()
                    if idx[0].shape[0]!=0:
                        sigma_y = theano.tensor.set_subtensor(sigma_y[idx,0], sigma_noise[be])
        
        else:
            sigma_noise = pm.Uniform('sigma_noise', lower=0, upper=100)
            sigma_y = theano.tensor.zeros(y_shape)
            for be in be_idx:
                a = []
                for i, b in enumerate(be):
                    a.append(batch_effects[:,i]==b)   
                idx = reduce(np.logical_and, a).nonzero()
                if idx[0].shape[0]!=0:
                    sigma_y = theano.tensor.set_subtensor(sigma_y[idx,0], sigma_noise)
            
        if configs['skewed_likelihood']:
            skewness = pm.Uniform('skewness', lower=-10, upper=10, shape=(batch_effects_size))
            alpha = theano.tensor.zeros(y_shape)
            for be in be_idx:
                a = []
                for i, b in enumerate(be):
                    a.append(batch_effects[:,i]==b)             
                idx = reduce(np.logical_and, a).nonzero()
                if idx[0].shape[0]!=0:
                    alpha = theano.tensor.set_subtensor(alpha[idx,0], skewness[be])
        else:
            alpha = 0
        
        y_like = pm.SkewNormal('y_like', mu=y_hat, sigma=sigma_y, alpha=alpha, observed=y)

    return model
    
class HBR:
    """Hierarchical Bayesian Regression for normative modeling

    Basic usage::

        model = HBR(age, site_id, gender, mri_voxel, model_type)
        trace = model.estimate()
        ys,s2 = model.predict(age, site_id, gender)

    where the variables are

    :param age: N-vector of ages for N subjects
    :param site_id: N-vector of site IDs for N subjects
    :param gender: N-vector of genders for N subjects
    :param mri_voxel: N-vector of one voxel values for N subjects
    :param model_type: string that defines the type of the model

    :returns: * ys - predictive mean
              * s2 - predictive variance

    Written by S.M. Kia
    """

    def __init__(self, configs):
        
        self.model_type = configs['type']
        self.configs = configs

            
    def estimate(self, X, y, batch_effects):
        """ Function to estimate the model """
        
        if len(X.shape)==1:
            X = np.expand_dims(X, axis=1)
        if len(y.shape)==1:
            y = np.expand_dims(y, axis=1)
        if len(batch_effects.shape)==1:
            batch_effects = np.expand_dims(batch_effects, axis=1)
        
        self.batch_effects_num = batch_effects.shape[1]
        self.batch_effects_size = []
        for i in range(self.batch_effects_num):
            self.batch_effects_size.append(len(np.unique(batch_effects[:,i])))
            
        if self.model_type == 'linear': 
            with linear_hbr(X, y, batch_effects, self.batch_effects_size, 
                               self.configs):    
                self.trace = pm.sample(1000, tune=500, chains=1,  target_accept=0.8, 
                                       cores=1)
        elif self.model_type == 'poly2': 
            with poly2_hbr(X, y, batch_effects, self.batch_effects_size, 
                               self.configs):    
                self.trace = pm.sample(1000, tune=500, chains=1,  target_accept=0.8, 
                                       cores=1)
        elif self.model_type == 'bspline': 
            self.bsp = bspline_fit(X, self.configs['order'], self.configs['nknots'])
            X = bspline_transform(X, self.bsp)
            with linear_hbr(X, y, batch_effects, self.batch_effects_size, 
                               self.configs):    
                self.trace = pm.sample(1000, tune=500, chains=1,  target_accept=0.8, 
                                       cores=1)
                
        return self.trace

    def predict(self, X, batch_effects, pred = 'single'):
        """ Function to make predictions from the model """
        
        if len(X.shape)==1:
            X = np.expand_dims(X, axis=1)
        if len(batch_effects.shape)==1:
            batch_effects = np.expand_dims(batch_effects, axis=1)
        
        samples = 1000
        if pred == 'single':
            y = np.zeros([X.shape[0],1])
            if self.model_type == 'linear': 
                with linear_hbr(X, y, batch_effects, self.batch_effects_size, self.configs):
                    ppc = pm.sample_posterior_predictive(self.trace, samples=samples, progressbar=True) 
            elif self.model_type == 'poly2': 
                with poly2_hbr(X, y, batch_effects, self.batch_effects_size, self.configs):
                    ppc = pm.sample_posterior_predictive(self.trace, samples=samples, progressbar=True) 
            elif self.model_type == 'bspline': 
                X = bspline_transform(X, self.bsp)
                with linear_hbr(X, y, batch_effects, self.batch_effects_size, self.configs):
                    ppc = pm.sample_posterior_predictive(self.trace, samples=samples, progressbar=True) 
            
            pred_mean = ppc['y_like'].mean(axis=0)
            pred_var = ppc['y_like'].var(axis=0)
        
        return pred_mean, pred_var
    

    def estimate_on_new_site(self, X, y, batch_effects):
        """ Function to adapt the model """
        
        if len(X.shape)==1:
            X = np.expand_dims(X, axis=1)
        if len(y.shape)==1:
            y = np.expand_dims(y, axis=1)
        if len(batch_effects.shape)==1:
            batch_effects = np.expand_dims(batch_effects, axis=1)
        
        self.batch_effects_num = batch_effects.shape[1]
        self.batch_effects_size = []
        for i in range(self.batch_effects_num):
            self.batch_effects_size.append(len(np.unique(batch_effects[:,i])))
            
        if self.model_type == 'linear': 
            with linear_hbr(X, y, batch_effects, self.batch_effects_size, 
                               self.configs, trace = self.trace):    
                trace = pm.sample(1000, tune=500, chains=1,  target_accept=0.8, 
                                       cores=1)
        elif self.model_type == 'poly2': 
            with poly2_hbr(X, y, batch_effects, self.batch_effects_size, 
                               self.configs, trace = self.trace):    
                trace = pm.sample(1000, tune=500, chains=1,  target_accept=0.8, 
                                       cores=1)
        if self.model_type == 'bspline': 
            #self.bsp = bspline_fit(X, self.configs['order'], self.configs['nknots'])
            X = bspline_transform(X, self.bsp)
            with linear_hbr(X, y, batch_effects, self.batch_effects_size, 
                               self.configs, trace = self.trace):    
                trace = pm.sample(1000, tune=500, chains=1,  target_accept=0.8, 
                                       cores=1)
                
        self.trace = trace    
        return trace
        
    
    def predict_on_new_site(self, X, batch_effects):
        """ Function to make predictions from the model """
        
        if len(X.shape)==1:
            X = np.expand_dims(X, axis=1)
        if len(batch_effects.shape)==1:
            batch_effects = np.expand_dims(batch_effects, axis=1)
        
        samples = 1000
        y = np.zeros([X.shape[0],1])
        if self.model_type == 'linear': 
            with linear_hbr(X, y, batch_effects, self.batch_effects_size, self.configs):
                ppc = pm.sample_posterior_predictive(self.trace, samples=samples, progressbar=True) 
        elif self.model_type == 'poly2': 
            with poly2_hbr(X, y, batch_effects, self.batch_effects_size, self.configs):
                ppc = pm.sample_posterior_predictive(self.trace, samples=samples, progressbar=True) 
        elif self.model_type == 'bspline': 
            X = bspline_transform(X, self.bsp)
            with linear_hbr(X, y, batch_effects, self.batch_effects_size, self.configs):
                ppc = pm.sample_posterior_predictive(self.trace, samples=samples, progressbar=True) 
        
        pred_mean = ppc['y_like'].mean(axis=0)
        pred_var = ppc['y_like'].var(axis=0)
        
        return pred_mean, pred_var
