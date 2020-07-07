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


def create_poly_basis(X, order):
    """ compute a polynomial basis expansion of the specified order"""
    
    if len(X.shape) == 1:
        X = X[:, np.newaxis]
    D = X.shape[1]
    Phi = np.zeros((X.shape[0], D*order))
    colid = np.arange(0, D)
    for d in range(1, order+1):
        Phi[:, colid] = X ** d
        colid += D
        
    return Phi


def from_posterior(param, samples, distribution = None, half = False, freedom=10):
    
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
            x = np.concatenate([x, [x[-1] + 0.1 * width]])
            y = np.concatenate([y, [0]])
        else:
            x = np.concatenate([[x[0] - 0.1 * width], x, [x[-1] + 0.1 * width]])
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
    

def hbr(X, y, batch_effects, batch_effects_size, configs, trace=None):
    
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
                                                    distribution=None)
            mu_prior_slope = from_posterior('mu_prior_slope', 
                                            trace['mu_prior_slope'], 
                                            distribution='normal')
            sigma_prior_slope = from_posterior('sigma_prior_slope', 
                                               trace['sigma_prior_slope'], 
                                               distribution='hcauchy')
        else:
            #mu_prior_intercept = pm.Normal('mu_prior_intercept', mu=0., sigma=1e3)
            mu_prior_intercept = pm.Uniform('mu_prior_intercept', lower=-100, upper=100)
            mu_prior_slope = pm.Normal('mu_prior_slope', mu=0., sigma=1e3, shape=(feature_num,))
            sigma_prior_slope = pm.HalfCauchy('sigma_prior_slope', 5, shape=(feature_num,))
        
        if configs['random_intercept']: 
            intercepts_offset = pm.Normal('intercepts_offset', mu=0, sd=1, 
                                          shape=(batch_effects_size))
        else:
            intercepts_offset = pm.Normal('intercepts_offset', mu=0, sd=1)
       
        intercepts = pm.Deterministic('intercepts', mu_prior_intercept + 
                                      intercepts_offset)
        
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
                    y_hat = theano.tensor.set_subtensor(y_hat[idx,0], 
                                        intercepts + theano.tensor.dot(X[idx,:], 
                                                                       slopes))
                elif (configs['random_intercept'] and not configs['random_slope']):
                    y_hat = theano.tensor.set_subtensor(y_hat[idx,0], 
                                    intercepts[be] + theano.tensor.dot(X[idx,:],
                                              slopes))
                elif (not configs['random_intercept'] and configs['random_slope']):
                    y_hat = theano.tensor.set_subtensor(y_hat[idx,0], 
                                    intercepts + theano.tensor.dot(X[idx,:], 
                                                                   slopes[be]))
                elif (configs['random_intercept'] and configs['random_slope']):        
                    y_hat = theano.tensor.set_subtensor(y_hat[idx,0], 
                                    intercepts[be] + theano.tensor.dot(X[idx,:], 
                                              slopes[be]))
        
        if configs['random_noise']:
            if configs['hetero_noise']:
                # Priors
                if trace is not None: # Used for transferring the priors
                    mu_prior_intercept_noise = from_posterior('mu_prior_intercept_noise', 
                                                            trace['mu_prior_intercept_noise'], 
                                                            distribution=None)
                    #sigma_prior_intercept_noise = from_posterior('sigma_prior_intercept_noise', 
                    #                                       trace['sigma_prior_intercept_noise'], 
                    #                                       distribution='hcauchy')
                    mu_prior_slope_noise = from_posterior('mu_prior_slope_noise', 
                                                    trace['mu_prior_slope_noise'], 
                                                    distribution='normal')
                    sigma_prior_slope_noise = from_posterior('sigma_prior_slope_noise', 
                                                       trace['sigma_prior_slope_noise'], 
                                                       distribution='hcauchy')
                else:
                    #mu_prior_intercept_noise = pm.HalfNormal('mu_prior_intercept_noise', 
                    #                                         sigma=1e3)
                    mu_prior_intercept_noise = pm.Uniform('mu_prior_intercept_noise', 
                                                          lower=0, upper=100)
                    #sigma_prior_intercept_noise = pm.HalfCauchy('sigma_prior_intercept_noise', 5)
                    mu_prior_slope_noise = pm.Normal('mu_prior_slope_noise',  mu=0., 
                                                     sigma=1e3, shape=(feature_num,))
                    sigma_prior_slope_noise = pm.HalfCauchy('sigma_prior_slope_noise', 
                                                            5, shape=(feature_num,))
                if configs['random_intercept']: 
                    intercepts_noise_offset = pm.Normal('intercepts_noise_offset',
                                                        sd=1, shape=(batch_effects_size))
                else:
                    intercepts_noise_offset = pm.Normal('intercepts_noise_offset',
                                                                    sd=1)
                    
                intercepts_noise = pm.Deterministic('intercepts_noise',
                                                    mu_prior_intercept_noise + 
                                                    intercepts_noise_offset)
    
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
                              
                sigma_y = pm.math.log1pexp(sigma_noise) + 1e-5
                        
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


def nn_hbr(X, y, batch_effects, batch_effects_size, configs, trace=None):
    
    n_hidden = configs['nn_hidden_neuron_num']
    n_layers = configs['nn_hidden_layers_num']
    feature_num = X.shape[1]
    batch_effects_num = batch_effects.shape[1]
    all_idx = []
    for i in range(batch_effects_num):
        all_idx.append(np.int16(np.unique(batch_effects[:,i])))
    be_idx = list(product(*all_idx))
        
    X = theano.shared(X)
    y = theano.shared(y)
    
    # Initialize random weights between each layer for the mu:
    init_1 = pm.floatX(np.random.randn(feature_num, n_hidden) * np.sqrt(1/feature_num))
    init_out = pm.floatX(np.random.randn(n_hidden) * np.sqrt(1/n_hidden)) 
    
    std_init_1 = pm.floatX(np.random.rand(feature_num, n_hidden))
    std_init_out = pm.floatX(np.random.rand(n_hidden))
    
    # And initialize random weights between each layer for sigma_noise:
    init_1_noise = pm.floatX(np.random.randn(feature_num, n_hidden) * np.sqrt(1/feature_num))
    init_out_noise = pm.floatX(np.random.randn(n_hidden) * np.sqrt(1/n_hidden)) 
    
    std_init_1_noise = pm.floatX(np.random.rand(feature_num, n_hidden))
    std_init_out_noise = pm.floatX(np.random.rand(n_hidden))
    
    # If there are two hidden layers, then initialize weights for the second layer:
    if n_layers == 2:
        init_2 = pm.floatX(np.random.randn(n_hidden, n_hidden) * np.sqrt(1/n_hidden))
        std_init_2 = pm.floatX(np.random.rand(n_hidden, n_hidden))
        init_2_noise = pm.floatX(np.random.randn(n_hidden, n_hidden) * np.sqrt(1/n_hidden))
        std_init_2_noise = pm.floatX(np.random.rand(n_hidden, n_hidden))
    
    with pm.Model() as model:
        if trace is not None: # Used when estimating/predicting on a new site
            weights_in_1_grp = from_posterior('w_in_1_grp', trace['w_in_1_grp'], 
                                            distribution='normal')
            
            weights_in_1_grp_sd = from_posterior('w_in_1_grp_sd', trace['w_in_1_grp_sd'], 
                                            distribution='hcauchy')
            
            if n_layers == 2:
                weights_1_2_grp = from_posterior('w_1_2_grp', trace['w_1_2_grp'], 
                                                distribution='normal') 
                
                weights_1_2_grp_sd = from_posterior('w_1_2_grp_sd', trace['w_1_2_grp_sd'], 
                                                distribution='hcauchy') 
                
            weights_2_out_grp = from_posterior('w_2_out_grp', trace['w_2_out_grp'], 
                                            distribution='normal') 
            
            weights_2_out_grp_sd = from_posterior('w_2_out_grp_sd', trace['w_2_out_grp_sd'], 
                                            distribution='hcauchy')
            
            mu_prior_intercept = from_posterior('mu_prior_intercept', trace['mu_prior_intercept'],
                                                distribution=None)
            #sigma_prior_intercept = from_posterior('sigma_prior_intercept', trace['sigma_prior_intercept'],
            #                                    distribution='hcauchy')
            
        else:
            # Group the mean distribution for input to the hidden layer:
            weights_in_1_grp = pm.Normal('w_in_1_grp', 0, sd=1, 
                                         shape=(feature_num, n_hidden), testval=init_1)
            
            # Group standard deviation:
            weights_in_1_grp_sd = pm.HalfCauchy('w_in_1_grp_sd', 1., 
                                         shape=(feature_num, n_hidden), testval=std_init_1)
            
            if n_layers == 2:
                # Group the mean distribution for hidden layer 1 to hidden layer 2:
                weights_1_2_grp = pm.Normal('w_1_2_grp', 0, sd=1, 
                                            shape=(n_hidden, n_hidden), testval=init_2)
                
                # Group standard deviation:
                weights_1_2_grp_sd = pm.HalfCauchy('w_1_2_grp_sd', 1., 
                                            shape=(n_hidden, n_hidden), testval=std_init_2)
                
            # Group the mean distribution for hidden to output:
            weights_2_out_grp = pm.Normal('w_2_out_grp', 0, sd=1, 
                                          shape=(n_hidden,), testval=init_out)
            
            # Group standard deviation:
            weights_2_out_grp_sd = pm.HalfCauchy('w_2_out_grp_sd', 1., 
                                          shape=(n_hidden,), testval=std_init_out)
            
            mu_prior_intercept = pm.Uniform('mu_prior_intercept', lower=-100, upper=100)
            #sigma_prior_intercept = pm.HalfCauchy('sigma_prior_intercept', 5)
            
        # Now create separate weights for each group, by doing
        # weights * group_sd + group_mean, we make sure the new weights are
        # coming from the (group_mean, group_sd) distribution.
        weights_in_1_raw = pm.Normal('w_in_1', 0, sd=1,
                                     shape=(batch_effects_size + [feature_num, n_hidden]))
        weights_in_1 = weights_in_1_raw * weights_in_1_grp_sd + weights_in_1_grp
        
        if n_layers == 2:
            weights_1_2_raw = pm.Normal('w_1_2', 0, sd=1,
                                        shape=(batch_effects_size + [n_hidden, n_hidden]))
            weights_1_2 = weights_1_2_raw * weights_1_2_grp_sd + weights_1_2_grp
            
        weights_2_out_raw = pm.Normal('w_2_out', 0, sd=1,
                                      shape=(batch_effects_size + [n_hidden]))
        weights_2_out = weights_2_out_raw * weights_2_out_grp_sd + weights_2_out_grp
        
        intercepts_offset = pm.Normal('intercepts_offset', mu=0, sd=1, 
                                          shape=(batch_effects_size))
       
        intercepts = pm.Deterministic('intercepts', intercepts_offset + mu_prior_intercept)
            
        # Build the neural network and estimate y_hat:
        y_hat = theano.tensor.zeros(y.shape)
        for be in be_idx:
            # Find the indices corresponding to 'group be': 
            a = []
            for i, b in enumerate(be):
                a.append(batch_effects[:,i]==b)
            idx = reduce(np.logical_and, a).nonzero()
            if idx[0].shape[0] != 0:
                act_1 = pm.math.tanh(theano.tensor.dot(X[idx,:], weights_in_1[be]))
                if n_layers == 2:
                    act_2 = pm.math.tanh(theano.tensor.dot(act_1, weights_1_2[be]))
                    y_hat = theano.tensor.set_subtensor(y_hat[idx,0], intercepts[be] + theano.tensor.dot(act_2, weights_2_out[be]))
                else:
                    y_hat = theano.tensor.set_subtensor(y_hat[idx,0], intercepts[be] + theano.tensor.dot(act_1, weights_2_out[be]))
                    
        # If we want to estimate varying noise terms across groups:
        if configs['random_noise']:
            if configs['hetero_noise']:
                if trace is not None: # # Used when estimating/predicting on a new site
                    weights_in_1_grp_noise = from_posterior('w_in_1_grp_noise', 
                                                            trace['w_in_1_grp_noise'], 
                                                            distribution='normal')
                    
                    weights_in_1_grp_sd_noise = from_posterior('w_in_1_grp_sd_noise', 
                                                               trace['w_in_1_grp_sd_noise'], 
                                                               distribution='hcauchy')
                    
                    if n_layers == 2:
                        weights_1_2_grp_noise = from_posterior('w_1_2_grp_noise', 
                                                               trace['w_1_2_grp_noise'], 
                                                               distribution='normal')
                        
                        weights_1_2_grp_sd_noise = from_posterior('w_1_2_grp_sd_noise', 
                                                                  trace['w_1_2_grp_sd_noise'], 
                                                                  distribution='hcauchy')
                        
                    weights_2_out_grp_noise = from_posterior('w_2_out_grp_noise', 
                                                             trace['w_2_out_grp_noise'], 
                                                             distribution='normal')
                    
                    weights_2_out_grp_sd_noise = from_posterior('w_2_out_grp_sd_noise', 
                                                                trace['w_2_out_grp_sd_noise'], 
                                                                distribution='hcauchy')
                    
                else:
                    # The input layer to the first hidden layer:
                    weights_in_1_grp_noise = pm.Normal('w_in_1_grp_noise', 0, sd=1, 
                                               shape=(feature_num,n_hidden), 
                                               testval=init_1_noise)
                    weights_in_1_grp_sd_noise = pm.HalfCauchy('w_in_1_grp_sd_noise', 1, 
                                               shape=(feature_num,n_hidden), 
                                               testval=std_init_1_noise)
                    
                    
                    # The first hidden layer to second hidden layer:
                    if n_layers == 2:
                        weights_1_2_grp_noise = pm.Normal('w_1_2_grp_noise', 0, sd=1, 
                                                          shape=(n_hidden, n_hidden), 
                                                          testval=init_2_noise)
                        weights_1_2_grp_sd_noise = pm.HalfCauchy('w_1_2_grp_sd_noise', 1, 
                                                          shape=(n_hidden, n_hidden), 
                                                          testval=std_init_2_noise)
                        
                    # The second hidden layer to output layer:
                    weights_2_out_grp_noise = pm.Normal('w_2_out_grp_noise', 0, sd=1, 
                                                        shape=(n_hidden,), 
                                                        testval=init_out_noise)
                    weights_2_out_grp_sd_noise = pm.HalfCauchy('w_2_out_grp_sd_noise', 1, 
                                                        shape=(n_hidden,), 
                                                        testval=std_init_out_noise)
                    
                    #mu_prior_intercept_noise = pm.HalfNormal('mu_prior_intercept_noise', sigma=1e3)
                    #sigma_prior_intercept_noise = pm.HalfCauchy('sigma_prior_intercept_noise', 5)
            
                # Now create separate weights for each group:
                weights_in_1_raw_noise = pm.Normal('w_in_1_noise', 0, sd=1,
                                                   shape=(batch_effects_size + [feature_num, n_hidden]))
                weights_in_1_noise = weights_in_1_raw_noise * weights_in_1_grp_sd_noise + weights_in_1_grp_noise
                
                if n_layers == 2:
                    weights_1_2_raw_noise = pm.Normal('w_1_2_noise', 0, sd=1,
                                                      shape=(batch_effects_size + [n_hidden, n_hidden]))
                    weights_1_2_noise = weights_1_2_raw_noise * weights_1_2_grp_sd_noise + weights_1_2_grp_noise
                    
                weights_2_out_raw_noise = pm.Normal('w_2_out_noise', 0, sd=1,
                                                    shape=(batch_effects_size + [n_hidden]))
                weights_2_out_noise = weights_2_out_raw_noise * weights_2_out_grp_sd_noise + weights_2_out_grp_noise
                
                #intercepts_offset_noise = pm.Normal('intercepts_offset_noise', mu=0, sd=1, 
                #                          shape=(batch_effects_size))
       
                #intercepts_noise = pm.Deterministic('intercepts_noise', mu_prior_intercept_noise + 
                #                      intercepts_offset_noise * sigma_prior_intercept_noise)
                
                # Build the neural network and estimate the sigma_y:
                sigma_y = theano.tensor.zeros(y.shape)
                for be in be_idx:
                    a = []
                    for i, b in enumerate(be):
                        a.append(batch_effects[:,i]==b) 
                    idx = reduce(np.logical_and, a).nonzero()
                    if idx[0].shape[0] != 0:
                        act_1_noise = pm.math.sigmoid(theano.tensor.dot(X[idx,:], weights_in_1_noise[be]))
                        if n_layers == 2:
                            act_2_noise = pm.math.sigmoid(theano.tensor.dot(act_1_noise, weights_1_2_noise[be]))
                            temp = pm.math.log1pexp(theano.tensor.dot(act_2_noise, weights_2_out_noise[be])) + 1e-5
                        else:
                            temp = pm.math.log1pexp(theano.tensor.dot(act_1_noise, weights_2_out_noise[be])) + 1e-5
                        sigma_y = theano.tensor.set_subtensor(sigma_y[idx,0], temp)
                
            else: # homoscedastic noise:
                sigma_noise = pm.Uniform('sigma_noise', lower=0, upper=100, shape=(batch_effects_size))
                sigma_y = theano.tensor.zeros(y.shape)
                for be in be_idx:
                    a = []
                    for i, b in enumerate(be):
                        a.append(batch_effects[:,i]==b)             
                    idx = reduce(np.logical_and, a).nonzero()
                    if idx[0].shape[0]!=0:
                        sigma_y = theano.tensor.set_subtensor(sigma_y[idx,0], sigma_noise[be])
        
        else: # do not allow for random noise terms across groups:
            sigma_noise = pm.Uniform('sigma_noise', lower=0, upper=100)
            sigma_y = theano.tensor.zeros(y.shape)
            for be in be_idx:
                a = []
                for i, b in enumerate(be):
                    a.append(batch_effects[:,i]==b)
                idx = reduce(np.logical_and, a).nonzero()
                if idx[0].shape[0]!=0:
                    sigma_y = theano.tensor.set_subtensor(sigma_y[idx,0], sigma_noise)
        
        if configs['skewed_likelihood']:
            skewness = pm.Uniform('skewness', lower=-10, upper=10, shape=(batch_effects_size))
            alpha = theano.tensor.zeros(y.shape)
            for be in be_idx:
                a = []
                for i, b in enumerate(be):
                    a.append(batch_effects[:,i]==b)             
                idx = reduce(np.logical_and, a).nonzero()
                if idx[0].shape[0]!=0:
                    alpha = theano.tensor.set_subtensor(alpha[idx,0], skewness[be])
        else: 
            alpha = 0 # symmetrical normal distribution
        
        y_like = pm.SkewNormal('y_like', mu=y_hat, sigma=sigma_y, alpha=alpha, observed=y) 
        
    return model


class HBR:
    """Hierarchical Bayesian Regression for normative modeling

    Basic usage::

        model = HBR(configs)
        trace = model.estimate(X, y, batch_effects)
        ys,s2 = model.predict(X, batch_effects)

    where the variables are

    :param configs: a dictionary of model configurations.
    :param X: N-by-P input matrix of P features for N subjects
    :param y: N-by-1 vector of outputs. 
    :param batch_effects: N-by-B matrix of B batch ids for N subjects.

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
            with hbr(X, y, batch_effects, self.batch_effects_size, 
                               self.configs):    
                self.trace = pm.sample(self.configs['n_samples'], 
                                       tune=self.configs['n_tuning'], 
                                       chains=self.configs['n_chains'],  
                                       target_accept=self.configs['target_accept'], 
                                       init=self.configs['init'], n_init=50000, 
                                       cores=self.configs['cores'])
        elif self.model_type == 'polynomial': 
            X = create_poly_basis(X, self.configs['order'])
            with hbr(X, y, batch_effects, self.batch_effects_size, 
                               self.configs):    
                self.trace = pm.sample(self.configs['n_samples'], 
                                       tune=self.configs['n_tuning'], 
                                       chains=self.configs['n_chains'],  
                                       target_accept=self.configs['target_accept'], 
                                       init=self.configs['init'], n_init=50000, 
                                       cores=self.configs['cores'])
        elif self.model_type == 'bspline': 
            self.bsp = bspline_fit(X, self.configs['order'], self.configs['nknots'])
            X = bspline_transform(X, self.bsp)
            with hbr(X, y, batch_effects, self.batch_effects_size, 
                               self.configs):    
                self.trace = pm.sample(self.configs['n_samples'], 
                                       tune=self.configs['n_tuning'], 
                                       chains=self.configs['n_chains'],  
                                       target_accept=self.configs['target_accept'], 
                                       init=self.configs['init'], n_init=50000, 
                                       cores=self.configs['cores'])
        elif self.model_type == 'nn': 
            with nn_hbr(X, y, batch_effects, self.batch_effects_size, 
                               self.configs):    
                self.trace = pm.sample(self.configs['n_samples'], 
                                       tune=self.configs['n_tuning'], 
                                       chains=self.configs['n_chains'], 
                                       target_accept=self.configs['target_accept'], 
                                       init=self.configs['init'], n_init=50000, 
                                       cores=self.configs['cores'])
                
        return self.trace

    def predict(self, X, batch_effects, pred = 'single'):
        """ Function to make predictions from the model """
        
        if len(X.shape)==1:
            X = np.expand_dims(X, axis=1)
        if len(batch_effects.shape)==1:
            batch_effects = np.expand_dims(batch_effects, axis=1)
        
        samples = self.configs['n_samples']
        if pred == 'single':
            y = np.zeros([X.shape[0],1])
            if self.model_type == 'linear': 
                with hbr(X, y, batch_effects, self.batch_effects_size, self.configs):
                    ppc = pm.sample_posterior_predictive(self.trace, samples=samples, progressbar=True) 
            elif self.model_type == 'polynomial':
                X = create_poly_basis(X, self.configs['order'])
                with hbr(X, y, batch_effects, self.batch_effects_size, self.configs):
                    ppc = pm.sample_posterior_predictive(self.trace, samples=samples, progressbar=True) 
            elif self.model_type == 'bspline': 
                X = bspline_transform(X, self.bsp)
                with hbr(X, y, batch_effects, self.batch_effects_size, self.configs):
                    ppc = pm.sample_posterior_predictive(self.trace, samples=samples, progressbar=True) 
            elif self.model_type == 'nn': 
                with nn_hbr(X, y, batch_effects, self.batch_effects_size, self.configs):
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
            with hbr(X, y, batch_effects, self.batch_effects_size, 
                               self.configs, trace = self.trace):    
                self.trace = pm.sample(self.configs['n_samples'], 
                                       tune=self.configs['n_tuning'], 
                                       chains=self.configs['n_chains'],  
                                       target_accept=self.configs['target_accept'], 
                                       init=self.configs['init'], n_init=50000, 
                                       cores=self.configs['cores'])
        elif self.model_type == 'polynomial': 
            X = create_poly_basis(X, self.configs['order'])
            with hbr(X, y, batch_effects, self.batch_effects_size, 
                               self.configs, trace = self.trace):    
                self.trace = pm.sample(self.configs['n_samples'], 
                                       tune=self.configs['n_tuning'], 
                                       chains=self.configs['n_chains'],  
                                       target_accept=self.configs['target_accept'], 
                                       init=self.configs['init'], n_init=50000, 
                                       cores=self.configs['cores'])
        if self.model_type == 'bspline': 
            X = bspline_transform(X, self.bsp)
            with hbr(X, y, batch_effects, self.batch_effects_size, 
                               self.configs, trace = self.trace):    
                self.trace = pm.sample(self.configs['n_samples'], 
                                       tune=self.configs['n_tuning'], 
                                       chains=self.configs['n_chains'], 
                                       target_accept=self.configs['target_accept'], 
                                       init=self.configs['init'], n_init=50000, 
                                       cores=self.configs['cores'])
        elif self.model_type == 'nn': 
            with nn_hbr(X, y, batch_effects, self.batch_effects_size, 
                               self.configs, trace = self.trace):    
                self.trace = pm.sample(self.configs['n_samples'], 
                                       tune=self.configs['n_tuning'], 
                                       chains=self.configs['n_chains'], 
                                       target_accept=self.configs['target_accept'], 
                                       init=self.configs['init'], n_init=50000, 
                                       cores=self.configs['cores'])
                
        return self.trace
        
    
    def predict_on_new_site(self, X, batch_effects):
        """ Function to make predictions from the model """
        
        if len(X.shape)==1:
            X = np.expand_dims(X, axis=1)
        if len(batch_effects.shape)==1:
            batch_effects = np.expand_dims(batch_effects, axis=1)
        
        samples = self.configs['n_samples']
        y = np.zeros([X.shape[0],1])
        if self.model_type == 'linear': 
            with hbr(X, y, batch_effects, self.batch_effects_size, self.configs):
                ppc = pm.sample_posterior_predictive(self.trace, samples=samples, progressbar=True) 
        elif self.model_type == 'polynomial': 
            X = create_poly_basis(X, self.configs['order'])
            with hbr(X, y, batch_effects, self.batch_effects_size, self.configs):
                ppc = pm.sample_posterior_predictive(self.trace, samples=samples, progressbar=True) 
        elif self.model_type == 'bspline': 
            X = bspline_transform(X, self.bsp)
            with hbr(X, y, batch_effects, self.batch_effects_size, self.configs):
                ppc = pm.sample_posterior_predictive(self.trace, samples=samples, progressbar=True) 
        elif self.model_type == 'nn': 
            with nn_hbr(X, y, batch_effects, self.batch_effects_size, self.configs):
                ppc = pm.sample_posterior_predictive(self.trace, samples=samples, progressbar=True) 
            
        pred_mean = ppc['y_like'].mean(axis=0)
        pred_var = ppc['y_like'].var(axis=0)
        
        return pred_mean, pred_var
    

    def generate(self, X, batch_effects, samples):
        """ Function to generate samples from posterior predictive distribution """
        
        if len(X.shape)==1:
            X = np.expand_dims(X, axis=1)
        if len(batch_effects.shape)==1:
            batch_effects = np.expand_dims(batch_effects, axis=1)
                
        y = np.zeros([X.shape[0],1])
        if self.model_type == 'linear': 
            with hbr(X, y, batch_effects, self.batch_effects_size, self.configs):
                ppc = pm.sample_posterior_predictive(self.trace, samples=samples, progressbar=True) 
        elif self.model_type == 'polynomial':
            X = create_poly_basis(X, self.configs['order'])
            with hbr(X, y, batch_effects, self.batch_effects_size, self.configs):
                ppc = pm.sample_posterior_predictive(self.trace, samples=samples, progressbar=True) 
        elif self.model_type == 'bspline': 
            X = bspline_transform(X, self.bsp)
            with hbr(X, y, batch_effects, self.batch_effects_size, self.configs):
                ppc = pm.sample_posterior_predictive(self.trace, samples=samples, progressbar=True) 
        elif self.model_type == 'nn': 
            with nn_hbr(X, y, batch_effects, self.batch_effects_size, self.configs):
                ppc = pm.sample_posterior_predictive(self.trace, samples=samples, progressbar=True) 
        
        generated_samples = np.reshape(ppc['y_like'].squeeze().T, [X.shape[0]*samples, 1])
        X = np.repeat(X, samples)
        if len(X.shape)==1:
            X = np.expand_dims(X, axis=1)
        batch_effects = np.repeat(batch_effects, samples)
        if len(batch_effects.shape)==1:
            batch_effects = np.expand_dims(batch_effects, axis=1)
        
        return X, batch_effects, generated_samples
    
    
    def sample_prior_predictive(self, X, batch_effects, samples, trace=None):
        """ Function to sample from prior predictive distribution """
        
        if len(X.shape)==1:
            X = np.expand_dims(X, axis=1)
        if len(batch_effects.shape)==1:
            batch_effects = np.expand_dims(batch_effects, axis=1)
            
        self.batch_effects_num = batch_effects.shape[1]
        self.batch_effects_size = []
        for i in range(self.batch_effects_num):
            self.batch_effects_size.append(len(np.unique(batch_effects[:,i])))
                
        y = np.zeros([X.shape[0],1])
        
        if self.model_type == 'linear': 
            with hbr(X, y, batch_effects, self.batch_effects_size, self.configs,
                     trace):
                ppc = pm.sample_prior_predictive(samples=samples) 
        elif self.model_type == 'polynomial':
            X = create_poly_basis(X, self.configs['order'])
            with hbr(X, y, batch_effects, self.batch_effects_size, self.configs,
                     trace):
                ppc = pm.sample_prior_predictive(samples=samples) 
        elif self.model_type == 'bspline': 
            self.bsp = bspline_fit(X, self.configs['order'], self.configs['nknots'])
            X = bspline_transform(X, self.bsp)
            with hbr(X, y, batch_effects, self.batch_effects_size, self.configs,
                     trace):
                ppc = pm.sample_prior_predictive(samples=samples) 
        elif self.model_type == 'nn': 
            with nn_hbr(X, y, batch_effects, self.batch_effects_size, self.configs,
                        trace):
                ppc = pm.sample_prior_predictive(samples=samples) 
        
        return ppc
    