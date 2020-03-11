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
from scipy import stats


def from_posterior(param, samples, distribution = None, half = False, freedom=1):
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
        return pm.Normal(param, mu=temp[0], sigma=freedom*temp[1])
    elif (distribution=='hnormal'):
        temp = stats.halfnorm.fit(samples)
        return pm.HalfNormal(param, sigma=freedom*temp[1])
    elif (distribution=='hcauchy'):
        temp = stats.halfcauchy.fit(samples)
        return pm.HalfCauchy(param, freedom*temp[1])

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

    def __init__(self, age, site_id, gender, y, configs):
        self.site_num = len(np.unique(site_id))
        self.gender_num = len(np.unique(gender))
        self.model_type = configs['type']
        self.noise_model = configs['noise_model']
        self.configs = configs
        if self.model_type == 'nn':
            age = np.expand_dims(age, axis = 1)
            #age = np.expand_dims(age, axis = 2)
            age = np.repeat(age, configs['nn_hidden_neuron_num'], axis = 1)
        self.s = theano.shared(site_id)
        self.g = theano.shared(gender)
        self.a = theano.shared(age)
        
        with pm.Model() as model:
            if self.model_type == 'linear':    
                # Priors
                mu_prior_intercept = pm.Normal('mu_prior_intercept', mu=0., sigma=1e5)
                sigma_prior_intercept = pm.HalfCauchy('sigma_prior_intercept', 5)
                mu_prior_slope = pm.Normal('mu_prior_slope', mu=0., sigma=1e5)
                sigma_prior_slope = pm.HalfCauchy('sigma_prior_slope', 5)
            
                if configs['random_intercept']: # Random intercepts
                    intercepts_offset = pm.Normal('intercepts_offset', mu=0, sd=1, 
                                                  shape=(self.gender_num,self.site_num))
                    #intercepts = pm.Normal('intercepts', mu=mu_prior_intercept, 
                    #                       sigma=sigma_prior_intercept, 
                    #                       shape=(self.gender_num,self.site_num))
                else:
                    #intercepts = pm.Normal('intercepts', mu=mu_prior_intercept, 
                    #                       sigma=sigma_prior_intercept)
                    intercepts_offset = pm.Normal('intercepts_offset', mu=0, sd=1)
                    
                intercepts = pm.Deterministic('intercepts', mu_prior_intercept + 
                                              intercepts_offset * sigma_prior_intercept)
                
                if configs['random_slope']:  # Random slopes
                    slopes_offset = pm.Normal('slopes_offset', mu=0, sd=1, 
                                             shape=(self.gender_num,self.site_num))
                    #slopes = pm.Normal('slopes', mu=mu_prior_slope, 
                    #                   sigma=sigma_prior_slope, 
                    #                   shape=(self.gender_num,self.site_num))
                else:
                    #slopes = pm.Normal('slopes', mu=mu_prior_slope, sigma=sigma_prior_slope)
                    slopes_offset = pm.Normal('slopes_offset', mu=0, sd=1)
                
                slopes = pm.Deterministic('slopes', mu_prior_slope + 
                                                  slopes_offset * sigma_prior_slope)
                
                if (not configs['random_intercept'] and not configs['random_slope']):
                    y_hat = intercepts + self.a * slopes
                elif (configs['random_intercept'] and not configs['random_slope']):
                    y_hat = intercepts[(self.g, self.s)] + self.a * slopes
                elif (not configs['random_intercept'] and configs['random_slope']):
                    y_hat = intercepts + self.a * slopes[(self.g, self.s)]
                elif (configs['random_intercept'] and configs['random_slope']):
                    y_hat = intercepts[(self.g, self.s)] + self.a * slopes[(self.g, self.s)]
            
            elif self.model_type == 'poly2':
                # Priors
                mu_prior_intercept = pm.Normal('mu_prior_intercept', mu=0., sigma=1e5)
                sigma_prior_intercept = pm.HalfCauchy('sigma_prior_intercept', 5)
                mu_prior_slope_1 = pm.Normal('mu_prior_slope_1', mu=0., sigma=1e5)
                sigma_prior_slope_1 = pm.HalfCauchy('sigma_prior_slope_1', 5)
                mu_prior_slope_2 = pm.Normal('mu_prior_slope_2', mu=0., sigma=1e5)
                sigma_prior_slope_2 = pm.HalfCauchy('sigma_prior_slope_2', 5)
                
                if configs['random_intercept']: # Random intercepts
                    #intercepts = pm.Normal('intercepts', mu=mu_prior_intercept, sigma=sigma_prior_intercept, shape=(self.gender_num,self.site_num))
                    intercepts_offset = pm.Normal('intercepts_offset', mu=0, sd=1,
                                                  shape=(self.gender_num,self.site_num))
                else:
                    #intercepts = pm.Normal('intercepts', mu=mu_prior_intercept, sigma=sigma_prior_intercept)
                    intercepts_offset = pm.Normal('intercepts_offset', mu=0, sd=1)
                    
                intercepts = pm.Deterministic('intercepts', mu_prior_intercept + 
                                              intercepts_offset * sigma_prior_intercept)
                
                if configs['random_slope']:  # Random slopes
                    #slopes_1 = pm.Normal('slopes_1', mu=mu_prior_slope_1, sigma=sigma_prior_slope_1, shape=(self.gender_num,self.site_num))
                    slopes_offset1 = pm.Normal('slopes_offset1', mu=0, sd=1, 
                                             shape=(self.gender_num,self.site_num))
                    #slopes_2 = pm.Normal('slopes_2', mu=mu_prior_slope_2, sigma=sigma_prior_slope_2, shape=(self.gender_num,self.site_num))
                    slopes_offset2 = pm.Normal('slopes_offset2', mu=0, sd=1, 
                                             shape=(self.gender_num,self.site_num))
                else:
                    slopes_offset1 = pm.Normal('slopes_offset1', mu=0, sd=1)
                    slopes_offset2 = pm.Normal('slopes_offset2', mu=0, sd=1)
                
                slopes_1 = pm.Deterministic('slopes_1', mu_prior_slope_1 + 
                                                  slopes_offset1 * sigma_prior_slope_1)
                slopes_2 = pm.Deterministic('slopes_2', mu_prior_slope_2 + 
                                                  slopes_offset2 * sigma_prior_slope_2)
                
                if (not configs['random_intercept'] and not configs['random_slope']):
                    y_hat = intercepts + self.a * slopes_1 + self.a**2 * slopes_2
                elif (configs['random_intercept'] and not configs['random_slope']):
                    y_hat = intercepts[(self.g, self.s)] + self.a * slopes_1 + self.a**2 * slopes_2
                elif (not configs['random_intercept'] and configs['random_slope']):
                    y_hat = intercepts + self.a * slopes_1[(self.g, self.s)]  + self.a**2 * slopes_2[(self.g, self.s)] 
                elif (configs['random_intercept'] and configs['random_slope']):
                    y_hat = intercepts[(self.g, self.s)] + self.a * slopes_1[(self.g, self.s)]  + self.a**2 * slopes_2[(self.g, self.s)] 
                    
            elif self.model_type == 'nn':
                n_hidden = self.configs['nn_hidden_neuron_num']
                init_1 = pm.floatX(np.random.randn(n_hidden))
                init_out = pm.floatX(np.random.randn(n_hidden))
                std_init_1 = pm.floatX(np.ones([n_hidden]))
                std_init_out = pm.floatX(np.ones([n_hidden,]))
                
                weights_in_1_grp = pm.Normal('w_in_1_grp', 0, sd=1., 
                                 shape=(n_hidden,), 
                                 testval=init_1)
                # Group standard-deviation
                weights_in_1_grp_sd = pm.HalfNormal('w_in_1_grp_sd', sd=1., 
                                         shape=(n_hidden,), 
                                         testval=std_init_1)
                # Group mean distribution from hidden layer to output
                weights_1_out_grp = pm.Normal('w_1_out_grp', 0, sd=1., 
                                          shape=(n_hidden,), 
                                          testval=init_out)
                weights_1_out_grp_sd = pm.HalfNormal('w_1_out_grp_sd', sd=1., 
                                          shape=(n_hidden,), 
                                          testval=std_init_out)
                # Separate weights for each different model
                weights_in_1_raw = pm.Normal('w_in_1', 
                                             shape=(self.gender_num, self.site_num, n_hidden))
                # Non-centered specification of hierarchical model
                weights_in_1 = pm.Deterministic('weights_in_1', weights_in_1_raw * weights_in_1_grp_sd + weights_in_1_grp)

                weights_1_out_raw = pm.Normal('w_1_out', 
                                              shape=(self.gender_num, self.site_num, n_hidden))
                weights_1_out = pm.Deterministic('weights_1_out', weights_1_out_raw * weights_1_out_grp_sd + weights_1_out_grp)
                
                # Build neural-network using tanh activation function
                act_1 = pm.math.tanh(self.a *  weights_in_1[(self.g, self.s)].squeeze())
                y_hat = pm.math.sum(act_1 * weights_1_out[(self.g, self.s)].squeeze(), axis=1)
                #y_hat = theano.tensor.dot(act_1, 
                #                    weights_1_out[(self.g, self.s)].squeeze())
                #act_1 = pm.math.tanh(theano.tensor.batched_dot(self.a, weights_in_1))
                #y_hat = theano.tensor.batched_dot(act_1, weights_1_out)
                
            if configs['random_noise']:  # Random Noise
                if configs['hetero_noise']:
                    if self.noise_model == 'linear':
                        # Priors
                        mu_prior_intercept_noise = pm.HalfNormal('mu_prior_intercept_noise',  sigma=1e5)
                        sigma_prior_intercept_noise = pm.HalfCauchy('sigma_prior_intercept_noise', 5)
                        mu_prior_slope_noise = pm.Normal('mu_prior_slope_noise',  mu=0., sigma=1e5)
                        sigma_prior_slope_noise = pm.HalfCauchy('sigma_prior_slope_noise', 5)
                        if configs['random_intercept']: # Random intercepts
                            intercepts_noise_offset = pm.HalfNormal('intercepts_noise_offset',
                                                                sd=1,
                                                                shape=(self.gender_num,self.site_num))
                            #intercepts_noise = pm.Normal('intercepts_noise', 
                            #                             mu=mu_prior_intercept_noise, 
                            #                             sigma=sigma_prior_intercept_noise, 
                            #                             shape=(self.gender_num,self.site_num))
                        else:
                            #intercepts_noise = pm.Normal('intercepts_noise', 
                            #                             mu=mu_prior_intercept_noise, 
                            #                             sigma=sigma_prior_intercept_noise)
                            intercepts_noise_offset = pm.HalfNormal('intercepts_noise_offset',
                                                                sd=1)
                            
                        intercepts_noise = pm.Deterministic('intercepts_noise', mu_prior_intercept_noise + 
                                                  intercepts_noise_offset * sigma_prior_intercept_noise)
                        
                        if configs['random_slope']:  # Random slopes
                            slopes_noise_offset = pm.Normal('slopes_noise_offset', mu=0, sd=1, 
                                             shape=(self.gender_num,self.site_num))
                            #slopes_noise = pm.Normal('slopes_noise', mu=mu_prior_slope_noise, 
                            #                         sigma=sigma_prior_slope_noise, 
                            #                         shape=(self.gender_num,self.site_num))
                        else:
                            #slopes_noise = pm.Normal('slopes_noise', 
                            #                         mu=mu_prior_slope_noise, 
                            #                         sigma=sigma_prior_slope_noise)
                            slopes_noise_offset = pm.Normal('slopes_noise_offset', mu=0, sd=1)
                            
                        slopes_noise = pm.Deterministic('slopes_noise', mu_prior_slope_noise + 
                                                  slopes_noise_offset * sigma_prior_slope_noise)
                        
                        if (not configs['random_intercept'] and not configs['random_slope']):
                            sigma_noise = intercepts_noise + self.a * slopes_noise
                        elif (configs['random_intercept'] and not configs['random_slope']):
                            sigma_noise = intercepts_noise[(self.g, self.s)] + self.a * slopes_noise
                        elif (not configs['random_intercept'] and configs['random_slope']):
                            sigma_noise = intercepts_noise + self.a * slopes_noise[(self.g, self.s)]
                        elif (configs['random_intercept'] and configs['random_slope']):
                            sigma_noise = intercepts_noise[(self.g, self.s)] + self.a * slopes_noise[(self.g, self.s)]
                      
                        sigma_y = pm.Deterministic('sigma_y', pm.math.log(1 + pm.math.exp(sigma_noise))+1e-3)
                        
                    elif self.noise_model == 'poly2':
                        
                        mu_prior_intercept_noise = pm.HalfNormal('mu_prior_intercept_noise', sigma=1e5)
                        sigma_prior_intercept_noise = pm.HalfCauchy('sigma_prior_intercept_noise', 5)
                        mu_prior_slope_1_noise = pm.Normal('mu_prior_slope_1_noise', mu=0., sigma=1e5)
                        sigma_prior_slope_1_noise = pm.HalfCauchy('sigma_prior_slope_1_noise', 5)
                        mu_prior_slope_2_noise = pm.Normal('mu_prior_slope_2_noise', mu=0., sigma=1e5)
                        sigma_prior_slope_2_noise = pm.HalfCauchy('sigma_prior_slope_2_noise', 5)
                        
                        if configs['random_intercept']: # Random intercepts
                            #intercepts_noise = pm.Normal('intercepts_noise', 
                            #                             mu=mu_prior_intercept_noise, 
                            #                             sigma=sigma_prior_intercept_noise, 
                            #                             shape=(self.gender_num,self.site_num))
                            intercepts_noise_offset = pm.HalfNormal('intercepts_noise_offset', sd=1,
                                                  shape=(self.gender_num,self.site_num))
                        else:
                            #intercepts_noise = pm.Normal('intercepts_noise', 
                            #                             mu=mu_prior_intercept_noise, 
                            #                             sigma=sigma_prior_intercept_noise)
                            intercepts_noise_offset = pm.HalfNormal('intercepts_noise_offset', sd=1)
                            
                        intercepts_noise = pm.Deterministic('intercepts_noise', mu_prior_intercept_noise + 
                                              intercepts_noise_offset * sigma_prior_intercept_noise)
                        
                        if configs['random_slope']:  # Random slopes
                            #slopes_1_noise = pm.Normal('slopes_1_noise', mu=mu_prior_slope_1_noise, 
                            #                         sigma=sigma_prior_slope_1_noise, 
                            #                         shape=(self.gender_num,self.site_num))
                            slopes_noise_offset1 = pm.Normal('slopes_noise_offset1', mu=0, sd=1, 
                                             shape=(self.gender_num,self.site_num))
                            #slopes_2_noise = pm.Normal('slopes_2_noise', mu=mu_prior_slope_2_noise, 
                            #                         sigma=sigma_prior_slope_2_noise, 
                            #                         shape=(self.gender_num,self.site_num))
                            slopes_noise_offset2 = pm.Normal('slopes_noise_offset2', mu=0, sd=1, 
                                             shape=(self.gender_num,self.site_num))
                        else:
                            #slopes_1_noise = pm.Normal('slopes_1_noise', 
                            #                         mu=mu_prior_slope_1_noise, 
                            #                         sigma=sigma_prior_slope_1_noise)
                            slopes_noise_offset1 = pm.Normal('slopes_noise_offset1', mu=0, sd=1)
                            #slopes_2_noise = pm.Normal('slopes_2_noise', 
                            #                         mu=mu_prior_slope_2_noise, 
                            #                         sigma=sigma_prior_slope_2_noise)
                            slopes_noise_offset2 = pm.Normal('slopes_noise_offset2', mu=0, sd=1)
                        
                        slopes_1_noise = pm.Deterministic('slopes_1_noise', mu_prior_slope_1_noise + 
                                                  slopes_noise_offset1 * sigma_prior_slope_1_noise)
                        slopes_2_noise = pm.Deterministic('slopes_2_noise', mu_prior_slope_2_noise + 
                                                  slopes_noise_offset2 * sigma_prior_slope_2_noise)
                
                        if (not configs['random_intercept'] and not configs['random_slope']):
                            sigma_noise = intercepts_noise + self.a * slopes_1_noise + self.a**2 * slopes_2_noise
                        elif (configs['random_intercept'] and not configs['random_slope']):
                            sigma_noise = intercepts_noise[(self.g, self.s)] + self.a * slopes_1_noise + self.a**2 * slopes_2_noise
                        elif (not configs['random_intercept'] and configs['random_slope']):
                            sigma_noise = intercepts_noise + self.a * slopes_1_noise[(self.g, self.s)] + self.a**2 * slopes_2_noise[(self.g, self.s)]
                        elif (configs['random_intercept'] and configs['random_slope']):
                            sigma_noise = intercepts_noise[(self.g, self.s)] + self.a * slopes_1_noise[(self.g, self.s)] + self.a**2 * slopes_2_noise[(self.g, self.s)]
                            
                        sigma_y = pm.Deterministic('sigma_y', pm.math.log(1 + pm.math.exp(sigma_noise))+1e-3)
                        
                    elif self.noise_model == 'nn':
                        n_hidden = self.configs['nn_hidden_neuron_num']
                        init_1_noise = pm.floatX(np.random.randn(n_hidden))
                        init_out_noise = pm.floatX(np.random.randn(n_hidden))
                        std_init_1_noise = pm.floatX(np.ones([n_hidden]))
                        std_init_out_noise = pm.floatX(np.ones([n_hidden,]))
                        
                        weights_in_1_grp_noise = pm.Normal('w_in_1_grp_noise', 0, sd=1., 
                                         shape=(n_hidden,), 
                                         testval=init_1_noise)
                        # Group standard-deviation
                        weights_in_1_grp_sd_noise = pm.HalfNormal('w_in_1_grp_sd_noise', sd=1., 
                                                 shape=(n_hidden,), 
                                                 testval=std_init_1_noise)
                        # Group mean distribution from hidden layer to output
                        weights_1_out_grp_noise = pm.Normal('w_1_out_grp_noise', 0, sd=1., 
                                                  shape=(n_hidden,), 
                                                  testval=init_out_noise)
                        weights_1_out_grp_sd_noise = pm.HalfNormal('w_1_out_grp_sd_noise', sd=1., 
                                                  shape=(n_hidden,), 
                                                  testval=std_init_out_noise)
                        # Separate weights for each different model
                        weights_in_1_raw_noise = pm.Normal('w_in_1_noise', 
                                                     shape=(self.gender_num, self.site_num, n_hidden))
                        # Non-centered specification of hierarchical model
                        weights_in_1_noise = pm.Deterministic('weights_in_1_noise', 
                                                              weights_in_1_raw_noise * weights_in_1_grp_sd_noise + weights_in_1_grp_noise)
        
                        weights_1_out_raw_noise = pm.Normal('w_1_out_noise', 
                                                      shape=(self.gender_num, self.site_num, n_hidden))
                        weights_1_out_noise = pm.Deterministic('weights_1_out_noise', 
                                                               weights_1_out_raw_noise * weights_1_out_grp_sd_noise + weights_1_out_grp_noise)
                        
                        # Build neural-network using tanh activation function
                        act_1_noise = pm.math.tanh(self.a *  weights_in_1_noise[(self.g, self.s)].squeeze())
                        sigma_y = pm.math.sum(act_1_noise * weights_1_out_noise[(self.g, self.s)].squeeze(), axis=1)
                        sigma_y = pm.Deterministic('sigma_y',pm.math.log(1 + pm.math.exp(sigma_y)) + 1e-3)
                                
                else:
                    sigma_noise = pm.Uniform('sigma_noise', lower=0, upper=100, shape=(self.gender_num,self.site_num))
                    sigma_y = sigma_noise[(self.g, self.s)]
            else:
                sigma_noise = pm.Uniform('sigma_noise', lower=0, upper=100)
                sigma_y = sigma_noise
            
            y_like = pm.Normal('y_like', mu=y_hat, sigma=sigma_y, observed=y)
            
        self.model = model

    def estimate(self):
        """ Function to estimate the model """
        with self.model:
            if self.model_type == 'nn':
                self.trace = pm.sample(1000, tune=1000, chains=2,  target_accept=0.9, init='advi+adapt_diag')
            else:    
                self.trace = pm.sample(1000, tune=1000, chains=2,  target_accept=0.9)
        return self.trace

    def predict(self, age, site_id, gender, pred = None):
        """ Function to make predictions from the model """
        
        samples = 1000
        if pred == 'single':
            if self.model_type == 'nn':
                age = np.expand_dims(age ,axis = 1)
                age = np.repeat(age, self.configs['nn_hidden_neuron_num'], axis = 1)
                #self.a = theano.shared(age)
                #self.s = theano.shared(site_id)
                #self.g = theano.shared(gender)
            with self.model:
                self.a.set_value(age)
                self.s.set_value(site_id)
                self.g.set_value(gender)
                ppc = pm.sample_posterior_predictive(self.trace, samples=samples, progressbar=True) 
            
            pred_mean = ppc['y_like'].mean(axis=0)
            pred_var = ppc['y_like'].var(axis=0)
        elif pred == 'group':
            temp = np.zeros([len(age), self.trace.nchains * samples])
            for i in range(temp.shape[0]):
                if (self.model_type == 'linear'):
                    temp[i,:] = age[i] * self.trace['mu_prior_slope'] + self.trace['mu_prior_intercept']
                elif self.model_type == 'poly2':
                    temp[i,:] = age[i]**2 * self.trace['mu_prior_slope_2'] + age[i] * self.trace['mu_prior_slope_1'] + self.trace['mu_prior_intercept']
                elif self.model_type == 'nn':
                    raise NotImplementedError("To be implemented")
                    #act_1 = np.tanh(age[i] * self.trace['w_in_1_grp'])
                    #for j in range(self.trace.nchains * samples):
                    #    temp[i,j] = np.dot(np.squeeze(act_1[j,:,:]), self.trace['w_1_out_grp'][j,:])
                    
            pred_mean = temp.mean(axis=1)
            pred_var = temp.var(axis=1)
        
        print('Done!')
        return pred_mean, pred_var

    def estimate_on_new_site(self, age, sites, gender, y):
        """ Function to adapt the model """
        configs = self.configs
        site_num = len(np.unique(sites))
        gender_num = len(np.unique(gender))
        self.s1 = theano.shared(sites)
        self.g1 = theano.shared(gender)
        self.a1 = theano.shared(age)
        with pm.Model() as new_model:
            if self.model_type == 'linear':    
                # Priors
                mu_prior_intercept = from_posterior('mu_prior_intercept', 
                                                    self.trace['mu_prior_intercept'], 
                                                    distribution='normal', freedom=1)
                sigma_prior_intercept = from_posterior('sigma_prior_intercept', 
                                                       self.trace['sigma_prior_intercept'], 
                                                       distribution='hcauchy', freedom=1)
                mu_prior_slope = from_posterior('mu_prior_slope', 
                                                self.trace['mu_prior_slope'], 
                                                distribution='normal', freedom=1)
                sigma_prior_slope = from_posterior('sigma_prior_slope', 
                                                   self.trace['sigma_prior_slope'], 
                                                   distribution='hcauchy', freedom=1)
            
                if configs['random_intercept']: # Random intercepts
                    intercepts_offset = pm.Normal('intercepts_offset', mu=0, sd=1, 
                                                  shape=(gender_num, site_num))
                else:
                    intercepts_offset = pm.Normal('intercepts_offset', mu=0, sd=1)
                    
                intercepts = pm.Deterministic('intercepts', mu_prior_intercept + 
                                              intercepts_offset * sigma_prior_intercept)
                
                if configs['random_slope']:  # Random slopes
                    slopes_offset = pm.Normal('slopes_offset', mu=0, sd=1, 
                                             shape=(gender_num, site_num))
                else:
                    slopes_offset = pm.Normal('slopes_offset', mu=0, sd=1)
                
                slopes = pm.Deterministic('slopes', mu_prior_slope + 
                                                  slopes_offset * sigma_prior_slope)
                
                if (not configs['random_intercept'] and not configs['random_slope']):
                    y_hat = intercepts + self.a1 * slopes
                elif (configs['random_intercept'] and not configs['random_slope']):
                    y_hat = intercepts[(self.g1,self.s1)] + self.a1  * slopes
                elif (not configs['random_intercept'] and configs['random_slope']):
                    y_hat = intercepts + self.a1 * slopes[(self.g1,self.s1)]
                elif (configs['random_intercept'] and configs['random_slope']):
                    y_hat = intercepts[(self.g1,self.s1)] + self.a1 * slopes[(self.g1,self.s1)]
                    
            elif self.model_type == 'poly2':
                # Priors
                mu_prior_intercept = from_posterior('mu_prior_intercept', 
                                                    self.trace['mu_prior_intercept'],
                                                    distribution='normal', freedom=1)
                sigma_prior_intercept = from_posterior('sigma_prior_intercept', 
                                                       self.trace['sigma_prior_intercept'], 
                                                       distribution='hcauchy', freedom=1)
                mu_prior_slope_1 = from_posterior('mu_prior_slope_1', 
                                                  self.trace['mu_prior_slope_1'],
                                                  distribution='normal', freedom=1)
                sigma_prior_slope_1 = from_posterior('sigma_prior_slope_1', 
                                                     self.trace['sigma_prior_slope_1'], 
                                                     distribution='hcauchy', freedom=1)
                mu_prior_slope_2 = from_posterior('mu_prior_slope_2', 
                                                  self.trace['mu_prior_slope_2'],
                                                  distribution='normal', freedom=1)
                sigma_prior_slope_2 = from_posterior('sigma_prior_slope_2', 
                                                     self.trace['sigma_prior_slope_2'], 
                                                     distribution='hcauchy', freedom=1)
                
                if configs['random_intercept']: # Random intercepts
                    intercepts_offset = pm.Normal('intercepts_offset', mu=0, sd=1,
                                                  shape=(gender_num,site_num))
                else:
                    intercepts_offset = pm.Normal('intercepts_offset', mu=0, sd=1)
                    
                intercepts = pm.Deterministic('intercepts', mu_prior_intercept + 
                                              intercepts_offset * sigma_prior_intercept)
                
                if configs['random_slope']:  # Random slopes
                    slopes_offset1 = pm.Normal('slopes_offset1', mu=0, sd=1, 
                                             shape=(gender_num,site_num))
                    slopes_offset2 = pm.Normal('slopes_offset2', mu=0, sd=1, 
                                             shape=(gender_num,site_num))
                else:
                    slopes_offset1 = pm.Normal('slopes_offset1', mu=0, sd=1)
                    slopes_offset2 = pm.Normal('slopes_offset2', mu=0, sd=1)
                
                slopes_1 = pm.Deterministic('slopes_1', mu_prior_slope_1 + 
                                                  slopes_offset1 * sigma_prior_slope_1)
                slopes_2 = pm.Deterministic('slopes_2', mu_prior_slope_2 + 
                                                  slopes_offset2 * sigma_prior_slope_2)
                
                if (not configs['random_intercept'] and not configs['random_slope']):
                    y_hat = intercepts + self.a1 * slopes_1 + self.a1**2 * slopes_2
                elif (configs['random_intercept'] and not configs['random_slope']):
                    y_hat = intercepts[(self.g1,self.s1)] + self.a1 * slopes_1 + self.a1**2 * slopes_2
                elif (not configs['random_intercept'] and configs['random_slope']):
                    y_hat = intercepts + self.a1 * slopes_1[(self.g1,self.s1)]  + self.a1**2 * slopes_2[(self.g1,self.s1)] 
                elif (configs['random_intercept'] and configs['random_slope']):
                    y_hat = intercepts[(self.g1,self.s1)] + self.a1 * slopes_1[(self.g1,self.s1)] + self.a1**2 * slopes_2[(self.g1,self.s1)] 
                    
            if configs['random_noise']:  # Random Noise
                if configs['hetero_noise']:
                    if self.noise_model == 'linear':
                        # Priors
                        mu_prior_intercept_noise = from_posterior('mu_prior_intercept_noise', 
                                                                  self.trace['mu_prior_intercept_noise'], 
                                                                  distribution='hnormal', freedom=1)
                        sigma_prior_intercept_noise = from_posterior('sigma_prior_intercept_noise', 
                                                                     self.trace['sigma_prior_intercept_noise'], 
                                                                     distribution='hcauchy', freedom=1)
                        mu_prior_slope_noise = from_posterior('mu_prior_slope_noise', 
                                                              self.trace['mu_prior_slope_noise'],
                                                              distribution='normal', freedom=1)
                        sigma_prior_slope_noise = from_posterior('sigma_prior_slope_noise', 
                                                                 self.trace['sigma_prior_slope_noise'], 
                                                                 distribution='hcauchy', freedom=1)
                        
                        if configs['random_intercept']: # Random intercepts
                            intercepts_noise_offset = pm.HalfNormal('intercepts_noise_offset',
                                                                sd=1, shape=(gender_num,site_num))
                        else:
                            intercepts_noise_offset = pm.HalfNormal('intercepts_noise_offset',
                                                                sd=1)
                            
                        intercepts_noise = pm.Deterministic('intercepts_noise', mu_prior_intercept_noise + 
                                                  intercepts_noise_offset * sigma_prior_intercept_noise)
                        
                        if configs['random_slope']:  # Random slopes
                            slopes_noise_offset = pm.Normal('slopes_noise_offset', mu=0, sd=1, 
                                             shape=(gender_num,site_num))
                        else:
                            slopes_noise_offset = pm.Normal('slopes_noise_offset', mu=0, sd=1)
                            
                        slopes_noise = pm.Deterministic('slopes_noise', mu_prior_slope_noise + 
                                                  slopes_noise_offset * sigma_prior_slope_noise)
                        
                        if (not configs['random_intercept'] and not configs['random_slope']):
                            sigma_noise = intercepts_noise + self.a1 * slopes_noise
                        elif (configs['random_intercept'] and not configs['random_slope']):
                            sigma_noise = intercepts_noise[(self.g1,self.s1)] + self.a1 * slopes_noise
                        elif (not configs['random_intercept'] and configs['random_slope']):
                            sigma_noise = intercepts_noise + self.a1 * slopes_noise[(self.g1,self.s1)]
                        elif (configs['random_intercept'] and configs['random_slope']):
                            sigma_noise = intercepts_noise[(self.g1,self.s1)] + self.a1 * slopes_noise[(self.g1,self.s1)]
                      
                        sigma_y = pm.Deterministic('sigma_y', pm.math.log(1 + pm.math.exp(sigma_noise))+1e-3)
                        
                    elif self.noise_model == 'poly2':
                        mu_prior_intercept_noise = from_posterior('mu_prior_intercept_noise', 
                                                                  self.trace['mu_prior_intercept_noise'], 
                                                                  distribution='hnormal', freedom=1)
                        sigma_prior_intercept_noise = from_posterior('sigma_prior_intercept_noise', 
                                                                     self.trace['sigma_prior_intercept_noise'], 
                                                                     distribution='hcauchy', freedom=1)
                        mu_prior_slope_1_noise = from_posterior('mu_prior_slope_1_noise', 
                                                                self.trace['mu_prior_slope_1_noise'],
                                                                distribution='normal', freedom=1)
                        sigma_prior_slope_1_noise = from_posterior('sigma_prior_slope_1_noise', 
                                                                   self.trace['sigma_prior_slope_1_noise'], 
                                                                   distribution='hcauchy', freedom=1)
                        mu_prior_slope_2_noise = from_posterior('mu_prior_slope_2_noise', 
                                                                self.trace['mu_prior_slope_2_noise'],
                                                                distribution='normal', freedom=1)
                        sigma_prior_slope_2_noise = from_posterior('sigma_prior_slope_2_noise', 
                                                                   self.trace['sigma_prior_slope_2_noise'], 
                                                                  distribution='hcauchy', freedom=1)
                        
                        if configs['random_intercept']: # Random intercepts
                            intercepts_noise_offset = pm.HalfNormal('intercepts_noise_offset', sd=1,
                                                  shape=(gender_num,site_num))
                        else:
                            intercepts_noise_offset = pm.HalfNormal('intercepts_noise_offset', sd=1)
                            
                        intercepts_noise = pm.Deterministic('intercepts_noise', mu_prior_intercept_noise + 
                                              intercepts_noise_offset * sigma_prior_intercept_noise)
                        
                        if configs['random_slope']:  # Random slopes
                            slopes_noise_offset1 = pm.Normal('slopes_noise_offset1', mu=0, sd=1, 
                                             shape=(gender_num,site_num))
                            slopes_noise_offset2 = pm.Normal('slopes_noise_offset2', mu=0, sd=1, 
                                             shape=(gender_num,site_num))
                        else:
                            slopes_noise_offset1 = pm.Normal('slopes_noise_offset1', mu=0, sd=1)
                            slopes_noise_offset2 = pm.Normal('slopes_noise_offset2', mu=0, sd=1)
                        
                        slopes_1_noise = pm.Deterministic('slopes_1_noise', mu_prior_slope_1_noise + 
                                                  slopes_noise_offset1 * sigma_prior_slope_1_noise)
                        slopes_2_noise = pm.Deterministic('slopes_2_noise', mu_prior_slope_2_noise + 
                                                  slopes_noise_offset2 * sigma_prior_slope_2_noise)
                
                        if (not configs['random_intercept'] and not configs['random_slope']):
                            sigma_noise = intercepts_noise + self.a1 * slopes_1_noise + self.a1**2 * slopes_2_noise
                        elif (configs['random_intercept'] and not configs['random_slope']):
                            sigma_noise = intercepts_noise[(self.g1,self.s1)] + self.a1 * slopes_1_noise + self.a1**2 * slopes_2_noise
                        elif (not configs['random_intercept'] and configs['random_slope']):
                            sigma_noise = intercepts_noise + self.a1 * slopes_1_noise[(self.g1,self.s1)] + self.a1**2 * slopes_2_noise[(self.g1,self.s1)]
                        elif (configs['random_intercept'] and configs['random_slope']):
                            sigma_noise = intercepts_noise[(self.g1,self.s1)] + self.a1 * slopes_1_noise[(self.g1,self.s1)] + self.a1**2 * slopes_2_noise[(self.g1,self.s1)]
                            
                        sigma_y = pm.Deterministic('sigma_y', pm.math.log(1 + pm.math.exp(sigma_noise))+1e-3)        
                else:
                    sigma_noise = pm.Uniform('sigma_noise', lower=0, upper=100, shape=(gender_num,site_num))
                    sigma_y = sigma_noise[(self.g1,self.s1)]
            else:
                sigma_noise = pm.Uniform('sigma_noise', lower=0, upper=100)
                sigma_y = sigma_noise
            
            y_like = pm.Normal('y_like', mu=y_hat, sigma=sigma_y, observed=y)
            
            trace = pm.sample(1000, tune=1000, chains=2,  target_accept=0.9)
        self.new_site_model = new_model
        self.new_site_trace = trace
        return trace
    
    def predict_on_new_site(self, age, sites, gender):
        """ Function to make predictions from the model """
        
        samples = 1000
        model = self.new_site_model
        with model:
            self.a1.set_value(age)
            self.s1.set_value(sites)
            self.g1.set_value(gender)
            ppc = pm.sample_posterior_predictive(self.new_site_trace, 
                                                 samples=samples, progressbar=True) 
            
        pred_mean = ppc['y_like'].mean(axis=0)
        pred_var = ppc['y_like'].var(axis=0)
       
        print('Done!')
        return pred_mean, pred_var