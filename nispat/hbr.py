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

class HBR:
    """Hierarchical Bayesian Regression for normative modeling

    Basic usage::

        model = HBR(age, site_id, gender, mri_voxel)
        trace = model.estimate()
        ys,s2 = model.predict(age, site_id, gender)

    where the variables are

    :param age: N-vector of ages for N subjects
    :param site_id: N-vector of site IDs for N subjects
    :param gender: N-vector of genders for N subjects
    :param mri_voxel: N-vector of one voxel values for N subjects

    :returns: * ys - predictive mean
              * s2 - predictive variance

    Written by S.M. Kia
    """

    def __init__(self, age, site_id, gender, y, model_type = 'lin'):
        self.site_num = len(np.unique(site_id))
        self.a = theano.shared(age)
        self.s = theano.shared(site_id)
        self.g = theano.shared(gender)
        with pm.Model() as model:
            # Priors
            mu_prior_intercept = pm.Normal('mu_prior_intercept', mu=0., sigma=1e5)
            sigma_prior_intercept = pm.HalfCauchy('sigma_prior_intercept', 5)
            mu_prior_slope = pm.Normal('mu_prior_slope', mu=0., sigma=1e5)
            sigma_prior_slope = pm.HalfCauchy('sigma_prior_slope', 5)
            
            # Random intercepts
            intercepts = pm.Normal('intercepts', mu=mu_prior_intercept, sigma=sigma_prior_intercept, shape=self.site_num)
            
            # Expected value
            if model_type == 'lin_rand_int':
                # Random slopes
                slopes = pm.Normal('slopes', mu=mu_prior_slope, sigma=sigma_prior_slope, shape=(2,))
                y_hat = intercepts[self.s] + self.a * slopes[self.g]
                # Model error
                sigma_error = pm.Uniform('sigma_error', lower=0, upper=100)
                sigma_y = sigma_error
            elif model_type == 'lin_rand_int_slp':
                # Random slopes
                slopes = pm.Normal('slopes', mu=mu_prior_slope, sigma=sigma_prior_slope, shape=(2,self.site_num))
                y_hat = intercepts[self.s] + self.a * slopes[self.g, self.s]
                # Model error
                sigma_error = pm.Uniform('sigma_error', lower=0, upper=100)
                sigma_y = sigma_error
            elif model_type == 'lin_rand_int_slp_nse':
                # Random slopes
                slopes = pm.Normal('slopes', mu=mu_prior_slope, sigma=sigma_prior_slope, shape=(2,self.site_num))
                y_hat = intercepts[self.s] + self.a * slopes[self.g, self.s]
                # Model error
                sigma_error = pm.Uniform('sigma_error', lower=0, upper=100, shape=(2,self.site_num))
                sigma_y = sigma_error[self.g, self.s]      
            elif model_type == 'poly2':
                slopes = pm.Normal('slopes', mu=mu_prior_slope, sigma=sigma_prior_slope, shape=(2,self.site_num))
                mu_prior_slope_2 = pm.Normal('mu_prior_slope_2', mu=0., sigma=1e5)
                sigma_prior_slope_2 = pm.HalfCauchy('sigma_prior_slope_2', 5)
                slopes_2 = pm.Normal('slopes_2', mu=mu_prior_slope_2, sigma=sigma_prior_slope_2, shape=(2,self.site_num))
                y_hat = intercepts[self.s] + self.a * slopes[self.g, self.s] + self.a**2 * slopes_2[self.g, self.s]
                # Model error
                sigma_error = pm.Uniform('sigma_error', lower=0, upper=100, shape=(2,self.site_num))
                sigma_y = sigma_error[self.g, self.s]
            else:
                raise ValueError('Unknown method:' + model_type)
            
            
            
            # Data likelihood
            y_like = pm.Normal('y_like', mu=y_hat, sigma=sigma_y, observed=y)
            
        self.model = model

    def estimate(self):
        """ Function to estimate the model """
        with self.model:
            self.trace = pm.sample(1000, tune=1000)
        
        return self.trace

    def predict(self, age, site_id, gender):
        """ Function to make predictions from the model """
        #self.a = theano.shared(age)
        #self.s = theano.shared(site_id)
        #self.g = theano.shared(gender)
        with self.model:
            self.a.set_value(age)
            self.s.set_value(site_id)
            self.g.set_value(gender)
            ppc = pm.sample_posterior_predictive(self.trace, samples=500, progressbar=True) 
        
        # Predicting on training set
        print('Done!')
        pred_mean = ppc['y_like'].mean(axis=0)
        pred_var = ppc['y_like'].var(axis=0)
        
        return pred_mean, pred_var
